// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package ebpfcommon // import "go.opentelemetry.io/obi/pkg/ebpf/common/http"

import (
	"cmp"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"net/http"
	"slices"
	"strings"

	"go.opentelemetry.io/obi/pkg/appolly/app/request"
)

// Bedrock streams (ConverseStream, InvokeModelWithResponseStream) are not SSE
// but AWS event-stream frames: total_len:u32 | headers_len:u32 | prelude_crc:u32 |
// headers | payload | message_crc:u32, big-endian.
const (
	eventStreamContentType = "application/vnd.amazon.eventstream"
	eventStreamPreludeLen  = 12
	eventStreamCRCLen      = 4

	eventStreamHeaderString = 7
	eventStreamHeaderBytes  = 6
)

// eventStreamHeaderSizes is the value size of each fixed-size header type:
// bool true, bool false, byte, short, int, long, (bytes), (string), timestamp, uuid.
var eventStreamHeaderSizes = [...]int{0, 0, 1, 2, 4, 8, -1, -1, 8, 16}

type eventStreamMessage struct {
	eventType string
	payload   []byte
}

func isEventStream(resp *http.Response) bool {
	return strings.HasPrefix(resp.Header.Get("Content-Type"), eventStreamContentType)
}

// decodeEventStream splits a body into its messages. It stops at the first
// frame that does not fit, which is where a truncated capture ends. CRCs are
// not checked: a capture hole already shows as an impossible length.
func decodeEventStream(body []byte) []eventStreamMessage {
	var msgs []eventStreamMessage
	for len(body) >= eventStreamPreludeLen+eventStreamCRCLen {
		total := int(binary.BigEndian.Uint32(body[0:4]))
		headersLen := int(binary.BigEndian.Uint32(body[4:8]))
		if total < eventStreamPreludeLen+eventStreamCRCLen+headersLen || total > len(body) {
			break
		}

		headers := body[eventStreamPreludeLen : eventStreamPreludeLen+headersLen]
		msgs = append(msgs, eventStreamMessage{
			eventType: eventStreamEventType(headers),
			payload:   body[eventStreamPreludeLen+headersLen : total-eventStreamCRCLen],
		})
		body = body[total:]
	}
	return msgs
}

// eventStreamEventType returns the :event-type header, skipping the others.
func eventStreamEventType(headers []byte) string {
	for len(headers) > 0 {
		nameLen := int(headers[0])
		if len(headers) < 1+nameLen+1 {
			return ""
		}
		name := string(headers[1 : 1+nameLen])
		valueType := int(headers[1+nameLen])
		headers = headers[2+nameLen:]

		var value []byte
		switch {
		case valueType == eventStreamHeaderString || valueType == eventStreamHeaderBytes:
			if len(headers) < 2 {
				return ""
			}
			n := int(binary.BigEndian.Uint16(headers))
			if len(headers) < 2+n {
				return ""
			}
			value, headers = headers[2:2+n], headers[2+n:]
		case valueType < len(eventStreamHeaderSizes) && len(headers) >= eventStreamHeaderSizes[valueType]:
			headers = headers[eventStreamHeaderSizes[valueType]:]
		default:
			return ""
		}

		if name == ":event-type" {
			return string(value)
		}
	}
	return ""
}

// parseBedrockEventStream rebuilds the response of a Bedrock stream: the
// Converse content blocks of a ConverseStream, or the Anthropic content of an
// InvokeModelWithResponseStream whose chunks wrap Anthropic stream events.
func parseBedrockEventStream(body []byte) request.BedrockResponse {
	msgs := decodeEventStream(body)
	if slices.ContainsFunc(msgs, func(m eventStreamMessage) bool { return m.eventType == "chunk" }) {
		return parseInvokeModelStream(msgs)
	}
	return parseConverseStream(msgs)
}

type converseBlock struct {
	Text    string           `json:"text,omitempty"`
	ToolUse *converseToolUse `json:"toolUse,omitempty"`
}

type converseToolUse struct {
	ToolUseID string          `json:"toolUseId"`
	Name      string          `json:"name"`
	Input     json.RawMessage `json:"input"`
	Type      string          `json:"type,omitempty"`
}

func parseConverseStream(msgs []eventStreamMessage) request.BedrockResponse {
	var resp request.BedrockResponse
	var event struct {
		Role       string `json:"role"`
		Index      int    `json:"contentBlockIndex"`
		StopReason string `json:"stopReason"`
		Start      struct {
			ToolUse *converseToolUse `json:"toolUse"`
		} `json:"start"`
		Delta struct {
			Text    string `json:"text"`
			ToolUse *struct {
				Input string `json:"input"`
			} `json:"toolUse"`
		} `json:"delta"`
		Usage request.BedrockUsage `json:"usage"`
	}

	role := "assistant"
	blocks := map[int]*converseBlock{}
	inputs := map[int][]byte{}
	texts := map[int]*strings.Builder{}
	block := func(i int) *converseBlock {
		if blocks[i] == nil {
			blocks[i] = &converseBlock{}
		}
		return blocks[i]
	}

	for _, m := range msgs {
		event.Start.ToolUse, event.Delta.ToolUse, event.Delta.Text = nil, nil, ""
		if !unmarshalJSON(m.payload, &event) {
			continue
		}
		switch m.eventType {
		case "messageStart":
			role = cmp.Or(event.Role, role)
		case "contentBlockStart":
			if event.Start.ToolUse != nil {
				block(event.Index).ToolUse = event.Start.ToolUse
			}
		case "contentBlockDelta":
			block(event.Index)
			if event.Delta.Text != "" {
				if texts[event.Index] == nil {
					texts[event.Index] = &strings.Builder{}
				}
				texts[event.Index].WriteString(event.Delta.Text)
			}
			if event.Delta.ToolUse != nil {
				inputs[event.Index] = append(inputs[event.Index], event.Delta.ToolUse.Input...)
			}
		case "messageStop":
			resp.StopReasonNova = event.StopReason
		case "metadata":
			resp.Usage = event.Usage
		}
	}

	resp.Output = &request.NovaOutput{Message: &request.NovaMessage{Role: role, Content: converseContent(blocks, texts, inputs)}}
	return resp
}

// converseContent orders the streamed blocks the way a Converse response
// lists them. A tool input cut by a truncated capture becomes empty.
func converseContent(blocks map[int]*converseBlock, texts map[int]*strings.Builder, inputs map[int][]byte) json.RawMessage {
	indexes := make([]int, 0, len(blocks))
	for i := range blocks {
		indexes = append(indexes, i)
	}
	slices.Sort(indexes)

	content := make([]converseBlock, 0, len(indexes))
	for _, i := range indexes {
		b := blocks[i]
		if text := texts[i]; text != nil {
			b.Text = text.String()
		}
		if b.ToolUse != nil {
			b.ToolUse.Input = json.RawMessage("{}")
			if json.Valid(inputs[i]) {
				b.ToolUse.Input = inputs[i]
			}
		}
		content = append(content, *b)
	}

	raw, err := json.Marshal(content)
	if err != nil {
		return nil
	}
	return raw
}

func parseInvokeModelStream(msgs []eventStreamMessage) request.BedrockResponse {
	var anthropic request.AnthropicResponse
	var text strings.Builder
	var toolCalls anthropicStreamToolCalls

	for _, m := range msgs {
		var chunk struct {
			Bytes string `json:"bytes"`
		}
		if m.eventType != "chunk" || !unmarshalJSON(m.payload, &chunk) {
			continue
		}
		data, err := base64.StdEncoding.DecodeString(chunk.Bytes)
		if err != nil {
			continue
		}
		var event AnthropicStreamEvent
		if unmarshalJSON(data, &event) {
			processEvent(event.Type, string(data), &anthropic, &text, &toolCalls)
		}
	}

	return request.BedrockResponse{
		Content:    anthropicStreamContent(text.String(), toolCalls.result()),
		StopReason: anthropic.StopReason,
	}
}

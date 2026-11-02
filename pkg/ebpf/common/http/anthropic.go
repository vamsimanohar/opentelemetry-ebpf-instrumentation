// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package ebpfcommon // import "go.opentelemetry.io/obi/pkg/ebpf/common/http"

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"

	"go.opentelemetry.io/obi/pkg/appolly/app/request"
)

const (
	anthropicMessagesPath = "/v1/messages"
	anthropicCompletePath = "/v1/complete"
)

type anthropicContentBlock struct {
	Type  string          `json:"type"`
	ID    string          `json:"id"`
	Name  string          `json:"name"`
	Input json.RawMessage `json:"input,omitempty"`
}

// anthropicServerToolBlocks are tool calls Anthropic executes itself; their
// results come back in the same response.
var anthropicServerToolBlocks = map[string]struct{}{
	"server_tool_use": {},
	"mcp_tool_use":    {},
}

// anthropicToolCall converts a tool block to a tool call, reporting whether the
// block is a tool call at all.
func anthropicToolCall(block *anthropicContentBlock, index int) (request.ToolCall, bool) {
	_, serverSide := anthropicServerToolBlocks[block.Type]
	if (block.Type != "tool_use" && !serverSide) || block.Name == "" {
		return request.ToolCall{}, false
	}

	return request.ToolCall{
		ID:         block.ID,
		Name:       block.Name,
		Arguments:  string(block.Input),
		Type:       "function",
		Index:      index,
		ServerSide: serverSide,
	}, true
}

func extractAnthropicToolCalls(content json.RawMessage) []request.ToolCall {
	if len(content) == 0 {
		return nil
	}

	var blocks []anthropicContentBlock
	if err := json.Unmarshal(content, &blocks); err != nil {
		return nil
	}

	var result []request.ToolCall
	for i := range blocks {
		if tc, ok := anthropicToolCall(&blocks[i], len(result)); ok {
			result = append(result, tc)
		}
	}
	return result
}

func isAnthropic(hdr http.Header) bool {
	isAnthropic := false
	for _, header := range []string{
		"Anthropic-Organization-Id",
		"Anthropic-Ratelimit-Input-Tokens-Remaining",
		"Anthropic-Ratelimit-Output-Tokens-Limit",
		"Anthropic-Ratelimit-Input-Tokens-Limit",
		"Anthropic-Ratelimit-Requests-Limit",
	} {
		if val := hdr.Get(header); val != "" {
			isAnthropic = true
			break
		}
	}

	// we do this extra check because for errors they don't
	// send the usual Anthropic headers
	if !isAnthropic {
		for _, v := range hdr {
			for _, hv := range v {
				if strings.Contains(hv, "api.anthropic.com") {
					isAnthropic = true
					break
				}
			}
		}
	}

	return isAnthropic
}

func looksLikeAnthropicBody(reqB, respB []byte) bool {
	if extractJSONRawField(reqB, "anthropic_version") != nil {
		return false
	}

	// Response: a message object, or an SSE stream beginning with message_start.
	if extractJSONStringField(respB, "type", responseHeaderSearchWindow) == "message" {
		return true
	}
	if bytes.Contains(respB, []byte("message_start")) {
		return true
	}

	// Request: a Claude model with a messages array.
	return strings.HasPrefix(strings.ToLower(extractModelField(reqB)), "claude") &&
		extractJSONRawField(reqB, "messages") != nil
}

func AnthropicSpan(baseSpan *request.Span, req *http.Request, resp *http.Response) (request.Span, bool) {
	isAnthropic := isAnthropic(resp.Header)
	maybeAnthropic := false

	if !isAnthropic {
		if !isHTTP2Request(req) || !strings.HasPrefix(baseSpan.Path, "/v1/messages") {
			return *baseSpan, false
		}
		maybeAnthropic = true
	}

	reqB, ok := readHTTPRequestBody("AnthropicSpan", req, baseSpan)
	if !ok {
		return *baseSpan, false
	}

	respB, ok := readHTTPResponseBody("AnthropicSpan", resp, baseSpan)
	if !ok {
		return *baseSpan, false
	}

	if maybeAnthropic {
		if !looksLikeAnthropicBody(reqB, respB) {
			return *baseSpan, false
		}
	}

	return anthropicSpan(baseSpan, req, reqB, respB), true
}

func anthropicSpan(baseSpan *request.Span, req *http.Request, reqB, respB []byte) request.Span {
	slog.Debug("Anthropic", "request", string(reqB), "response", string(respB))

	parsedRequest := parseAnthropicRequest(reqB)

	var parsedResponse request.AnthropicResponse
	var toolCalls []request.ToolCall
	if looksLikeJSON(respB) {
		parsedResponse = parseAnthropicResponse(respB)
		toolCalls = extractAnthropicToolCalls(parsedResponse.Content)
	} else {
		reader := bytes.NewReader(respB)
		streamResponse, tc, err := parseAnthropicStream(reader)
		if err != nil {
			slog.Debug("failed to parse complete Anthropic stream, continuing with partial fields", "error", err)
		}
		if streamResponse != nil {
			parsedResponse = *streamResponse
			toolCalls = tc
		}
	}

	parsedResponse.Type = anthropicOperation(req)

	baseSpan.SubType = request.HTTPSubtypeAnthropic
	baseSpan.GenAI = &request.GenAI{
		Anthropic: &request.VendorAnthropic{
			Input:     parsedRequest,
			Output:    parsedResponse,
			ToolCalls: toolCalls,
		},
	}

	return *baseSpan
}

// anthropicOperation names the operation from the request path, which is the
// only part of the exchange that names it consistently. A response body reports
// `message`/`completion` when the call succeeded, `error` when it failed, and
// nothing at all when it was truncated out of the capture buffer, so reading it
// would split one endpoint across several operation names.
//
// The endpoint is matched as a path suffix so a deployment mounted under a
// prefix still resolves, while the sub-resources of the Messages API
// (/v1/messages/count_tokens, /v1/messages/batches) stay out of `chat`:
// they are endpoints of their own, not a Messages API call, and semconv has no
// operation for them.
func anthropicOperation(req *http.Request) string {
	switch path := strings.TrimSuffix(requestPath(req), "/"); {
	case strings.HasSuffix(path, anthropicMessagesPath):
		return request.ChatOperationName
	case strings.HasSuffix(path, anthropicCompletePath):
		return request.CompletionOperationName
	}

	return request.OtherOperationName
}

// AnthropicStreamEvent represents different types of streaming events
type AnthropicStreamEvent struct {
	Type string `json:"type"`
}

type MessageStartEvent struct {
	Type    string `json:"type"`
	Message struct {
		Model string `json:"model"`
		ID    string `json:"id"`
		Type  string `json:"type"`
		Role  string `json:"role"`
	} `json:"message"`
}

type ContentBlockDelta struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
	Delta struct {
		Type        string `json:"type"`
		Text        string `json:"text"`
		PartialJSON string `json:"partial_json"`
	} `json:"delta"`
}

type MessageDeltaEvent struct {
	Type  string `json:"type"`
	Delta struct {
		StopReason   string  `json:"stop_reason"`
		StopSequence *string `json:"stop_sequence"`
	} `json:"delta"`
}

// parseAnthropicStream parses the SSE stream from Anthropic API and returns the complete response
func parseAnthropicStream(reader io.Reader) (*request.AnthropicResponse, []request.ToolCall, error) {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 0, 256*1024), 256*1024)
	response := &request.AnthropicResponse{}

	var contentBuilder strings.Builder
	var toolCalls anthropicStreamToolCalls
	var currentEvent string
	var currentData string
	flushEvent := func() {
		eventType, data := currentEvent, currentData
		currentEvent = ""
		currentData = ""
		if eventType == "" || data == "" {
			return
		}
		processEvent(eventType, data, response, &contentBuilder, &toolCalls)
	}

	for scanner.Scan() {
		line := scanner.Text()

		// Skip empty lines (they separate events)
		if line == "" {
			flushEvent()
			continue
		}

		// Parse event line
		if after, ok := strings.CutPrefix(line, "event:"); ok {
			currentEvent = strings.TrimSpace(after)
			continue
		}

		// Parse data line
		if data, ok := extractSSEData(line); ok {
			currentData = strings.TrimSpace(data)
			continue
		}
	}

	flushEvent()

	calls := toolCalls.result()
	response.Content = anthropicStreamContent(contentBuilder.String(), calls)
	if err := scanner.Err(); err != nil {
		return response, calls, fmt.Errorf("error reading stream: %w", err)
	}
	return response, calls, nil
}

// anthropicStreamToolCalls assembles streamed tool_use blocks: the id and name
// arrive in content_block_start, the input as partial JSON in the deltas of
// the same block index.
type anthropicStreamToolCalls struct {
	calls   []request.ToolCall
	inputs  [][]byte
	byIndex map[int]int
}

func (a *anthropicStreamToolCalls) start(blockIndex int, block *anthropicContentBlock) {
	tc, ok := anthropicToolCall(block, len(a.calls))
	if !ok {
		return
	}
	if a.byIndex == nil {
		a.byIndex = map[int]int{}
	}
	a.byIndex[blockIndex] = len(a.calls)
	a.calls = append(a.calls, tc)
	a.inputs = append(a.inputs, nil)
}

func (a *anthropicStreamToolCalls) addInput(blockIndex int, partial string) {
	if i, ok := a.byIndex[blockIndex]; ok {
		a.inputs[i] = append(a.inputs[i], partial...)
	}
}

func (a *anthropicStreamToolCalls) result() []request.ToolCall {
	for i := range a.calls {
		if len(a.inputs[i]) > 0 {
			a.calls[i].Arguments = string(a.inputs[i])
		}
	}
	return a.calls
}

// anthropicStreamContent rebuilds the response content. Text-only streams keep
// the plain text; streams with tool calls get content blocks so the output
// messages carry the tool_call parts too.
func anthropicStreamContent(text string, calls []request.ToolCall) json.RawMessage {
	if len(calls) == 0 {
		return json.RawMessage(text)
	}

	type textBlock struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}

	blocks := make([]any, 0, len(calls)+1)
	if text != "" {
		blocks = append(blocks, textBlock{Type: "text", Text: text})
	}
	for i := range calls {
		if calls[i].ServerSide {
			continue
		}
		input := json.RawMessage(calls[i].Arguments)
		if !json.Valid(input) {
			// a truncated stream leaves the input incomplete
			input = json.RawMessage("{}")
		}
		blocks = append(blocks, anthropicContentBlock{Type: "tool_use", ID: calls[i].ID, Name: calls[i].Name, Input: input})
	}

	content, err := json.Marshal(blocks)
	if err != nil {
		return json.RawMessage(text)
	}
	return content
}

func processEvent(eventType, data string, response *request.AnthropicResponse, contentBuilder *strings.Builder, toolCalls *anthropicStreamToolCalls) {
	switch eventType {
	case "message_start":
		var event MessageStartEvent
		unmarshalJSONBestEffort([]byte(data), &event)
		response.Model = event.Message.Model
		response.ID = event.Message.ID
		response.Role = event.Message.Role
		response.Type = event.Message.Type
		var usage request.AnthropicUsage
		if unmarshalJSONContainerBestEffort([]byte(data), &usage, "message", "usage") {
			response.Usage.Merge(usage)
		}

	case "content_block_delta":
		var event ContentBlockDelta
		unmarshalJSONBestEffort([]byte(data), &event)
		switch event.Delta.Type {
		case "text_delta":
			contentBuilder.WriteString(event.Delta.Text)
		case "input_json_delta":
			toolCalls.addInput(event.Index, event.Delta.PartialJSON)
		}

	case "message_delta":
		var event MessageDeltaEvent
		unmarshalJSONBestEffort([]byte(data), &event)
		response.StopReason = event.Delta.StopReason
		response.StopSequence = event.Delta.StopSequence
		var usage request.AnthropicUsage
		if unmarshalJSONContainerBestEffort([]byte(data), &usage, "usage") {
			response.Usage.Merge(usage)
		}

	case "content_block_start":
		var event struct {
			Index        int                   `json:"index"`
			ContentBlock anthropicContentBlock `json:"content_block"`
		}
		unmarshalJSONBestEffort([]byte(data), &event)
		// the start block's input is a placeholder; the deltas carry the real one
		event.ContentBlock.Input = nil
		toolCalls.start(event.Index, &event.ContentBlock)

	case "ping", "content_block_stop", "message_stop":
		return

	default:
		return
	}
}

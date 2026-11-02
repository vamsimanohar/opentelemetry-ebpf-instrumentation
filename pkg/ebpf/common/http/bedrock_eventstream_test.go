// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package ebpfcommon

import (
	"encoding/base64"
	"encoding/binary"
	"hash/crc32"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"go.opentelemetry.io/obi/pkg/appolly/app/request"
)

// eventStreamFrame encodes one AWS event-stream message with string headers.
func eventStreamFrame(eventType, payload string) []byte {
	var headers []byte
	for _, h := range [][2]string{{":event-type", eventType}, {":content-type", "application/json"}, {":message-type", "event"}} {
		headers = append(headers, byte(len(h[0])))
		headers = append(headers, h[0]...)
		headers = append(headers, eventStreamHeaderString)
		headers = binary.BigEndian.AppendUint16(headers, uint16(len(h[1])))
		headers = append(headers, h[1]...)
	}

	total := eventStreamPreludeLen + len(headers) + len(payload) + eventStreamCRCLen
	frame := binary.BigEndian.AppendUint32(nil, uint32(total))
	frame = binary.BigEndian.AppendUint32(frame, uint32(len(headers)))
	frame = binary.BigEndian.AppendUint32(frame, crc32.ChecksumIEEE(frame))
	frame = append(frame, headers...)
	frame = append(frame, payload...)
	return binary.BigEndian.AppendUint32(frame, crc32.ChecksumIEEE(frame))
}

func eventStreamBody(frames ...[]byte) []byte {
	var body []byte
	for _, f := range frames {
		body = append(body, f...)
	}
	return body
}

func TestParseConverseStream(t *testing.T) {
	body := eventStreamBody(
		eventStreamFrame("messageStart", `{"role":"assistant","p":"abc"}`),
		eventStreamFrame("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"text":"Let me "}}`),
		eventStreamFrame("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"text":"check."}}`),
		eventStreamFrame("contentBlockStop", `{"contentBlockIndex":0}`),
		eventStreamFrame("contentBlockStart", `{"contentBlockIndex":1,"start":{"toolUse":{"toolUseId":"tooluse_1","name":"read_file"}}}`),
		eventStreamFrame("contentBlockDelta", `{"contentBlockIndex":1,"delta":{"toolUse":{"input":"{\"pa"}}}`),
		eventStreamFrame("contentBlockDelta", `{"contentBlockIndex":1,"delta":{"toolUse":{"input":"th\": \"alpha.txt\"}"}}}`),
		eventStreamFrame("contentBlockStop", `{"contentBlockIndex":1}`),
		eventStreamFrame("messageStop", `{"stopReason":"tool_use"}`),
		eventStreamFrame("metadata", `{"usage":{"inputTokens":12,"outputTokens":7,"totalTokens":19}}`),
	)

	resp := parseBedrockEventStream(body)
	assert.Equal(t, "tool_use", resp.StopReasonNova)
	assert.Equal(t, 12, tokenValue(resp.Usage.InputTokens))
	assert.Equal(t, []request.ToolCall{
		{ID: "tooluse_1", Name: "read_file", Arguments: `{"path":"alpha.txt"}`, Type: "function"},
	}, extractBedrockToolCalls(&resp))
	require.NotNil(t, resp.Output)
	assert.Contains(t, string(resp.Output.Message.Content), `{"text":"Let me check."}`)

	// a capture cut inside a frame keeps the frames before it
	truncated := parseBedrockEventStream(body[:len(body)-20])
	require.Len(t, extractBedrockToolCalls(&truncated), 1)
}

func TestParseInvokeModelStream(t *testing.T) {
	chunk := func(event string) []byte {
		return eventStreamFrame("chunk", `{"bytes":"`+base64.StdEncoding.EncodeToString([]byte(event))+`"}`)
	}
	body := eventStreamBody(
		chunk(`{"type":"message_start","message":{"id":"msg_1","role":"assistant","model":"claude"}}`),
		chunk(`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}`),
		chunk(`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"Oslo\"}"}}`),
		chunk(`{"type":"content_block_stop","index":0}`),
		chunk(`{"type":"message_delta","delta":{"stop_reason":"tool_use"}}`),
	)

	resp := parseBedrockEventStream(body)
	assert.Equal(t, "tool_use", resp.StopReason)
	assert.Equal(t, []request.ToolCall{
		{ID: "toolu_1", Name: "get_weather", Arguments: `{"city":"Oslo"}`, Type: "function"},
	}, extractBedrockToolCalls(&resp))
}

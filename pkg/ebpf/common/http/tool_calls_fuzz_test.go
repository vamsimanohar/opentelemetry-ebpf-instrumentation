// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package ebpfcommon

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// The tool-call parsers read captured bodies, which may be cut anywhere or
// hold arbitrary bytes: they must neither panic nor loop.

func FuzzBedrockEventStream(f *testing.F) {
	f.Add(eventStreamFrame("contentBlockStart", `{"contentBlockIndex":0,"start":{"toolUse":{"toolUseId":"t","name":"n"}}}`))
	f.Add(eventStreamFrame("chunk", `{"bytes":"eyJ0eXBlIjoibWVzc2FnZV9zdGFydCJ9"}`))
	f.Add([]byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0, 0, 0, 0})
	f.Fuzz(func(_ *testing.T, b []byte) {
		_ = parseBedrockEventStream(b)
	})
}

func FuzzOpenAIStream(f *testing.F) {
	f.Add([]byte("data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c\",\"function\":{\"name\":\"n\",\"arguments\":\"{\"}}]}}]}\n\n"))
	f.Add([]byte("event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"output\":[{\"type\":\"function_call\",\"call_id\":\"c\",\"name\":\"n\"}]}}\n\n"))
	f.Add([]byte("data: {\"type\":\"response.output_item.done\",\"output_index\":3,\"item\":{\"type\":\"function_call\",\"call_id\":\"c\",\"name\":\"n\"}}\n\n"))
	f.Fuzz(func(_ *testing.T, b []byte) {
		_, _ = parseOpenAIStream(bytes.NewReader(b))
	})
}

func FuzzAnthropicStream(f *testing.F) {
	f.Add([]byte("event: content_block_start\ndata: {\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t\",\"name\":\"n\"}}\n\n" +
		"event: content_block_delta\ndata: {\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"a\"}}\n\n"))
	f.Fuzz(func(_ *testing.T, b []byte) {
		_, _, _ = parseAnthropicStream(bytes.NewReader(b))
	})
}

func FuzzGeminiStream(f *testing.F) {
	f.Add([]byte("data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"id\":\"x\",\"name\":\"n\",\"args\":{}}}]}}]}\n\n"))
	f.Fuzz(func(_ *testing.T, b []byte) {
		_, _ = parseGeminiStream(bytes.NewReader(b))
	})
}

func FuzzResponsesToolCalls(f *testing.F) {
	f.Add([]byte(`[{"type":"function_call","call_id":"c","name":"n","arguments":"{}"}]`))
	f.Fuzz(func(_ *testing.T, b []byte) {
		_ = extractResponsesToolCalls(json.RawMessage(b))
	})
}

func TestConverseStreamTextIsLinear(t *testing.T) {
	var body bytes.Buffer
	deltas := 0
	for body.Len() < 256<<10 {
		body.Write(eventStreamFrame("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"text":"abcdefghij"}}`))
		deltas++
	}

	start := time.Now()
	resp := parseBedrockEventStream(body.Bytes())
	elapsed := time.Since(start)

	assert.Contains(t, string(resp.Output.Message.Content), strings.Repeat("abcdefghij", deltas))
	assert.Less(t, elapsed, time.Second)
}

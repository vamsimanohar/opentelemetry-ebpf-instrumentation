// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package request

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestToolResults(t *testing.T) {
	tests := []struct {
		name  string
		genAI GenAI
		want  []ToolResult
	}{
		{
			name: "chat replays history; only the trailing turn counts",
			genAI: GenAI{OpenAI: &VendorOpenAI{Request: OpenAIInput{Messages: json.RawMessage(`[
				{"role":"user","content":"go"},
				{"role":"assistant","tool_calls":[{"id":"call_old","type":"function","function":{"name":"a","arguments":"{}"}}]},
				{"role":"tool","tool_call_id":"call_old","content":"old"},
				{"role":"assistant","tool_calls":[{"id":"call_1"},{"id":"call_2"}]},
				{"role":"tool","tool_call_id":"call_1","content":"one"},
				{"role":"tool","tool_call_id":"call_2","content":"two"}]`)}}},
			want: []ToolResult{
				{ID: "call_1", Index: 0, Raw: json.RawMessage(`"one"`)},
				{ID: "call_2", Index: 1, Raw: json.RawMessage(`"two"`)},
			},
		},
		{
			name: "chat results without ids keep the tool name",
			genAI: GenAI{OpenAI: &VendorOpenAI{Request: OpenAIInput{Messages: json.RawMessage(`[
				{"role":"assistant","tool_calls":[{"id":null,"function":{"name":"read_file"}}]},
				{"role":"tool","name":"read_file","content":"x"},
				{"role":"tool","content":"y"}]`)}}},
			want: []ToolResult{
				{Name: "read_file", Index: 0, Raw: json.RawMessage(`"x"`)},
				{Index: 1, Raw: json.RawMessage(`"y"`)},
			},
		},
		{
			name: "ollama names the tool",
			genAI: GenAI{Ollama: &VendorOpenAI{Request: OpenAIInput{Messages: json.RawMessage(`[
				{"role":"assistant","tool_calls":[{"function":{"name":"get_temperature"}}]},
				{"role":"tool","tool_name":"get_temperature","content":"22"}]`)}}},
			want: []ToolResult{{Name: "get_temperature", Raw: json.RawMessage(`"22"`)}},
		},
		{
			name: "chat request with no tool turn",
			genAI: GenAI{OpenAI: &VendorOpenAI{Request: OpenAIInput{Messages: json.RawMessage(`[
				{"role":"user","content":"hello"}]`)}}},
		},
		{
			name: "responses stateless replay pairs on call_id",
			genAI: GenAI{OpenAI: &VendorOpenAI{Request: OpenAIInput{InputItems: json.RawMessage(`[
				{"role":"user","content":"go"},
				{"type":"function_call","id":"fc_0","call_id":"call_0","name":"a","arguments":"{}"},
				{"type":"function_call_output","call_id":"call_0","output":"old"},
				{"type":"function_call","id":"fc_1","call_id":"call_1","name":"a","arguments":"{}"},
				{"type":"function_call_output","call_id":"call_1","output":"new"}]`)}}},
			want: []ToolResult{{ID: "call_1", Raw: json.RawMessage(`"new"`)}},
		},
		{
			name: "responses stateful continuation carries only outputs",
			genAI: GenAI{OpenAI: &VendorOpenAI{Request: OpenAIInput{PreviousResponseID: "resp_1", InputItems: json.RawMessage(`[
				{"type":"function_call_output","call_id":"call_1","output":"a"},
				{"type":"custom_tool_call_output","call_id":"call_2","output":"b"}]`)}}},
			want: []ToolResult{
				{ID: "call_1", Index: 0, Raw: json.RawMessage(`"a"`)},
				{ID: "call_2", Index: 1, Raw: json.RawMessage(`"b"`)},
			},
		},
		{
			name: "anthropic is_error",
			genAI: GenAI{Anthropic: &VendorAnthropic{Input: AnthropicRequest{Messages: json.RawMessage(`[
				{"role":"user","content":"go"},
				{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"a","input":{}},{"type":"tool_use","id":"toolu_2","name":"b","input":{}}]},
				{"role":"user","content":[
					{"type":"tool_result","tool_use_id":"toolu_1","content":"ok"},
					{"type":"tool_result","tool_use_id":"toolu_2","content":"boom","is_error":true},
					{"type":"text","text":"continue"}]}]`)}}},
			want: []ToolResult{
				{ID: "toolu_1", Index: 0, Raw: json.RawMessage(`"ok"`)},
				{ID: "toolu_2", Index: 1, IsError: true, Raw: json.RawMessage(`"boom"`)},
			},
		},
		{
			name: "gemini function responses with and without ids",
			genAI: GenAI{Gemini: &VendorGemini{Input: GeminiRequest{Contents: json.RawMessage(`[
				{"role":"user","parts":[{"text":"go"}]},
				{"role":"model","parts":[{"functionCall":{"id":"fc-1","name":"a"}},{"functionCall":{"name":"b"}}]},
				{"role":"user","parts":[
					{"functionResponse":{"id":"fc-1","name":"a","response":{"output":"x"}}},
					{"functionResponse":{"name":"b","response":{"error":"failed"}}}]}]`)}}},
			want: []ToolResult{
				{ID: "fc-1", Name: "a", Index: 0, Raw: json.RawMessage(`{"output":"x"}`)},
				{Name: "b", Index: 1, IsError: true, Raw: json.RawMessage(`{"error":"failed"}`)},
			},
		},
		{
			name: "bedrock converse status error",
			genAI: GenAI{Bedrock: &VendorBedrock{Input: BedrockRequest{Messages: json.RawMessage(`[
				{"role":"user","content":[{"text":"go"}]},
				{"role":"assistant","content":[{"toolUse":{"toolUseId":"tooluse_1","name":"a","input":{}}}]},
				{"role":"user","content":[{"toolResult":{"toolUseId":"tooluse_1","content":[{"text":"no"}],"status":"error"}}]}]`)}}},
			want: []ToolResult{{ID: "tooluse_1", IsError: true, Raw: json.RawMessage(`[{"text":"no"}]`)}},
		},
		{
			name: "bedrock invoke model with an anthropic body",
			genAI: GenAI{Bedrock: &VendorBedrock{Input: BedrockRequest{Messages: json.RawMessage(`[
				{"role":"assistant","content":[{"type":"tool_use","id":"toolu_b","name":"a","input":{}}]},
				{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_b","content":"fine"}]}]`)}}},
			want: []ToolResult{{ID: "toolu_b", Raw: json.RawMessage(`"fine"`)}},
		},
		{
			name: "truncated messages yield nothing",
			genAI: GenAI{OpenAI: &VendorOpenAI{Request: OpenAIInput{Messages: json.RawMessage(`[
				{"role":"assistant","tool_calls":[{"id":"call_1"}]},{"role":"tool","tool_call_id":"ca`)}}},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, tc.genAI.ToolResults())
		})
	}
}

func TestToolCallsAccessor(t *testing.T) {
	calls := []ToolCall{{ID: "call_1", Name: "a"}}

	for _, genAI := range []GenAI{
		{OpenAI: &VendorOpenAI{ToolCalls: calls}},
		{Qwen: &VendorOpenAI{ToolCalls: calls}},
		{OpenAICompatible: &VendorOpenAI{ToolCalls: calls}},
		{Anthropic: &VendorAnthropic{ToolCalls: calls}},
		{Gemini: &VendorGemini{ToolCalls: calls}},
		{Bedrock: &VendorBedrock{ToolCalls: calls}},
	} {
		assert.Equal(t, calls, genAI.ToolCalls())
	}

	g := GenAI{OpenAI: &VendorOpenAI{ID: "resp_2", Request: OpenAIInput{PreviousResponseID: "resp_1"}}}
	assert.Equal(t, "resp_2", g.ResponseID())
	assert.Equal(t, "resp_1", g.PreviousResponseID())
}

func TestAgentToolSpanNameAndStatus(t *testing.T) {
	tool := Span{Type: EventTypeGenAIToolExec, InferredTool: &InferredTool{Name: "read_file"}}
	assert.Equal(t, "execute_tool read_file", tool.TraceName())
	assert.Equal(t, StatusCodeUnset, SpanStatusCode(&tool))

	tool.InferredTool.IsError = true
	assert.Equal(t, StatusCodeError, SpanStatusCode(&tool))
	assert.Equal(t, ErrorTypeOther, SpanErrorType(&tool))

	window := Span{Type: EventTypeGenAIToolWindow, InferredTool: &InferredTool{ParallelCount: 2}}
	assert.Equal(t, ToolWindowSpanName, window.TraceName())

	proc := Span{Type: EventTypeProcessExec, Process: &ObservedProcess{ExecutableName: "uname", Exited: true, ExitCode: 1}}
	assert.Equal(t, "uname", proc.TraceName())
	assert.Equal(t, StatusCodeError, SpanStatusCode(&proc))
	require.False(t, proc.IsClientSpan(), "process spans are not network client spans")
}

func TestGeminiErrorMember(t *testing.T) {
	for raw, isError := range map[string]bool{
		`null`:                 false,
		`false`:                false,
		`""`:                   false,
		`{}`:                   false,
		`{ }`:                  false,
		`"failed"`:             true,
		`{"message":"denied"}`: true,
	} {
		g := &GenAI{Gemini: &VendorGemini{}}
		g.Gemini.Input.Contents = json.RawMessage(`[{"role":"model","parts":[{"functionCall":{"name":"f"}}]},
			{"role":"user","parts":[{"functionResponse":{"name":"f","response":{"result":"ok","error":` + raw + `}}}]}]`)
		res := g.ToolResults()
		require.Len(t, res, 1)
		assert.Equal(t, isError, res[0].IsError, raw)
	}
}

func TestRequestHistory(t *testing.T) {
	chat := GenAI{OpenAI: &VendorOpenAI{Request: OpenAIInput{Messages: json.RawMessage(`[{"role":"user"}]`)}}}
	assert.JSONEq(t, `[{"role":"user"}]`, string(chat.RequestHistory()))

	responses := GenAI{OpenAI: &VendorOpenAI{Request: OpenAIInput{InputItems: json.RawMessage(`[{"type":"message"}]`)}}}
	assert.JSONEq(t, `[{"type":"message"}]`, string(responses.RequestHistory()))

	cut := GenAI{Anthropic: &VendorAnthropic{}, UnparsedRequest: []byte(`{"model":"m","messages" : [{"role":"user","content":"hel`)}
	assert.Equal(t, `[{"role":"user","content":"hel`, string(cut.RequestHistory()))
}

// FuzzToolResults checks that the trailing turn splits arrays the way
// encoding/json reads them, and never panics on captured bytes.
func FuzzToolResults(f *testing.F) {
	f.Add([]byte(`[{"role":"assistant"},{"role":"tool","tool_call_id":"x","content":"y"}]`), uint8(0))
	f.Add([]byte(`[{"a":"]\"["}, [1, {"b": [2]}], "s\\", 3, null]`), uint8(0))
	f.Add([]byte(` [ ] `), uint8(1))
	f.Fuzz(func(t *testing.T, b []byte, which uint8) {
		elems, ok := jsonArrayElements(b)
		var want []json.RawMessage
		if bytes.HasPrefix(bytes.TrimLeft(b, " \t\r\n"), []byte("[")) && json.Unmarshal(b, &want) == nil {
			require.True(t, ok, "a valid array splits")
			require.Len(t, elems, len(want))
			for i := range want {
				assert.JSONEq(t, string(want[i]), string(elems[i]))
			}
		}

		g := &GenAI{}
		switch which % 4 {
		case 0:
			g.OpenAI = &VendorOpenAI{Request: OpenAIInput{Messages: b}}
		case 1:
			g.OpenAI = &VendorOpenAI{Request: OpenAIInput{InputItems: b}}
		case 2:
			g.Anthropic = &VendorAnthropic{}
			g.Anthropic.Input.Messages = b
		case 3:
			g.Gemini = &VendorGemini{}
			g.Gemini.Input.Contents = b
		}
		_ = g.ToolResults()
		_ = g.RequestHistory()
	})
}

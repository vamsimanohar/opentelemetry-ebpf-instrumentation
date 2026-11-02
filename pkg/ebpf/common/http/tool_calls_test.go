// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package ebpfcommon

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"go.opentelemetry.io/obi/pkg/appolly/app/request"
)

// sseData frames each payload as an SSE data event.
func sseData(payloads ...string) string {
	var b strings.Builder
	for _, p := range payloads {
		b.WriteString("data: " + p + "\n\n")
	}
	return b.String()
}

// sseEvents frames name/payload pairs as named SSE events.
func sseEvents(pairs ...string) string {
	var b strings.Builder
	for i := 0; i+1 < len(pairs); i += 2 {
		b.WriteString("event: " + pairs[i] + "\ndata: " + pairs[i+1] + "\n\n")
	}
	return b.String()
}

func TestToolCallExtraction(t *testing.T) {
	type parsed struct {
		calls  []request.ToolCall
		output string
	}

	openAIChatSSE := sseData(
		`{"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"read_file","arguments":""}}]}}]}`,
		`{"id":"chatcmpl-1","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"run_command","arguments":"{\"cm"}}]}}]}`,
		// fragments of both calls interleave and split inside keys and values
		`{"id":"chatcmpl-1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"pa"}}]}}]}`,
		`{"id":"chatcmpl-1","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"d\": \"uname -a\"}"}}]}}]}`,
		`{"id":"chatcmpl-1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\": \"beta.txt\"}"}}]}}]}`,
		`{"id":"chatcmpl-1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		`[DONE]`,
	)

	responsesJSON := `{"id":"resp_1","object":"response","status":"completed","output":[
		{"type":"reasoning","id":"rs_1"},
		{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"Paris\"}","status":"completed"},
		{"type":"web_search_call","id":"ws_1","status":"completed"},
		{"type":"custom_tool_call","id":"ctc_1","call_id":"call_2","name":"apply","input":"*** patch"},
		{"type":"local_shell_call","id":"lsc_1","call_id":"call_3","action":{"type":"exec","command":["ls"]}}]}`

	responsesItem := `{"type":"function_call","id":"fc_9","call_id":"call_9","name":"get_weather","arguments":"{\"city\":\"Rome\"}","status":"completed"}`
	responsesSSE := sseEvents(
		"response.created", `{"type":"response.created","response":{"id":"resp_9","model":"gpt-4o","output":[]}}`,
		"response.output_item.added", `{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_9","call_id":"call_9","name":"get_weather","arguments":""}}`,
		"response.function_call_arguments.delta", `{"type":"response.function_call_arguments.delta","item_id":"fc_9","output_index":0,"delta":"{\"city\":"}`,
		"response.output_item.done", `{"type":"response.output_item.done","output_index":0,"item":`+responsesItem+`}`,
		"response.completed", `{"type":"response.completed","response":{"id":"resp_9","model":"gpt-4o","output":[`+responsesItem+`]}}`,
	)
	// a capture cut before response.completed still has the finished items
	responsesSSETruncated := responsesSSE[:strings.Index(responsesSSE, "event: response.completed")]

	anthropicJSON := `{"id":"msg_1","type":"message","role":"assistant","stop_reason":"tool_use","content":[
		{"type":"text","text":"Checking."},
		{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"x"}},
		{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"location":"Paris"}}]}`

	anthropicSSE := sseEvents(
		"message_start", `{"type":"message_start","message":{"id":"msg_2","type":"message","role":"assistant","model":"claude"}}`,
		"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"On it."}}`,
		"content_block_stop", `{"type":"content_block_stop","index":0}`,
		"content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_a","name":"read_file","input":{}}}`,
		"ping", `{"type":"ping"}`,
		"content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"pa"}}`,
		"content_block_start", `{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_b","name":"http_get","input":{}}}`,
		"content_block_delta", `{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"url\": \"http://x\"}"}}`,
		"content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"th\": \"beta.txt\"}"}}`,
		"content_block_stop", `{"type":"content_block_stop","index":1}`,
		"content_block_stop", `{"type":"content_block_stop","index":2}`,
		"message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use"}}`,
		"message_stop", `{"type":"message_stop"}`,
	)
	// truncated inside the input of the only tool call
	anthropicSSETruncated := sseEvents(
		"message_start", `{"type":"message_start","message":{"id":"msg_3","type":"message","role":"assistant","model":"claude"}}`,
		"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_t","name":"read_file","input":{}}}`,
		"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\": \"al"}}`,
	)

	geminiJSON := `{"candidates":[{"content":{"role":"model","parts":[
		{"functionCall":{"id":"fc-1","name":"get_weather","args":{"city":"Paris"}},"thoughtSignature":"sig"},
		{"functionCall":{"name":"get_time","args":{}}}]},"finishReason":"STOP"}],"modelVersion":"gemini-3"}`
	geminiSSE := "data: " + `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"id":"fc-2","name":"read_file","args":{"path":"alpha.txt"}}}]},"index":0}]}` + "\r\n\r\n" +
		"data: " + `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"id":"fc-3","name":"run_command","args":{"cmd":"uname -a"}}}]},"finishReason":"STOP","index":0}],"modelVersion":"gemini-3"}` + "\r\n\r\n"

	bedrockConverse := `{"output":{"message":{"role":"assistant","content":[
		{"text":"Let me check."},
		{"toolUse":{"toolUseId":"tooluse_1","name":"get_weather","input":{"city":"Paris"}}},
		{"toolUse":{"toolUseId":"tooluse_2","name":"web_grounding","input":{},"type":"server_tool_use"}}]}},
		"stopReason":"tool_use","usage":{"inputTokens":5,"outputTokens":2,"totalTokens":7}}`
	bedrockInvoke := `{"id":"msg_b","type":"message","role":"assistant","content":[{"type":"tool_use","id":"toolu_bd","name":"get_weather","input":{"city":"Oslo"}}],"stop_reason":"tool_use"}`

	tests := []struct {
		name  string
		parse func() parsed
		want  []request.ToolCall
		// output must contain these strings once normalized
		output []string
	}{
		{
			name: "openai chat json",
			parse: func() parsed {
				resp, calls := parseOpenAICompatibleResponse([]byte(`{"id":"chatcmpl-j","object":"chat.completion","choices":[{"index":0,"finish_reason":"tool_calls",
					"message":{"role":"assistant","content":null,"tool_calls":[
						{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}},
						{"id":"call_2","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"/etc/hostname\"}"}}]}}]}`))
				return parsed{calls: calls, output: resp.GetOutput()}
			},
			want: []request.ToolCall{
				{ID: "call_1", Name: "get_weather", Arguments: `{"city":"Paris"}`, Type: "function", Index: 0},
				{ID: "call_2", Name: "read_file", Arguments: `{"path":"/etc/hostname"}`, Type: "function", Index: 1},
			},
		},
		{
			name: "openai chat sse with interleaved argument fragments",
			parse: func() parsed {
				resp, calls := parseOpenAIStream(strings.NewReader(openAIChatSSE))
				return parsed{calls: calls, output: resp.GetOutput()}
			},
			want: []request.ToolCall{
				{ID: "call_a", Name: "read_file", Arguments: `{"path": "beta.txt"}`, Type: "function", Index: 0},
				{ID: "call_b", Name: "run_command", Arguments: `{"cmd": "uname -a"}`, Type: "function", Index: 1},
			},
			output: []string{`"type":"tool_call"`, `"id":"call_a"`, `"id":"call_b"`, `"finish_reason":"tool_calls"`},
		},
		{
			name: "openai responses json keys on call_id and flags server tools",
			parse: func() parsed {
				resp, calls := parseOpenAICompatibleResponse([]byte(responsesJSON))
				return parsed{calls: calls, output: resp.GetOutput()}
			},
			want: []request.ToolCall{
				{ID: "call_1", Name: "get_weather", Arguments: `{"city":"Paris"}`, Type: "function", Index: 0},
				{ID: "ws_1", Name: "web_search", Type: "web_search_call", Index: 1, ServerSide: true},
				{ID: "call_2", Name: "apply", Arguments: "*** patch", Type: "custom", Index: 2},
				{ID: "call_3", Name: "local_shell", Arguments: `{"type":"exec","command":["ls"]}`, Type: "local_shell", Index: 3},
			},
		},
		{
			name: "openai responses sse",
			parse: func() parsed {
				resp, calls := parseOpenAIStream(strings.NewReader(responsesSSE))
				return parsed{calls: calls, output: resp.GetOutput()}
			},
			want:   []request.ToolCall{{ID: "call_9", Name: "get_weather", Arguments: `{"city":"Rome"}`, Type: "function"}},
			output: []string{`"type":"tool_call"`, `"id":"call_9"`},
		},
		{
			name: "openai responses sse truncated before completion",
			parse: func() parsed {
				resp, calls := parseOpenAIStream(strings.NewReader(responsesSSETruncated))
				return parsed{calls: calls, output: resp.GetOutput()}
			},
			want: []request.ToolCall{{ID: "call_9", Name: "get_weather", Arguments: `{"city":"Rome"}`, Type: "function"}},
		},
		{
			name: "anthropic json",
			parse: func() parsed {
				return parsed{calls: extractAnthropicToolCalls(parseAnthropicResponse([]byte(anthropicJSON)).Content)}
			},
			want: []request.ToolCall{
				{ID: "srvtoolu_1", Name: "web_search", Arguments: `{"query":"x"}`, Type: "function", Index: 0, ServerSide: true},
				{ID: "toolu_1", Name: "get_weather", Arguments: `{"location":"Paris"}`, Type: "function", Index: 1},
			},
		},
		{
			name: "anthropic sse with input split across deltas",
			parse: func() parsed {
				resp, calls, err := parseAnthropicStream(strings.NewReader(anthropicSSE))
				require.NoError(t, err)
				return parsed{calls: calls, output: request.NormalizeAnthropicOutput(resp)}
			},
			want: []request.ToolCall{
				{ID: "toolu_a", Name: "read_file", Arguments: `{"path": "beta.txt"}`, Type: "function", Index: 0},
				{ID: "toolu_b", Name: "http_get", Arguments: `{"url": "http://x"}`, Type: "function", Index: 1},
			},
			output: []string{`"content":"On it."`, `"type":"tool_call","id":"toolu_a"`, `"path":"beta.txt"`, `"id":"toolu_b"`},
		},
		{
			name: "anthropic sse truncated inside the input",
			parse: func() parsed {
				resp, calls, err := parseAnthropicStream(strings.NewReader(anthropicSSETruncated))
				require.NoError(t, err)
				return parsed{calls: calls, output: request.NormalizeAnthropicOutput(resp)}
			},
			want:   []request.ToolCall{{ID: "toolu_t", Name: "read_file", Arguments: `{"path": "al`, Type: "function"}},
			output: []string{`"id":"toolu_t"`},
		},
		{
			name: "gemini json keeps ids and leaves missing ones empty",
			parse: func() parsed {
				var resp request.GeminiResponse
				require.NoError(t, json.Unmarshal([]byte(geminiJSON), &resp))
				return parsed{calls: extractGeminiFunctionCalls(&resp)}
			},
			want: []request.ToolCall{
				{ID: "fc-1", Name: "get_weather", Arguments: `{"city":"Paris"}`, Type: "function", Index: 0},
				{Name: "get_time", Arguments: `{}`, Type: "function", Index: 1},
			},
		},
		{
			name: "gemini sse",
			parse: func() parsed {
				resp, calls := parseGeminiStream(strings.NewReader(geminiSSE))
				return parsed{calls: calls, output: resp.Candidates[0].Content.Role}
			},
			want: []request.ToolCall{
				{ID: "fc-2", Name: "read_file", Arguments: `{"path":"alpha.txt"}`, Type: "function", Index: 0},
				{ID: "fc-3", Name: "run_command", Arguments: `{"cmd":"uname -a"}`, Type: "function", Index: 1},
			},
		},
		{
			name: "bedrock converse",
			parse: func() parsed {
				var resp request.BedrockResponse
				require.NoError(t, json.Unmarshal([]byte(bedrockConverse), &resp))
				return parsed{calls: extractBedrockToolCalls(&resp)}
			},
			want: []request.ToolCall{
				{ID: "tooluse_1", Name: "get_weather", Arguments: `{"city":"Paris"}`, Type: "function", Index: 0},
				{ID: "tooluse_2", Name: "web_grounding", Arguments: `{}`, Type: "function", Index: 1, ServerSide: true},
			},
		},
		{
			name: "bedrock invoke model with an anthropic body",
			parse: func() parsed {
				var resp request.BedrockResponse
				require.NoError(t, json.Unmarshal([]byte(bedrockInvoke), &resp))
				return parsed{calls: extractBedrockToolCalls(&resp)}
			},
			want: []request.ToolCall{{ID: "toolu_bd", Name: "get_weather", Arguments: `{"city":"Oslo"}`, Type: "function"}},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.parse()
			assert.Equal(t, tc.want, got.calls)
			for _, want := range tc.output {
				assert.Contains(t, got.output, want)
			}
		})
	}
}

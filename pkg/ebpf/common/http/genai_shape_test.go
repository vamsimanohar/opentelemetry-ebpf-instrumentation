// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package ebpfcommon

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"go.opentelemetry.io/obi/pkg/appolly/app/request"
	"go.opentelemetry.io/obi/pkg/config"
)

const (
	shapeChatRequest = `{"messages":[{"role":"user","content":"weather in Paris?"}],"model":"meta-llama/Llama-3.1-8B-Instruct",` +
		`"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}]}`
	shapeChatResponse = `{"id":"chatcmpl-0f1e","object":"chat.completion","created":1793168101,"model":"meta-llama/Llama-3.1-8B-Instruct",` +
		`"choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"chatcmpl-tool-9a1",` +
		`"type":"function","function":{"name":"get_weather","arguments":"{\"city\": \"Paris\"}"}}]},"finish_reason":"tool_calls"}],` +
		`"usage":{"prompt_tokens":42,"completion_tokens":17,"total_tokens":59}}`
	shapeChatStream = "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"qwen2.5\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hi\"}}]}\n\n" +
		"data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"qwen2.5\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: [DONE]\n\n"
	shapeResponsesRequest  = `{"model":"gpt-oss-20b","input":[{"role":"user","content":"hello"}]}`
	shapeResponsesResponse = `{"id":"resp_01","object":"response","created_at":1793168101,"status":"completed","model":"gpt-oss-20b",` +
		`"output":[{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"Hi"}]}]}`
	shapeResponsesStream = "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_02\",\"object\":\"response\",\"model\":\"gpt-oss-20b\"}}\n\n"
	shapeMessagesRequest = `{"model":"claude-sonnet-4-5","max_tokens":256,"messages":[{"role":"user","content":"hello"}]}`
	shapeMessagesResult  = `{"id":"msg_01","type":"message","role":"assistant","model":"claude-sonnet-4-5",` +
		`"content":[{"type":"text","text":"Hi"}],"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":2}}`
	shapeMessagesStream = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_02\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-5\"}}\n\n"
)

// Anthropic's "overloaded" status
const anthropicOverloadedStatus = 529

func shapeConfig() *config.GenAIConfig {
	return &config.GenAIConfig{
		OpenAI:         config.OpenAIConfig{Enabled: true},
		Anthropic:      config.AnthropicConfig{Enabled: true},
		ShapeDetection: config.GenAIShapeDetectionConfig{Enabled: true},
	}
}

func jsonHeaders() http.Header {
	h := http.Header{}
	h.Set("Content-Type", "application/json")
	return h
}

func shapeSpan(t *testing.T, cfg *config.GenAIConfig, method, url string, reqHeaders http.Header, reqBody string,
	status int, respHeaders http.Header, respBody string,
) (request.Span, bool) {
	t.Helper()
	req := makeRequest(t, method, url, reqBody)
	for k, v := range reqHeaders {
		req.Header[k] = v
	}
	base := request.Span{Type: request.EventTypeHTTPClient, Path: req.URL.Path}
	return GenAIShapeSpan(&base, req, makePlainResponse(status, respHeaders, respBody), cfg)
}

func TestGenAIShapeSpan_SelfHostedChatCompletions(t *testing.T) {
	span, ok := shapeSpan(t, shapeConfig(), http.MethodPost, "http://vllm.internal:8000/v1/chat/completions", nil,
		shapeChatRequest, http.StatusOK, jsonHeaders(), shapeChatResponse)
	require.True(t, ok)

	assert.Equal(t, request.HTTPSubtypeOpenAICompatible, span.SubType)
	ai := span.GenAI.OpenAICompatible
	require.NotNil(t, ai)
	assert.Equal(t, "custom", span.GenAIProviderName())
	assert.Equal(t, request.ChatOperationName, ai.OperationName)
	assert.Equal(t, "meta-llama/Llama-3.1-8B-Instruct", ai.Request.Model)
	require.Len(t, ai.ToolCalls, 1)
	assert.Equal(t, "chatcmpl-tool-9a1", ai.ToolCalls[0].ID)
	assert.Equal(t, "get_weather", ai.ToolCalls[0].Name)
}

func TestGenAIShapeSpan_ChatCompletionsStream(t *testing.T) {
	span, ok := shapeSpan(t, shapeConfig(), http.MethodPost, "http://127.0.0.1:11434/v1/chat/completions", nil,
		`{"model":"qwen2.5","stream":true,"messages":[{"role":"user","content":"hi"}]}`, http.StatusOK, http.Header{}, shapeChatStream)
	require.True(t, ok)
	require.NotNil(t, span.GenAI.OpenAICompatible)
	assert.Equal(t, "qwen2.5", span.GenAI.OpenAICompatible.ResponseModel)
	assert.True(t, span.GenAI.OpenAICompatible.Request.Stream)
}

func TestGenAIShapeSpan_LiteLLMProxyNamesItself(t *testing.T) {
	hdr := jsonHeaders()
	hdr.Set("X-Litellm-Call-Id", "6d9c4c1e")
	span, ok := shapeSpan(t, shapeConfig(), http.MethodPost, "http://litellm:4000/v1/chat/completions", nil,
		shapeChatRequest, http.StatusOK, hdr, shapeChatResponse)
	require.True(t, ok)
	assert.Equal(t, "litellm", span.GenAIProviderName())
}

func TestGenAIShapeSpan_GatewayPrefixedPath(t *testing.T) {
	_, ok := shapeSpan(t, shapeConfig(), http.MethodPost, "https://gw.corp.example/llm/openai/v1/chat/completions", nil,
		shapeChatRequest, http.StatusOK, jsonHeaders(), shapeChatResponse)
	assert.True(t, ok)
}

func TestGenAIShapeSpan_Responses(t *testing.T) {
	for name, body := range map[string]string{"json": shapeResponsesResponse, "stream": shapeResponsesStream} {
		t.Run(name, func(t *testing.T) {
			span, ok := shapeSpan(t, shapeConfig(), http.MethodPost, "http://vllm.internal:8000/v1/responses", nil,
				shapeResponsesRequest, http.StatusOK, http.Header{}, body)
			require.True(t, ok)
			require.NotNil(t, span.GenAI.OpenAICompatible)
			assert.Equal(t, openAIAPITypeResponses, span.GenAI.OpenAICompatible.APIType)
		})
	}
}

func TestGenAIShapeSpan_AnthropicMessages(t *testing.T) {
	version := http.Header{"Anthropic-Version": {"2023-06-01"}}
	for name, body := range map[string]string{"json": shapeMessagesResult, "stream": shapeMessagesStream} {
		t.Run(name, func(t *testing.T) {
			span, ok := shapeSpan(t, shapeConfig(), http.MethodPost, "http://gateway.internal/v1/messages", version,
				shapeMessagesRequest, http.StatusOK, http.Header{}, body)
			require.True(t, ok)
			assert.Equal(t, request.HTTPSubtypeAnthropic, span.SubType)
			require.NotNil(t, span.GenAI.Anthropic)
			assert.Equal(t, "claude-sonnet-4-5", span.GenAI.Anthropic.Input.Model)
		})
	}
}

func TestGenAIShapeSpan_ErrorResponses(t *testing.T) {
	_, ok := shapeSpan(t, shapeConfig(), http.MethodPost, "http://vllm.internal:8000/v1/chat/completions", nil,
		shapeChatRequest, http.StatusBadRequest, jsonHeaders(), `{"object":"error","message":"model not found","type":"NotFoundError","code":404}`)
	assert.True(t, ok, "vLLM's flat error")

	_, ok = shapeSpan(t, shapeConfig(), http.MethodPost, "http://gw/v1/chat/completions", nil,
		shapeChatRequest, http.StatusTooManyRequests, jsonHeaders(), `{"error":{"message":"rate limited","type":"rate_limit"}}`)
	assert.True(t, ok, "OpenAI error object")

	_, ok = shapeSpan(t, shapeConfig(), http.MethodPost, "http://gw/v1/messages", http.Header{"Anthropic-Version": {"2023-06-01"}},
		shapeMessagesRequest, anthropicOverloadedStatus, jsonHeaders(), `{"type":"error","error":{"type":"overloaded_error"}}`)
	assert.True(t, ok, "Anthropic error")

	_, ok = shapeSpan(t, shapeConfig(), http.MethodPost, "http://gw/v1/chat/completions", nil,
		shapeChatRequest, http.StatusOK, jsonHeaders(), `{"error":{"message":"not an LLM"}}`)
	assert.False(t, ok, "an error body on a successful status is not an LLM error")
}

func TestGenAIShapeSpan_TruncatedRequestKeepsConversation(t *testing.T) {
	// the model follows the messages and was cut off with the rest of the body
	truncated := shapeChatRequest[:len(shapeChatRequest)/2]
	_, ok := shapeSpan(t, shapeConfig(), http.MethodPost, "http://vllm.internal:8000/v1/chat/completions", nil,
		truncated, http.StatusOK, jsonHeaders(), shapeChatResponse)
	assert.True(t, ok)
}

func TestGenAIShapeSpan_UnrelatedAPIsAreNotLLMCalls(t *testing.T) {
	version := http.Header{"Anthropic-Version": {"2023-06-01"}}
	cases := []struct {
		name, method, url string
		reqHeaders        http.Header
		reqBody, respBody string
	}{
		{"chat app messages without anthropic-version", http.MethodPost, "http://chat.example/v1/messages", nil,
			`{"model":"x","messages":[{"role":"user","content":"hi"}]}`, shapeMessagesResult},
		{"survey responses", http.MethodPost, "http://forms.example/forms/7/responses", nil,
			`{"model":"survey-v2","input":"yes"}`, `{"id":"r1","status":"saved"}`},
		{"responses without input", http.MethodPost, "http://vllm.internal/v1/responses", nil,
			`{"model":"gpt-oss-20b","answers":[1,2]}`, shapeResponsesResponse},
		{"chat completions read back", http.MethodGet, "http://vllm.internal/v1/chat/completions", nil,
			"", shapeChatResponse},
		{"messages that are not an array", http.MethodPost, "http://vllm.internal/v1/chat/completions", nil,
			`{"model":"m","messages":"hello"}`, shapeChatResponse},
		{"complete request without a model", http.MethodPost, "http://vllm.internal/v1/chat/completions", nil,
			`{"messages":[{"role":"user","content":"hi"}]}`, shapeChatResponse},
		{"response of another shape", http.MethodPost, "http://vllm.internal/v1/chat/completions", nil,
			shapeChatRequest, `{"id":"1","result":"ok"}`},
		{"html response", http.MethodPost, "http://vllm.internal/v1/chat/completions", nil,
			shapeChatRequest, "<html><body>Bad gateway</body></html>"},
		{"non-JSON request", http.MethodPost, "http://vllm.internal/v1/chat/completions", nil,
			"model=m&messages=hi", shapeChatResponse},
		{"anthropic stream on the openai endpoint", http.MethodPost, "http://vllm.internal/v1/chat/completions", nil,
			shapeChatRequest, shapeMessagesStream},
		{"openai response on the anthropic endpoint", http.MethodPost, "http://gw/v1/messages", version,
			shapeMessagesRequest, shapeChatResponse},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, ok := shapeSpan(t, shapeConfig(), tc.method, tc.url, tc.reqHeaders, tc.reqBody, http.StatusOK, jsonHeaders(), tc.respBody)
			assert.False(t, ok)
		})
	}
}

func TestGenAIShapeSpan_FollowsExtractorEnablement(t *testing.T) {
	version := http.Header{"Anthropic-Version": {"2023-06-01"}}

	onlyAnthropic := shapeConfig()
	onlyAnthropic.OpenAI.Enabled = false
	_, ok := shapeSpan(t, onlyAnthropic, http.MethodPost, "http://vllm.internal/v1/chat/completions", nil,
		shapeChatRequest, http.StatusOK, jsonHeaders(), shapeChatResponse)
	assert.False(t, ok, "OpenAI format needs the openai or openai_compatible extractor")

	onlyAnthropic.OpenAICompatible.Enabled = true
	_, ok = shapeSpan(t, onlyAnthropic, http.MethodPost, "http://vllm.internal/v1/chat/completions", nil,
		shapeChatRequest, http.StatusOK, jsonHeaders(), shapeChatResponse)
	assert.True(t, ok)

	onlyOpenAI := shapeConfig()
	onlyOpenAI.Anthropic.Enabled = false
	_, ok = shapeSpan(t, onlyOpenAI, http.MethodPost, "http://gw/v1/messages", version,
		shapeMessagesRequest, http.StatusOK, jsonHeaders(), shapeMessagesResult)
	assert.False(t, ok, "Anthropic format needs the anthropic extractor")
}

func TestTopLevelJSONKinds(t *testing.T) {
	kinds, complete, ok := topLevelJSONKinds([]byte(` {"model" : "m", "messages":[{"a":1}], "n":2, "o":{}, "s":null}`))
	require.True(t, ok)
	assert.True(t, complete)
	assert.Equal(t, map[string]byte{"model": '"', "messages": '[', "n": '2', "o": '{', "s": 'n'}, kinds)

	kinds, complete, ok = topLevelJSONKinds([]byte(`{"messages":[{"role":"user","content":"long`))
	require.True(t, ok)
	assert.False(t, complete)
	assert.Equal(t, byte('['), kinds["messages"])

	_, _, ok = topLevelJSONKinds([]byte(`[{"model":"m"}]`))
	assert.False(t, ok)
}

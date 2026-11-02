// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package request

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestTrailingTurnKind(t *testing.T) {
	chat := func(messages string) *GenAI {
		return &GenAI{OpenAI: &VendorOpenAI{Request: OpenAIInput{Messages: json.RawMessage(messages)}}}
	}
	tests := []struct {
		name string
		ai   *GenAI
		want string
	}{
		{"first user turn", chat(`[{"role":"system","content":"s"},{"role":"user","content":"u"}]`), TrailingTurnUser},
		{"tool results", chat(`[{"role":"user","content":"u"},{"role":"assistant","tool_calls":[{"id":"c"}]},` +
			`{"role":"tool","tool_call_id":"c","content":"r"}]`), TrailingTurnToolResults},
		{"next user turn", chat(`[{"role":"user","content":"u"},{"role":"assistant","content":"a"},{"role":"user","content":"v"}]`),
			TrailingTurnUser},
		{"prefill", chat(`[{"role":"user","content":"u"},{"role":"assistant","content":"a"}]`), TrailingTurnOther},
		{"responses string input", &GenAI{OpenAI: &VendorOpenAI{Request: OpenAIInput{Input: "hi"}}}, TrailingTurnUser},
		{"responses items", &GenAI{OpenAI: &VendorOpenAI{Request: OpenAIInput{InputItems: json.RawMessage(
			`[{"type":"function_call","call_id":"c"},{"type":"function_call_output","call_id":"c","output":"r"}]`)}}},
			TrailingTurnToolResults},
		{"anthropic tool result", &GenAI{Anthropic: &VendorAnthropic{Input: AnthropicRequest{Messages: json.RawMessage(
			`[{"role":"user","content":"u"},{"role":"assistant","content":[{"type":"tool_use","id":"t"}]},` +
				`{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":"r"}]}]`)}}},
			TrailingTurnToolResults},
		{"gemini user", &GenAI{Gemini: &VendorGemini{Input: GeminiRequest{Contents: json.RawMessage(
			`[{"role":"user","parts":[{"text":"u"}]},{"role":"model","parts":[{"text":"a"}]},{"role":"user","parts":[{"text":"v"}]}]`)}}},
			TrailingTurnUser},
		{"unparsed", &GenAI{UnparsedRequest: []byte(`{"messages":[`)}, TrailingTurnOther},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, tt.ai.TrailingTurnKind(tt.ai.ToolResults()))
		})
	}
}

func TestContinuesConversation(t *testing.T) {
	chat := func(messages string) *GenAI {
		return &GenAI{OpenAI: &VendorOpenAI{Request: OpenAIInput{Messages: json.RawMessage(messages)}}}
	}
	assert.False(t, chat(`[{"role":"system","content":"s"},{"role":"user","content":"u"}]`).ContinuesConversation())
	assert.True(t, chat(`[{"role":"user","content":"u"},{"role":"assistant","content":"a"},{"role":"user","content":"v"}]`).
		ContinuesConversation())
	// a history the capture cut is looked at up to the cut
	assert.True(t, chat(`[{"role":"user","content":"u"},{"role":"assistant","content":"a"},{"role":"user","cont`).
		ContinuesConversation())
	assert.True(t, (&GenAI{OpenAI: &VendorOpenAI{Request: OpenAIInput{PreviousResponseID: "resp_1"}}}).ContinuesConversation())
}

func TestOffersTools(t *testing.T) {
	assert.True(t, (&GenAI{OpenAI: &VendorOpenAI{Request: OpenAIInput{Tools: json.RawMessage(`[{"type":"function"}]`)}}}).OffersTools())
	assert.False(t, (&GenAI{OpenAI: &VendorOpenAI{Request: OpenAIInput{Tools: json.RawMessage(` [ ] `)}}}).OffersTools())
	assert.False(t, (&GenAI{OpenAI: &VendorOpenAI{Request: OpenAIInput{Tools: json.RawMessage(`null`)}}}).OffersTools())
	assert.True(t, (&GenAI{Bedrock: &VendorBedrock{Input: BedrockRequest{ToolConfig: json.RawMessage(`{"tools":[{}]}`)}}}).OffersTools())
	assert.True(t, (&GenAI{Anthropic: &VendorAnthropic{Input: AnthropicRequest{Tools: json.RawMessage(`[{"name":"x"}]`)}}}).OffersTools())
	assert.False(t, (&GenAI{Gemini: &VendorGemini{}}).OffersTools())
}

func TestArrayElementEnds(t *testing.T) {
	raw := []byte(`[ {"a":"]"} , "x,y", 3 ,{"b":[1,2]}]`)
	ends := ArrayElementEnds(raw)
	var elems []string
	from := 1
	for _, end := range ends {
		elems = append(elems, string(raw[from:end]))
		from = end
	}
	assert.Equal(t, []string{` {"a":"]"}`, ` , "x,y"`, `, 3`, ` ,{"b":[1,2]}`}, elems)

	assert.Equal(t, []int{8}, ArrayElementEnds([]byte(`[{"a":1},{"b":`)))
	assert.Empty(t, ArrayElementEnds([]byte(`[]`)))
	assert.Empty(t, ArrayElementEnds([]byte(`{"a":1}`)))
}

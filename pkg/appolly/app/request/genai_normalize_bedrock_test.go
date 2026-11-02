// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package request

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNormalizeBedrockOutput_TextBlocks(t *testing.T) {
	resp := &BedrockResponse{
		Content:    json.RawMessage(`[{"type":"text","text":"eBPF is a kernel technology."}]`),
		StopReason: "end_turn",
	}
	result := NormalizeBedrockOutput(resp)

	var msgs []normalizedMessage
	require.NoError(t, json.Unmarshal([]byte(result), &msgs))
	require.Len(t, msgs, 1)
	assert.Equal(t, "assistant", msgs[0].Role)
	assert.Equal(t, "end_turn", msgs[0].FinishReason)
	require.Len(t, msgs[0].Parts, 1)
	assert.Equal(t, "text", msgs[0].Parts[0].Type)
	assert.Equal(t, "eBPF is a kernel technology.", msgs[0].Parts[0].Content)
}

func TestNormalizeBedrockOutput_ToolUseBlocks(t *testing.T) {
	resp := &BedrockResponse{
		Content:    json.RawMessage(`[{"type":"tool_use","id":"tu_1","name":"lookup","input":{"key":"val"}}]`),
		StopReason: "tool_use",
	}
	result := NormalizeBedrockOutput(resp)

	var msgs []normalizedMessage
	require.NoError(t, json.Unmarshal([]byte(result), &msgs))
	require.Len(t, msgs, 1)
	require.Len(t, msgs[0].Parts, 1)
	assert.Equal(t, "tool_call", msgs[0].Parts[0].Type)
	assert.Equal(t, "tu_1", msgs[0].Parts[0].ID)
	assert.Equal(t, "lookup", msgs[0].Parts[0].Name)
}

func TestNormalizeBedrockOutput_MixedBlocks(t *testing.T) {
	resp := &BedrockResponse{
		Content:    json.RawMessage(`[{"type":"text","text":"thinking..."},{"type":"tool_use","id":"tu_2","name":"calc","input":{"x":1}}]`),
		StopReason: "tool_use",
	}
	result := NormalizeBedrockOutput(resp)

	var msgs []normalizedMessage
	require.NoError(t, json.Unmarshal([]byte(result), &msgs))
	require.Len(t, msgs, 1)
	require.Len(t, msgs[0].Parts, 2)
	assert.Equal(t, "text", msgs[0].Parts[0].Type)
	assert.Equal(t, "tool_call", msgs[0].Parts[1].Type)
}

func TestNormalizeBedrockOutput_Empty(t *testing.T) {
	resp := &BedrockResponse{}
	assert.Empty(t, NormalizeBedrockOutput(resp))
}

func TestNormalizeBedrockOutput_UnmarshalFailure(t *testing.T) {
	resp := &BedrockResponse{
		Content:    json.RawMessage(`not valid json`),
		StopReason: "end_turn",
	}
	result := NormalizeBedrockOutput(resp)

	var msgs []normalizedMessage
	require.NoError(t, json.Unmarshal([]byte(result), &msgs))
	require.Len(t, msgs, 1)
	assert.Equal(t, "assistant", msgs[0].Role)
	assert.Equal(t, "end_turn", msgs[0].FinishReason)
	assert.Equal(t, "not valid json", msgs[0].Parts[0].Content)
}

func TestNormalizeBedrockOutput_UnknownBlockType(t *testing.T) {
	resp := &BedrockResponse{
		Content:    json.RawMessage(`[{"type":"custom","text":"custom content"}]`),
		StopReason: "end_turn",
	}
	result := NormalizeBedrockOutput(resp)

	var msgs []normalizedMessage
	require.NoError(t, json.Unmarshal([]byte(result), &msgs))
	require.Len(t, msgs, 1)
	require.Len(t, msgs[0].Parts, 1)
	assert.Equal(t, "custom", msgs[0].Parts[0].Type)
	assert.Equal(t, "custom content", msgs[0].Parts[0].Content)
}

func TestNormalizeConverseContent(t *testing.T) {
	b := VendorBedrock{
		Input: BedrockRequest{Messages: json.RawMessage(`[
			{"role":"user","content":[{"text":"weather?"}]},
			{"role":"assistant","content":[{"toolUse":{"toolUseId":"tooluse_1","name":"get_weather","input":{"city":"Paris"}}}]},
			{"role":"user","content":[{"toolResult":{"toolUseId":"tooluse_1","content":[{"text":"rainy"}],"status":"success"}}]}]`)},
		Output: BedrockResponse{
			StopReasonNova: "tool_use",
			Output: &NovaOutput{Message: &NovaMessage{Role: "assistant", Content: json.RawMessage(
				`[{"text":"checking"},{"toolUse":{"toolUseId":"tooluse_2","name":"get_time","input":{}}}]`)}},
		},
	}

	assert.JSONEq(t, `[
		{"role":"user","parts":[{"type":"text","content":"weather?"}]},
		{"role":"assistant","parts":[{"type":"tool_call","id":"tooluse_1","name":"get_weather","arguments":{"city":"Paris"}}]},
		{"role":"user","parts":[{"type":"tool_call_response","id":"tooluse_1","result":[{"text":"rainy"}]}]}]`, b.GetInput())
	assert.JSONEq(t, `[{"role":"assistant","finish_reason":"tool_use","parts":[
		{"type":"text","content":"checking"},
		{"type":"tool_call","id":"tooluse_2","name":"get_time","arguments":{}}]}]`, b.GetOutput())

	// Anthropic-shaped content keeps its own normalization
	b.Input.Messages = json.RawMessage(`[{"role":"user","content":[{"type":"text","text":"hi"}]}]`)
	assert.JSONEq(t, `[{"role":"user","parts":[{"type":"text","content":"hi"}]}]`, b.GetInput())
}

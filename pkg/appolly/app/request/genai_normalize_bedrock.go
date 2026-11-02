// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package request // import "go.opentelemetry.io/obi/pkg/appolly/app/request"

import (
	"encoding/json"
)

// NormalizeBedrockOutput converts Bedrock Claude-style content blocks
// to the semconv output messages schema. For Bedrock responses that use
// Anthropic Claude format, the content blocks are identical to Anthropic.
func NormalizeBedrockOutput(resp *BedrockResponse) string {
	if len(resp.Content) == 0 {
		return ""
	}

	var blocks []struct {
		Type  string          `json:"type"`
		Text  string          `json:"text,omitempty"`
		ID    string          `json:"id,omitempty"`
		Name  string          `json:"name,omitempty"`
		Input json.RawMessage `json:"input,omitempty"`
	}
	if err := json.Unmarshal(resp.Content, &blocks); err != nil {
		return wrapTextAsOutputMessage("assistant", string(resp.Content), resp.StopReason)
	}

	var parts []normalizedPart
	for _, b := range blocks {
		switch b.Type {
		case "text":
			parts = append(parts, normalizedPart{Type: "text", Content: b.Text})
		case "tool_use":
			parts = append(parts, normalizedPart{
				Type:      "tool_call",
				ID:        b.ID,
				Name:      b.Name,
				Arguments: b.Input,
			})
		default:
			parts = append(parts, normalizedPart{Type: b.Type, Content: b.Text})
		}
	}

	msg := normalizedMessage{
		Role:         "assistant",
		Parts:        parts,
		FinishReason: resp.StopReason,
	}

	out, err := json.Marshal([]normalizedMessage{msg})
	if err != nil {
		return string(resp.Content)
	}
	return string(out)
}

// converseBlock is a Bedrock Converse (and Amazon Nova) content block. It
// names its kind by key rather than by a "type" field.
type converseBlock struct {
	Type    string `json:"type"`
	Text    string `json:"text"`
	ToolUse *struct {
		ToolUseID string          `json:"toolUseId"`
		Name      string          `json:"name"`
		Input     json.RawMessage `json:"input"`
	} `json:"toolUse"`
	ToolResult *struct {
		ToolUseID string          `json:"toolUseId"`
		Content   json.RawMessage `json:"content"`
	} `json:"toolResult"`
}

// converseContentToParts converts Converse content blocks to semconv parts.
// It reports false for content in another shape, such as Anthropic blocks.
func converseContentToParts(raw json.RawMessage) ([]normalizedPart, bool) {
	var blocks []converseBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return nil, false
	}

	parts := make([]normalizedPart, 0, len(blocks))
	for _, b := range blocks {
		switch {
		case b.Type != "":
			return nil, false
		case b.ToolUse != nil:
			parts = append(parts, normalizedPart{Type: "tool_call", ID: b.ToolUse.ToolUseID, Name: b.ToolUse.Name, Arguments: b.ToolUse.Input})
		case b.ToolResult != nil:
			parts = append(parts, normalizedPart{Type: "tool_call_response", ID: b.ToolResult.ToolUseID, Response: extractToolResultContent(b.ToolResult.Content)})
		default:
			parts = append(parts, normalizedPart{Type: "text", Content: b.Text})
		}
	}
	return parts, true
}

// normalizeConverseMessages converts Converse request messages, or reports
// false when they are Anthropic-shaped.
func normalizeConverseMessages(raw json.RawMessage) (string, bool) {
	var msgs []struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(raw, &msgs); err != nil {
		return "", false
	}

	out := make([]normalizedMessage, 0, len(msgs))
	for _, m := range msgs {
		parts, ok := converseContentToParts(m.Content)
		if !ok {
			return "", false
		}
		out = append(out, normalizedMessage{Role: m.Role, Parts: parts})
	}

	b, err := json.Marshal(out)
	if err != nil {
		return "", false
	}
	return string(b), true
}

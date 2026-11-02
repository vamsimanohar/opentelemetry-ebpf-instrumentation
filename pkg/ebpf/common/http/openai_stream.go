// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package ebpfcommon // import "go.opentelemetry.io/obi/pkg/ebpf/common/http"

import (
	"bufio"
	"encoding/json"
	"io"
	"log/slog"
	"maps"
	"slices"
	"strings"

	jsoniter "github.com/json-iterator/go"

	"go.opentelemetry.io/obi/pkg/appolly/app/request"
)

// maxStreamToolCalls caps the tool-call accumulator to prevent unbounded
// growth from untrusted tool_calls[].index values.
const maxStreamToolCalls = 256

type openAIStreamChunk struct {
	ID     string `json:"id"`
	Object string `json:"object"`
	Model  string `json:"model"`
	// Type, Item and OutputIndex are set on Responses API stream events.
	Type        string          `json:"type"`
	Item        json.RawMessage `json:"item"`
	OutputIndex int             `json:"output_index"`
	Choices     []struct {
		Index int `json:"index"`
		Delta struct {
			Role      string                 `json:"role"`
			Content   string                 `json:"content"`
			ToolCalls []openAIStreamToolCall `json:"tool_calls"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
}

type openAIStreamToolCall struct {
	Index    int    `json:"index"`
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// parseOpenAIStream parses the SSE stream from OpenAI-compatible APIs (including Qwen/DashScope)
// and returns the aggregated response with usage statistics and tool calls.
func parseOpenAIStream(reader io.Reader) (*request.VendorOpenAI, []request.ToolCall) {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 0, 256*1024), 256*1024)
	response := &request.VendorOpenAI{}

	var finishReason string
	var role string
	var contentBuilder strings.Builder
	var accumulators []openAIToolCallAccum
	var responses responsesStreamOutput

	for scanner.Scan() {
		line := scanner.Text()

		data, ok := extractSSEData(line)
		if !ok {
			continue
		}

		if data == "[DONE]" {
			break
		}

		var chunk openAIStreamChunk
		unmarshalJSONBestEffort([]byte(data), &chunk)
		responses.add(&chunk, []byte(data))
		var responseFields struct {
			ID    string `json:"id"`
			Model string `json:"model"`
		}
		if unmarshalJSONContainerBestEffort([]byte(data), &responseFields, "response") {
			if chunk.ID == "" {
				chunk.ID = responseFields.ID
			}
			if chunk.Model == "" {
				chunk.Model = responseFields.Model
			}
		}
		var usage request.OpenAIUsage
		if unmarshalJSONContainerBestEffort([]byte(data), &usage, "response", "usage") {
			response.Usage.Merge(usage)
		}
		if unmarshalJSONContainerBestEffort([]byte(data), &usage, "usage") {
			response.Usage.Merge(usage)
		}

		// Extract model and id from the first chunk that has them.
		if response.ID == "" && chunk.ID != "" {
			response.ID = chunk.ID
		}
		if response.ResponseModel == "" && chunk.Model != "" {
			response.ResponseModel = chunk.Model
		}

		// Process choices.
		for i := range chunk.Choices {
			choice := &chunk.Choices[i]

			// Track finish reason from the last choice that reports one.
			if choice.FinishReason != nil && *choice.FinishReason != "" {
				finishReason = *choice.FinishReason
			}

			// Capture assistant role (typically in the first delta) and
			// accumulate content fragments to reconstruct the full message.
			if choice.Delta.Role != "" {
				role = choice.Delta.Role
			}
			if choice.Delta.Content != "" {
				contentBuilder.WriteString(choice.Delta.Content)
			}

			// Accumulate tool calls by index.
			for j := range choice.Delta.ToolCalls {
				tc := &choice.Delta.ToolCalls[j]
				idx := tc.Index
				if idx < 0 || idx >= maxStreamToolCalls {
					continue
				}

				// Grow the accumulator slice as needed.
				for len(accumulators) <= idx {
					accumulators = append(accumulators, openAIToolCallAccum{})
				}

				accumulators[idx].add(tc)
			}
		}
	}

	if err := scanner.Err(); err != nil {
		slog.Debug("parseOpenAIStream: scanner error", "error", err)
	}

	// Build the final tool calls list.
	var toolCalls []request.ToolCall
	for i := range accumulators {
		if accumulators[i].name == "" {
			continue
		}
		toolCalls = append(toolCalls, request.ToolCall{
			ID:        accumulators[i].id,
			Name:      accumulators[i].name,
			Arguments: string(accumulators[i].arguments),
			Type:      accumulators[i].toolType,
			Index:     len(toolCalls),
		})
	}

	// Build the Choices JSON with the aggregated message content, tool calls
	// and finish_reason so that VendorOpenAI.GetFinishReasons() and the GenAI
	// output normalization (normalizeOpenAIChoices) work correctly.
	if finishReason != "" || contentBuilder.Len() > 0 || len(toolCalls) > 0 {
		if choicesJSON, err := json.Marshal([]openAIStreamChoice{newOpenAIStreamChoice(role, contentBuilder.String(), finishReason, toolCalls)}); err == nil {
			response.Choices = choicesJSON
		}
	}

	// A Responses API stream carries its output as items, not choices.
	if output := responses.output(); len(output) > 0 {
		response.Output = output
		if len(toolCalls) == 0 {
			toolCalls = extractResponsesToolCalls(output)
		}
	}

	_, inputReported := response.Usage.InputTokenCount()
	_, outputReported := response.Usage.OutputTokenCount()
	if !inputReported && !outputReported && response.ID != "" {
		slog.Debug("parseOpenAIStream: no usage data found in SSE stream, token counts will be 0",
			"id", response.ID, "model", response.ResponseModel, "finishReason", finishReason)
	}

	return response, toolCalls
}

// openAIToolCallAccum accumulates the fragments of one streamed tool call. Only
// the first delta of a call carries its id and name; the arguments arrive as
// string fragments in order.
type openAIToolCallAccum struct {
	id        string
	name      string
	toolType  string
	arguments []byte
}

func (a *openAIToolCallAccum) add(tc *openAIStreamToolCall) {
	if tc.ID != "" {
		a.id = tc.ID
	}
	if tc.Function.Name != "" {
		a.name = tc.Function.Name
	}
	if tc.Type != "" {
		a.toolType = tc.Type
	}
	a.arguments = append(a.arguments, tc.Function.Arguments...)
}

type openAIStreamChoice struct {
	Message struct {
		Role      string                 `json:"role"`
		Content   string                 `json:"content"`
		ToolCalls []openAIStreamToolCall `json:"tool_calls,omitempty"`
	} `json:"message"`
	FinishReason string `json:"finish_reason"`
}

func newOpenAIStreamChoice(role, content, finishReason string, toolCalls []request.ToolCall) openAIStreamChoice {
	sc := openAIStreamChoice{FinishReason: finishReason}
	sc.Message.Role = role
	if sc.Message.Role == "" {
		sc.Message.Role = "assistant"
	}
	sc.Message.Content = content

	for i := range toolCalls {
		tc := openAIStreamToolCall{Index: toolCalls[i].Index, ID: toolCalls[i].ID, Type: toolCalls[i].Type}
		tc.Function.Name = toolCalls[i].Name
		tc.Function.Arguments = toolCalls[i].Arguments
		sc.Message.ToolCalls = append(sc.Message.ToolCalls, tc)
	}

	return sc
}

// responsesStreamOutput rebuilds the `output` array of a Responses API stream:
// from the terminal event's full response when it was captured, otherwise from
// the completed items in output order.
type responsesStreamOutput struct {
	final json.RawMessage
	items map[int]json.RawMessage
}

func (r *responsesStreamOutput) add(chunk *openAIStreamChunk, data []byte) {
	switch chunk.Type {
	case "response.output_item.done":
		if len(chunk.Item) == 0 || chunk.OutputIndex < 0 || chunk.OutputIndex >= maxStreamToolCalls {
			return
		}
		if r.items == nil {
			r.items = map[int]json.RawMessage{}
		}
		r.items[chunk.OutputIndex] = chunk.Item
	case "response.completed", "response.incomplete", "response.failed":
		if output := jsonBestEffort.Get(data, "response", "output"); output.ValueType() == jsoniter.ArrayValue {
			r.final = json.RawMessage(output.ToString())
		}
	}
}

func (r *responsesStreamOutput) output() json.RawMessage {
	if len(r.final) > 0 {
		return r.final
	}
	if len(r.items) == 0 {
		return nil
	}

	indexes := slices.Sorted(maps.Keys(r.items))
	items := make([]json.RawMessage, 0, len(indexes))
	for _, i := range indexes {
		items = append(items, r.items[i])
	}

	output, err := json.Marshal(items)
	if err != nil {
		return nil
	}

	return output
}

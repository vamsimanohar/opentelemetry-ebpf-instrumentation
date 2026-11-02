// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package ebpfcommon // import "go.opentelemetry.io/obi/pkg/ebpf/common/http"

import (
	"cmp"
	"encoding/json"
	"strings"

	"go.opentelemetry.io/obi/pkg/appolly/app/request"
)

// responsesClientTools maps the Responses API output items that the client
// executes to the tool type they report and the name used when the item has
// none. Their results come back keyed by call_id, never by the `fc_` item id.
var responsesClientTools = map[string]struct {
	toolType, defaultName string
}{
	"function_call":    {toolType: "function"},
	"custom_tool_call": {toolType: "custom"},
	"local_shell_call": {toolType: "local_shell", defaultName: "local_shell"},
	"shell_call":       {toolType: "shell", defaultName: "shell"},
	"apply_patch_call": {toolType: "apply_patch", defaultName: "apply_patch"},
	"computer_call":    {toolType: "computer", defaultName: "computer"},
}

// responsesServerTools are output items the provider executes itself.
var responsesServerTools = map[string]struct{}{
	"web_search_call":       {},
	"file_search_call":      {},
	"code_interpreter_call": {},
	"image_generation_call": {},
	"mcp_call":              {},
}

type responsesOutputItem struct {
	Type      string          `json:"type"`
	ID        string          `json:"id"`
	CallID    string          `json:"call_id"`
	Name      string          `json:"name"`
	Arguments string          `json:"arguments"`
	Input     string          `json:"input"`
	Action    json.RawMessage `json:"action"`
	Operation json.RawMessage `json:"operation"`
}

// extractResponsesToolCalls returns the tool calls of a Responses API `output`
// array. Server-side items are returned flagged so callers can tell them apart.
func extractResponsesToolCalls(output json.RawMessage) []request.ToolCall {
	if len(output) == 0 {
		return nil
	}

	var items []responsesOutputItem
	if err := json.Unmarshal(output, &items); err != nil {
		return nil
	}

	var result []request.ToolCall
	for i := range items {
		if tc, ok := responsesToolCall(&items[i]); ok {
			tc.Index = len(result)
			result = append(result, tc)
		}
	}

	return result
}

func responsesToolCall(item *responsesOutputItem) (request.ToolCall, bool) {
	if _, ok := responsesServerTools[item.Type]; ok {
		return request.ToolCall{
			ID:         item.ID,
			Name:       cmp.Or(item.Name, strings.TrimSuffix(item.Type, "_call")),
			Type:       item.Type,
			ServerSide: true,
		}, true
	}

	tool, ok := responsesClientTools[item.Type]
	if !ok {
		return request.ToolCall{}, false
	}

	name := cmp.Or(item.Name, tool.defaultName)
	if name == "" {
		return request.ToolCall{}, false
	}

	return request.ToolCall{
		ID:        item.CallID,
		Name:      name,
		Type:      tool.toolType,
		Arguments: responsesToolArguments(item),
	}, true
}

// responsesToolArguments returns the input of the call: a JSON-encoded string
// for function calls, free text for custom tools, and the action or operation
// object for the built-in client tools.
func responsesToolArguments(item *responsesOutputItem) string {
	switch {
	case item.Arguments != "":
		return item.Arguments
	case item.Input != "":
		return item.Input
	case len(item.Action) > 0:
		return string(item.Action)
	case len(item.Operation) > 0:
		return string(item.Operation)
	}

	return ""
}

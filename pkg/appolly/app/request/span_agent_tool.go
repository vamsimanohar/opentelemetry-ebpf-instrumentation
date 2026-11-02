// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package request // import "go.opentelemetry.io/obi/pkg/appolly/app/request"

import (
	"bytes"
	"cmp"
	"encoding/json"
	"strings"
)

// How the bounds of an inferred tool window were established. OBI never sees a
// tool run: the window opens when the call was available to the agent and
// closes when the agent sent its result, so its duration is an upper bound.
const (
	ToolWindowStartResponseEnd = "response_end"

	ToolWindowEndToolResult = "tool_result"
	// the next request of the same conversation, which carried no parsable
	// result: it was cut by the capture or encoded as text
	ToolWindowEndNextRequest = "next_request"
	// the next model call of the process, for a call that never got a result:
	// nothing shows that call belongs to the same conversation, so the end is
	// not proven to bound the tool
	ToolWindowEndNextInference = "next_inference"
	ToolWindowEndTimeout       = "timeout"
	ToolWindowEndProcessExit   = "process_exit"
)

// How an inferred tool call was paired with its result.
const (
	ToolOutcomeResult           = "result"
	ToolOutcomeResultByName     = "result_by_name"
	ToolOutcomeNextRequest      = "next_request"
	ToolOutcomeUnmatchedExit    = "unmatched_exit"
	ToolOutcomeUnmatchedTimeout = "unmatched_timeout"
)

// How kernel activity was assigned to an inferred tool call, from the
// strongest evidence to the weakest. A span reports the weakest rule that
// gave it activity.
const (
	ToolAttributionArgumentMatch = "argument_match"
	ToolAttributionLineage       = "lineage"
	ToolAttributionThread        = "thread"
	// the only window open, from a thread or a pool worker that does nothing
	// outside tool windows
	ToolAttributionExclusive = "exclusive"
	ToolAttributionAmbiguous = "ambiguous"
)

// InferredTool describes a tool execution OBI inferred from an LLM response
// that requested it and the later request that carried its result. It is also
// used for the window span that groups parallel calls, which has no Name.
type InferredTool struct {
	Name     string
	CallID   string // only when the provider supplied one
	Type     string
	Provider string
	// Arguments and Result are the raw provider payloads; both are exported
	// only when selected.
	Arguments string
	Result    string
	// PayloadTruncated tells that Arguments or Result were cut to fit
	PayloadTruncated bool
	IsError          bool

	WindowStart   string
	WindowEnd     string
	Outcome       string
	Attribution   string
	ParallelCount int
	// ActivityObserved tells that kernel activity was collected for the
	// window. ActivityComplete tells that none of it was lost: dropped by a
	// limit, expired before the window was known, or not collected because
	// the process was not yet known to run tool calls. Zero counts mean
	// nothing happened only when both hold.
	ActivityObserved bool
	ActivityComplete bool
	Activity         ToolActivityCounts
	// ActiveNs spans the first to the last attributed kernel event: the busy
	// part of the window.
	ActiveNs int64

	// Command is the single process the tool ran, when exactly one ran.
	Command *ObservedProcess
}

// ToolActivityCounts counts the kernel activity attributed to a span.
type ToolActivityCounts struct {
	Execs     int
	FileOpens int
	Connects  int
	Dropped   int
	// Ambiguous counts the activity of a single call's window that no rule
	// tied to the call; the window span of parallel calls carries it instead.
	Ambiguous int
	// AmbiguousOverlap counts the activity inside the window that is left
	// unattributed because another tool window of the process was open at
	// the same time and nothing tells which one it belongs to.
	AmbiguousOverlap int
}

// ObservedProcess is a process the kernel reported while a tool window was open.
type ObservedProcess struct {
	PID            uint32
	ExecutableName string
	ExecutablePath string
	CommandArgs    []string
	ExitCode       int
	Exited         bool
	// OutlivedWindow marks a process still running when its window closed: its
	// span end is the window end.
	OutlivedWindow bool
	// Attribution is the rule that tied the process to its tool call.
	Attribution string
}

// ToolResult is a tool result the agent sent back to the model.
type ToolResult struct {
	// ID is the tool call the result answers, when the request names one.
	ID      string
	Name    string
	Index   int
	IsError bool
	Raw     json.RawMessage
}

// ToolWindowSpanName names the span that groups parallel inferred tool calls.
const ToolWindowSpanName = "obi.gen_ai.tool_window"

func (s *Span) agentToolSpanName() string {
	switch {
	case s.Type == EventTypeProcessExec && s.Process != nil:
		return s.Process.ExecutableName
	case s.Type == EventTypeGenAIToolWindow:
		return ToolWindowSpanName
	case s.InferredTool != nil:
		return ExecuteToolOperationName + " " + s.InferredTool.Name
	}

	return ExecuteToolOperationName
}

// agentToolFailed reports a tool result flagged as an error, or a process that
// exited non-zero.
func (s *Span) agentToolFailed() bool {
	switch {
	case s.InferredTool != nil:
		return s.InferredTool.IsError
	case s.Process != nil:
		return s.Process.Exited && s.Process.ExitCode != 0
	}

	return false
}

// ToolCalls returns the tool calls of the response, whatever the provider.
func (g *GenAI) ToolCalls() []ToolCall {
	if ai := g.openAIFamily(); ai != nil {
		return ai.ToolCalls
	}

	switch {
	case g.Anthropic != nil:
		return g.Anthropic.ToolCalls
	case g.Gemini != nil:
		return g.Gemini.ToolCalls
	case g.Bedrock != nil:
		return g.Bedrock.ToolCalls
	}

	return nil
}

// ToolResults returns the tool results of the request's trailing turn: the
// messages after the model's last turn. Results replayed in the history of a
// stateless conversation answer calls that closed long ago, so they are left
// out. Only the trailing turn is decoded: histories grow with every turn and
// this runs for every GenAI span.
func (g *GenAI) ToolResults() []ToolResult {
	if ai := g.openAIFamily(); ai != nil {
		if len(ai.Request.InputItems) > 0 {
			return responsesToolResults(ai.Request.InputItems)
		}
		return chatToolResults(ai.Request.Messages)
	}

	switch {
	case g.Anthropic != nil:
		return messagesToolResults(g.Anthropic.Input.Messages)
	case g.Gemini != nil:
		return geminiToolResults(g.Gemini.Input.Contents)
	case g.Bedrock != nil:
		return messagesToolResults(g.Bedrock.Input.Messages)
	}

	return nil
}

// RequestHistory returns the conversation the request sent, as captured: the
// chat messages, the Responses input items or the Gemini contents. A
// stateless conversation resends its whole history, so a later request of the
// same conversation starts with the same bytes. For a body that did not
// parse, it is the part of the body from the conversation array on.
func (g *GenAI) RequestHistory() []byte {
	var history json.RawMessage
	if ai := g.openAIFamily(); ai != nil {
		history = ai.Request.InputItems
		if len(history) == 0 {
			history = ai.Request.Messages
		}
	} else {
		switch {
		case g.Anthropic != nil:
			history = g.Anthropic.Input.Messages
		case g.Gemini != nil:
			history = g.Gemini.Input.Contents
		case g.Bedrock != nil:
			history = g.Bedrock.Input.Messages
		}
	}
	if len(history) > 0 {
		return history
	}

	return unparsedHistory(g.UnparsedRequest)
}

// historyKeys name the conversation array in the request bodies of the
// supported providers.
var historyKeys = [][]byte{[]byte(`"messages"`), []byte(`"input"`), []byte(`"contents"`)}

func unparsedHistory(body []byte) []byte {
	for _, key := range historyKeys {
		i := bytes.Index(body, key)
		if i < 0 {
			continue
		}
		i = skipJSONSpace(body, i+len(key))
		if i >= len(body) || body[i] != ':' {
			continue
		}
		i = skipJSONSpace(body, i+1)
		if i < len(body) && body[i] == '[' {
			return body[i:]
		}
	}

	return nil
}

// ResponseID returns the provider's identifier of the response.
func (g *GenAI) ResponseID() string {
	if ai := g.openAIFamily(); ai != nil {
		return ai.ID
	}

	switch {
	case g.Anthropic != nil:
		return g.Anthropic.Output.ID
	case g.Gemini != nil:
		return g.Gemini.Output.ResponseID
	}

	return ""
}

// PreviousResponseID returns the response a stateful request continues.
func (g *GenAI) PreviousResponseID() string {
	if ai := g.openAIFamily(); ai != nil {
		return ai.Request.PreviousResponseID
	}

	return ""
}

func (g *GenAI) openAIFamily() *VendorOpenAI {
	switch {
	case g.OpenAI != nil:
		return g.OpenAI
	case g.Qwen != nil:
		return g.Qwen
	case g.Ollama != nil:
		return g.Ollama
	case g.Cohere != nil:
		return g.Cohere
	case g.OpenAICompatible != nil:
		return g.OpenAICompatible
	}

	return nil
}

// chatToolResults reads Chat Completions `tool` messages, the legacy `function`
// role, and Ollama's id-less `tool_name`.
func chatToolResults(raw json.RawMessage) []ToolResult {
	trailing := trailingTurn(raw, func(elem []byte) bool {
		var m struct {
			Role string `json:"role"`
		}
		return json.Unmarshal(elem, &m) == nil && m.Role == "assistant"
	})

	var results []ToolResult
	for _, elem := range trailing {
		var m struct {
			Role       string          `json:"role"`
			ToolCallID string          `json:"tool_call_id"`
			Name       string          `json:"name"`
			ToolName   string          `json:"tool_name"`
			Content    json.RawMessage `json:"content"`
		}
		if json.Unmarshal(elem, &m) != nil || (m.Role != "tool" && m.Role != "function") {
			continue
		}
		results = append(results, ToolResult{
			ID:    m.ToolCallID,
			Name:  cmp.Or(m.Name, m.ToolName),
			Index: len(results),
			Raw:   m.Content,
		})
	}

	return results
}

// responsesItemCallTypes are the Responses API items a model emits for a tool
// the client runs. Their results come back as `<type>_output` items.
var responsesItemCallTypes = map[string]struct{}{
	"function_call":    {},
	"custom_tool_call": {},
	"local_shell_call": {},
	"shell_call":       {},
	"apply_patch_call": {},
	"computer_call":    {},
}

type responsesItem struct {
	Type   string          `json:"type"`
	Role   string          `json:"role"`
	CallID string          `json:"call_id"`
	Status string          `json:"status"`
	Output json.RawMessage `json:"output"`
}

// responsesToolResults reads the `*_call_output` items of the Responses API,
// keyed by call_id. A stateful continuation (previous_response_id) carries only
// those items, so all of them are trailing.
func responsesToolResults(raw json.RawMessage) []ToolResult {
	trailing := trailingTurn(raw, isResponsesModelItem)

	var results []ToolResult
	for _, elem := range trailing {
		var it responsesItem
		if json.Unmarshal(elem, &it) != nil {
			continue
		}
		callType, isOutput := strings.CutSuffix(it.Type, "_output")
		if !isOutput {
			continue
		}
		if _, ok := responsesItemCallTypes[callType]; !ok {
			continue
		}
		results = append(results, ToolResult{
			ID:      it.CallID,
			Index:   len(results),
			IsError: it.Status == "failed",
			Raw:     it.Output,
		})
	}

	return results
}

// isResponsesModelItem tells whether a Responses API input item is part of a
// model turn: a call, a reasoning item or an assistant message.
func isResponsesModelItem(elem []byte) bool {
	var it struct {
		Type string `json:"type"`
		Role string `json:"role"`
	}
	if json.Unmarshal(elem, &it) != nil {
		return false
	}
	_, isCall := responsesItemCallTypes[it.Type]
	return isCall || it.Type == "reasoning" || (it.Type == "message" && it.Role == "assistant")
}

// messagesToolResults reads Anthropic Messages `tool_result` blocks, which
// Bedrock InvokeModel carries too, and Bedrock Converse `toolResult` blocks.
func messagesToolResults(raw json.RawMessage) []ToolResult {
	trailing := trailingTurn(raw, func(elem []byte) bool {
		var m struct {
			Role string `json:"role"`
		}
		return json.Unmarshal(elem, &m) == nil && m.Role == "assistant"
	})

	var results []ToolResult
	for _, elem := range trailing {
		var m struct {
			Content json.RawMessage `json:"content"`
		}
		if json.Unmarshal(elem, &m) == nil {
			results = appendContentToolResults(results, m.Content)
		}
	}

	return results
}

func appendContentToolResults(results []ToolResult, content json.RawMessage) []ToolResult {
	var blocks []struct {
		Type       string          `json:"type"`
		ToolUseID  string          `json:"tool_use_id"`
		IsError    bool            `json:"is_error"`
		Content    json.RawMessage `json:"content"`
		ToolResult *struct {
			ToolUseID string          `json:"toolUseId"`
			Status    string          `json:"status"`
			Content   json.RawMessage `json:"content"`
		} `json:"toolResult"`
	}
	// a plain string content holds no results
	if json.Unmarshal(content, &blocks) != nil {
		return results
	}

	for i := range blocks {
		b := &blocks[i]
		switch {
		case b.Type == "tool_result":
			results = append(results, ToolResult{ID: b.ToolUseID, Index: len(results), IsError: b.IsError, Raw: b.Content})
		case b.ToolResult != nil:
			tr := b.ToolResult
			results = append(results, ToolResult{ID: tr.ToolUseID, Index: len(results), IsError: tr.Status == "error", Raw: tr.Content})
		}
	}

	return results
}

// geminiToolResults reads `functionResponse` parts after the model's last turn.
// Their id is optional; the only error signal is an `error` key in the response.
func geminiToolResults(raw json.RawMessage) []ToolResult {
	trailing := trailingTurn(raw, func(elem []byte) bool {
		var c struct {
			Role string `json:"role"`
		}
		return json.Unmarshal(elem, &c) == nil && c.Role == "model"
	})

	var results []ToolResult
	for _, elem := range trailing {
		var content struct {
			Parts []geminiPart `json:"parts"`
		}
		if json.Unmarshal(elem, &content) != nil {
			continue
		}
		for _, p := range content.Parts {
			fr := p.FunctionResponse
			if fr == nil {
				continue
			}
			var resp struct {
				Error json.RawMessage `json:"error"`
			}
			_ = json.Unmarshal(fr.Response, &resp)
			results = append(results, ToolResult{
				ID:      fr.ID,
				Name:    fr.Name,
				Index:   len(results),
				IsError: isErrorValue(resp.Error),
				Raw:     fr.Response,
			})
		}
	}

	return results
}

// isErrorValue tells whether an `error` member reports an error: SDKs fill
// it with null, false, an empty string or an empty object on success.
func isErrorValue(raw json.RawMessage) bool {
	switch v := strings.TrimSpace(string(raw)); v {
	case "", "null", "false", `""`, "{}", "[]":
		return false
	default:
		return !emptyJSONContainer(v)
	}
}

func emptyJSONContainer(v string) bool {
	if len(v) < 2 || (v[0] != '{' && v[0] != '[') {
		return false
	}
	return strings.TrimSpace(v[1:len(v)-1]) == ""
}

// TrailingTurnKind tells what the request sent after the model's last turn,
// given the tool results of that turn: tool results, a user turn, or
// anything else, such as a prefill of the model's answer.
func (g *GenAI) TrailingTurnKind(results []ToolResult) string {
	if len(results) > 0 {
		return TrailingTurnToolResults
	}
	if g.endsWithUserTurn() {
		return TrailingTurnUser
	}

	return TrailingTurnOther
}

func (g *GenAI) endsWithUserTurn() bool {
	history, isModelTurn := g.historyTurns()
	if len(history) > 0 {
		return trailingHasRole(history, isModelTurn, "user")
	}
	if ai := g.openAIFamily(); ai != nil {
		return ai.Request.Input != "" || ai.Request.Prompt != ""
	}

	return false
}

// ContinuesConversation tells whether the request carries an earlier turn of
// the model: it continues a stored response, or its history holds a model
// turn. A history the capture cut is looked at up to the cut.
func (g *GenAI) ContinuesConversation() bool {
	if g.PreviousResponseID() != "" {
		return true
	}
	history, isModelTurn := g.historyTurns()
	spans := arrayElementSpans(history)
	for i := len(spans) - 1; i >= 0; i-- {
		if isModelTurn(history[spans[i].start:spans[i].end]) {
			return true
		}
	}

	return false
}

// historyTurns returns the conversation array of the request and the
// predicate that tells the model's turns in it.
func (g *GenAI) historyTurns() (json.RawMessage, func(elem []byte) bool) {
	if ai := g.openAIFamily(); ai != nil {
		if len(ai.Request.InputItems) > 0 {
			return ai.Request.InputItems, isResponsesModelItem
		}
		return ai.Request.Messages, hasRole("assistant")
	}

	switch {
	case g.Anthropic != nil:
		return g.Anthropic.Input.Messages, hasRole("assistant")
	case g.Gemini != nil:
		return g.Gemini.Input.Contents, hasRole("model")
	case g.Bedrock != nil:
		return g.Bedrock.Input.Messages, hasRole("assistant")
	}

	return nil, nil
}

// OffersTools tells whether the request declared tools the model may call.
func (g *GenAI) OffersTools() bool {
	if ai := g.openAIFamily(); ai != nil {
		return declared(ai.Request.Tools)
	}

	switch {
	case g.Anthropic != nil:
		return declared(g.Anthropic.Input.Tools)
	case g.Gemini != nil:
		return declared(g.Gemini.Input.Tools)
	case g.Bedrock != nil:
		return declared(g.Bedrock.Input.Tools) || declared(g.Bedrock.Input.ToolConfig)
	}

	return false
}

func declared(raw json.RawMessage) bool {
	v := strings.TrimSpace(string(raw))
	return v != "" && v != "null" && !emptyJSONContainer(v)
}

func hasRole(role string) func(elem []byte) bool {
	return func(elem []byte) bool {
		var m struct {
			Role string `json:"role"`
		}
		return json.Unmarshal(elem, &m) == nil && m.Role == role
	}
}

// trailingHasRole tells whether an element of the trailing turn has the role.
func trailingHasRole(raw []byte, isModelTurn func(elem []byte) bool, role string) bool {
	is := hasRole(role)
	for _, elem := range trailingTurn(raw, isModelTurn) {
		if is(elem) {
			return true
		}
	}

	return false
}

// trailingTurn returns the elements of a JSON array after the last one for
// which isModelTurn holds, all of them when none does. Elements are only
// split, not decoded, and the model turns are looked for from the end, so the
// cost of a long history is one scan of its bytes. A malformed array, such as
// one cut by the capture, yields nothing.
func trailingTurn(raw []byte, isModelTurn func(elem []byte) bool) [][]byte {
	elems, ok := jsonArrayElements(raw)
	if !ok {
		return nil
	}
	for i := len(elems) - 1; i >= 0; i-- {
		if isModelTurn(elems[i]) {
			return elems[i+1:]
		}
	}

	return elems
}

// jsonArrayElements splits a JSON array into its top-level elements without
// decoding them. It reports false when raw is not one complete array.
func jsonArrayElements(raw []byte) ([][]byte, bool) {
	i := skipJSONSpace(raw, 0)
	if i >= len(raw) || raw[i] != '[' {
		return nil, false
	}
	i = skipJSONSpace(raw, i+1)
	if i < len(raw) && raw[i] == ']' {
		return nil, skipJSONSpace(raw, i+1) == len(raw)
	}

	var elems [][]byte
	for i < len(raw) {
		end := jsonValueEnd(raw, i)
		if end < 0 {
			return nil, false
		}
		elems = append(elems, raw[i:end])
		i = skipJSONSpace(raw, end)
		if i >= len(raw) {
			return nil, false
		}
		switch raw[i] {
		case ',':
			i = skipJSONSpace(raw, i+1)
		case ']':
			return elems, skipJSONSpace(raw, i+1) == len(raw)
		default:
			return nil, false
		}
	}

	return nil, false
}

type elementSpan struct {
	start, end int
}

// arrayElementSpans returns the bounds of the top-level elements of a JSON
// array, up to the first element the capture cut. Elements are only split,
// not decoded.
func arrayElementSpans(raw []byte) []elementSpan {
	i := skipJSONSpace(raw, 0)
	if i >= len(raw) || raw[i] != '[' {
		return nil
	}

	var spans []elementSpan
	for i = skipJSONSpace(raw, i+1); i < len(raw) && raw[i] != ']'; {
		end := jsonValueEnd(raw, i)
		if end < 0 {
			break
		}
		spans = append(spans, elementSpan{start: i, end: end})
		i = skipJSONSpace(raw, end)
		if i >= len(raw) || raw[i] != ',' {
			break
		}
		i = skipJSONSpace(raw, i+1)
	}

	return spans
}

// ArrayElementEnds returns the offset after each top-level element of a JSON
// array, up to the first element the capture cut: the bytes before each
// offset are the array with its later elements left out.
func ArrayElementEnds(raw []byte) []int {
	spans := arrayElementSpans(raw)
	ends := make([]int, len(spans))
	for i := range spans {
		ends[i] = spans[i].end
	}

	return ends
}

// jsonValueEnd returns the index after the JSON value that starts at i, or -1
// when it does not end within raw.
func jsonValueEnd(raw []byte, i int) int {
	switch raw[i] {
	case '"':
		return jsonStringEnd(raw, i)
	case '{', '[':
		depth := 0
		for j := i; j < len(raw); j++ {
			switch raw[j] {
			case '"':
				end := jsonStringEnd(raw, j)
				if end < 0 {
					return -1
				}
				j = end - 1
			case '{', '[':
				depth++
			case '}', ']':
				depth--
				if depth == 0 {
					return j + 1
				}
			}
		}
		return -1
	}

	j := i
	for j < len(raw) {
		switch raw[j] {
		case ',', ']', '}', ' ', '\t', '\n', '\r':
			if j == i {
				return -1
			}
			return j
		}
		j++
	}

	return j
}

// jsonStringEnd returns the index after the string whose quote is at i, or -1.
func jsonStringEnd(raw []byte, i int) int {
	for j := i + 1; j < len(raw); j++ {
		switch raw[j] {
		case '\\':
			j++
		case '"':
			return j + 1
		}
	}

	return -1
}

func skipJSONSpace(raw []byte, i int) int {
	for i < len(raw) {
		switch raw[i] {
		case ' ', '\t', '\n', '\r':
			i++
		default:
			return i
		}
	}

	return i
}

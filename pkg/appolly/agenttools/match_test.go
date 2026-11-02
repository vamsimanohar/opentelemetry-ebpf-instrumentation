// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package agenttools

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/trace"

	"go.opentelemetry.io/obi/pkg/appolly/app"
	"go.opentelemetry.io/obi/pkg/appolly/app/agentact"
	"go.opentelemetry.io/obi/pkg/appolly/app/request"
	"go.opentelemetry.io/obi/pkg/appolly/app/svc"
	"go.opentelemetry.io/obi/pkg/config"
)

const (
	agentPID = 1000
	llmPort  = 443
	ms       = int64(time.Millisecond)
)

func testConfig() *config.GenAIToolCallsConfig {
	return &config.GenAIToolCallsConfig{
		Enabled:            true,
		MaxWindow:          10 * time.Minute,
		OnTimeout:          config.ToolCallTimeoutEmit,
		CloseOnNextRequest: true,
		MatchByName:        true,
		EmitUnmatched:      true,
		KernelActivity: config.GenAIToolKernelActivityConfig{
			Exec:                true,
			FileOpen:            true,
			Connect:             true,
			MaxLineageDepth:     8,
			PathExclude:         []string{"*/lib/python3*"},
			ExcludeLLMEndpoints: true,
			MaxEventsPerWindow:  512,
			MaxEventsPerSpan:    64,
			RateLimitPerProcess: 2000,
		},
	}
}

func testTracker(cfg *config.GenAIToolCallsConfig) *tracker {
	return newTracker(cfg, 0, nil)
}

var spanSeq byte

// inference builds an OpenAI chat span: its request carries the messages,
// its response the tool calls.
func inference(start, end int64, messages string, calls ...request.ToolCall) request.Span {
	spanSeq++
	for i := range calls {
		calls[i].Index = i
	}
	return request.Span{
		Type:         request.EventTypeHTTPClient,
		SubType:      request.HTTPSubtypeOpenAI,
		Host:         "10.0.0.1",
		HostPort:     llmPort,
		Start:        start,
		RequestStart: start,
		End:          end,
		TraceID:      trace.TraceID{spanSeq, 1},
		SpanID:       trace.SpanID{spanSeq, 2},
		ParentSpanID: trace.SpanID{0xee},
		TID:          10,
		Pid:          request.PidInfo{HostPID: agentPID},
		Service:      svc.Attrs{UID: svc.UID{Name: "agent"}},
		GenAI: &request.GenAI{OpenAI: &request.VendorOpenAI{
			ID:        "chatcmpl-" + string(rune('a'+spanSeq)),
			ToolCalls: calls,
			Request:   request.OpenAIInput{Messages: json.RawMessage(messages)},
		}},
	}
}

func call(id, name, args string) request.ToolCall {
	return request.ToolCall{ID: id, Name: name, Arguments: args, Type: "function"}
}

const noTools = `[{"role":"user","content":"hi"}]`

// conversation builds the chat history of one conversation: the first
// message, then for every turn the assistant calls and their results.
type chat struct {
	first string
	turns []string
}

func newChat(first string) *chat {
	return &chat{first: `{"role":"user","content":"` + first + `"}`}
}

// history is the conversation the next request sends.
func (c *chat) history() string {
	return "[" + strings.Join(append([]string{c.first}, c.turns...), ",") + "]"
}

// answer appends the assistant turn calling ids and the results of the ids.
func (c *chat) answer(ids ...string) *chat {
	var calls []string
	for _, id := range ids {
		calls = append(calls, `{"id":"`+id+`"}`)
	}
	c.turns = append(c.turns, `{"role":"assistant","tool_calls":[`+strings.Join(calls, ",")+`]}`)
	for _, id := range ids {
		c.turns = append(c.turns, `{"role":"tool","tool_call_id":"`+id+`","content":"result of `+id+`"}`)
	}
	return c
}

// say appends turns that carry no structured result.
func (c *chat) say(turns ...string) *chat {
	c.turns = append(c.turns, turns...)
	return c
}

// results builds chat messages whose trailing turn answers the given ids,
// continuing the noTools conversation.
func results(ids ...string) string {
	return newChat("hi").answer(ids...).history()
}

func toolSpans(spans []request.Span) []request.Span {
	var out []request.Span
	for _, s := range spans {
		if s.InferredTool != nil || s.Process != nil {
			out = append(out, s)
		}
	}
	return out
}

func feed(t *testing.T, tr *tracker, spans ...request.Span) []request.Span {
	t.Helper()
	var out []request.Span
	for _, s := range spans {
		out = append(out, toolSpans(tr.onSpans([]request.Span{s}))...)
	}
	return out
}

func TestSingleCallClosedByID(t *testing.T) {
	tr := testTracker(testConfig())
	n := inference(0, 100*ms, noTools, call("call_Q1rT7m2Lx9Pz", "get_weather", `{"city":"Paris"}`))
	n1 := inference(108*ms, 150*ms, results("call_Q1rT7m2Lx9Pz"))

	assert.Empty(t, feed(t, tr, n))
	out := feed(t, tr, n1)
	require.Len(t, out, 1)

	s := out[0]
	assert.Equal(t, request.EventTypeGenAIToolExec, s.Type)
	assert.Equal(t, 100*ms, s.Start)
	assert.Equal(t, 108*ms, s.End)
	assert.Equal(t, n.TraceID, s.TraceID, "sibling of the requesting inference, in its trace")
	assert.Equal(t, n.ParentSpanID, s.ParentSpanID)
	assert.True(t, request.IgnoreMetrics(&s))
	require.Len(t, s.Links, 2)
	assert.Equal(t, n.SpanID, s.Links[0].SpanID)
	assert.Equal(t, linkKindSource, s.Links[0].Attributes[0].Value.AsString())
	assert.Equal(t, n1.SpanID, s.Links[1].SpanID)

	tool := s.InferredTool
	assert.Equal(t, "get_weather", tool.Name)
	assert.Equal(t, "call_Q1rT7m2Lx9Pz", tool.CallID)
	assert.Equal(t, "openai", tool.Provider)
	assert.Equal(t, request.ToolOutcomeResult, tool.Outcome)
	assert.Equal(t, request.ToolWindowStartResponseEnd, tool.WindowStart)
	assert.Equal(t, request.ToolWindowEndToolResult, tool.WindowEnd)
	assert.Equal(t, 1, tool.ParallelCount)
	assert.JSONEq(t, `"result of call_Q1rT7m2Lx9Pz"`, tool.Result)
	assert.False(t, tool.ActivityObserved)
}

func TestResultArrivesBeforeResponse(t *testing.T) {
	tr := testTracker(testConfig())
	n := inference(0, 100*ms, noTools, call("call_Q1rT7m2Lx9Pz", "get_weather", `{}`))
	n1 := inference(110*ms, 150*ms, results("call_Q1rT7m2Lx9Pz"))

	// response spans flush late: N+1 reaches the stage first
	assert.Empty(t, feed(t, tr, n1))
	out := feed(t, tr, n)
	require.Len(t, out, 1)
	assert.Equal(t, request.ToolOutcomeResult, out[0].InferredTool.Outcome)
	assert.Equal(t, 110*ms, out[0].End)
}

func TestHistoryReplayDoesNotReopen(t *testing.T) {
	tr := testTracker(testConfig())
	c := newChat("hi")
	n := inference(0, 100*ms, c.history(), call("call_r1aD4bF7hJ2k", "read_file", `{}`))
	n1 := inference(110*ms, 200*ms, c.answer("call_r1aD4bF7hJ2k").history(), call("call_c2Kd9sLm3Qw8", "run_command", `{}`))
	// the stateless history replays call 1 and its result before the new turn
	n2 := inference(210*ms, 300*ms, c.answer("call_c2Kd9sLm3Qw8").history())
	// the same response reaching the stage twice must not reopen its calls
	dupN := n
	dupN.SpanID = trace.SpanID{0xdd}

	out := feed(t, tr, n, n1, n2, dupN)
	require.Len(t, out, 2)
	assert.Equal(t, "call_r1aD4bF7hJ2k", out[0].InferredTool.CallID)
	assert.Equal(t, 110*ms, out[0].End)
	assert.Equal(t, "call_c2Kd9sLm3Qw8", out[1].InferredTool.CallID)
	assert.Equal(t, 210*ms, out[1].End)
	assert.Empty(t, tr.store.procs.Values()[0].pending)
}

func TestParallelCallsShareWindow(t *testing.T) {
	tr := testTracker(testConfig())
	n := inference(0, 100*ms, noTools,
		call("call_aaaaaaaaaaaa", "read_file", `{}`), call("call_bbbbbbbbbbbb", "run_command", `{}`), call("call_cccccccccccc", "http_get", `{}`))
	n1 := inference(120*ms, 150*ms, results("call_aaaaaaaaaaaa", "call_bbbbbbbbbbbb", "call_cccccccccccc"))

	out := feed(t, tr, n, n1)
	require.Len(t, out, 4)

	window := out[0]
	assert.Equal(t, request.EventTypeGenAIToolWindow, window.Type)
	assert.Equal(t, n.ParentSpanID, window.ParentSpanID)
	assert.Equal(t, 3, window.InferredTool.ParallelCount)
	for _, s := range out[1:] {
		assert.Equal(t, request.EventTypeGenAIToolExec, s.Type)
		assert.Equal(t, window.SpanID, s.ParentSpanID, "parallel calls nest under their window")
		assert.Equal(t, 3, s.InferredTool.ParallelCount)
		assert.Equal(t, 100*ms, s.Start)
		assert.Equal(t, 120*ms, s.End)
	}
}

func TestStatefulContinuation(t *testing.T) {
	tr := testTracker(testConfig())
	n := inference(0, 100*ms, noTools, call("call_s7Dk2Qm4Lp9x", "get_weather", `{}`))
	n.GenAI.OpenAI.ID = "resp_1"

	// previous_response_id: the request carries only the output item
	n1 := inference(110*ms, 150*ms, "")
	n1.GenAI.OpenAI.Request = request.OpenAIInput{
		PreviousResponseID: "resp_1",
		InputItems:         json.RawMessage(`[{"type":"function_call_output","call_id":"call_s7Dk2Qm4Lp9x","output":"rainy"}]`),
	}

	out := feed(t, tr, n, n1)
	require.Len(t, out, 1)
	assert.Equal(t, request.ToolOutcomeResult, out[0].InferredTool.Outcome)
	assert.JSONEq(t, `"rainy"`, out[0].InferredTool.Result)
}

func TestIDLessResultsMatchByNameThenOrder(t *testing.T) {
	tr := testTracker(testConfig())
	c := newChat("hi")
	n := inference(0, 100*ms, c.history(), call("call_aaaaaaaaaaaa", "read_file", `{}`), call("call_bbbbbbbbbbbb", "run_command", `{}`))
	// CrewAI with LiteLLM streaming drops the ids from the history it sends back
	n1 := inference(110*ms, 150*ms, c.say(`{"role":"assistant","tool_calls":[{"id":null}]}`,
		`{"role":"tool","name":"run_command","content":"Linux"}`,
		`{"role":"tool","content":"alpha"}`).history())

	out := feed(t, tr, n, n1)
	require.Len(t, out, 3)
	byID := map[string]*request.InferredTool{}
	for _, s := range out[1:] {
		byID[s.InferredTool.CallID] = s.InferredTool
	}
	assert.Equal(t, request.ToolOutcomeResultByName, byID["call_bbbbbbbbbbbb"].Outcome)
	assert.JSONEq(t, `"Linux"`, byID["call_bbbbbbbbbbbb"].Result)
	assert.Equal(t, request.ToolOutcomeResultByName, byID["call_aaaaaaaaaaaa"].Outcome)
	assert.JSONEq(t, `"alpha"`, byID["call_aaaaaaaaaaaa"].Result)
}

func TestIDLessResultsOfAnotherConversationDoNotMatch(t *testing.T) {
	tr := testTracker(testConfig())
	n := inference(0, 100*ms, newChat("task A").history(), call("", "read_file", `{}`))
	// conversation B sends an id-less result for its own read_file
	other := inference(110*ms, 150*ms, newChat("task B").say(`{"role":"assistant","tool_calls":[{"id":null}]}`,
		`{"role":"tool","name":"read_file","content":"b"}`).history())

	assert.Empty(t, feed(t, tr, n, other))
}

func TestNextRequestOfTheSameConversation(t *testing.T) {
	tr := testTracker(testConfig())
	c := newChat("find the answer")
	n := inference(0, 100*ms, c.history(), call("call_w3bS3archQ1x", "web_search", `{}`))
	// a request of another conversation is not the agent's next model call
	other := inference(105*ms, 106*ms, newChat("something else").history())
	// smolagents sends results as plain text after a turn naming the call
	n1 := inference(110*ms, 150*ms, c.say(
		`{"role":"assistant","content":"Calling tools: [{'id': 'call_w3bS3archQ1x', 'type': 'function'}]"}`,
		`{"role":"user","content":"Observation: 42"}`).history())
	// on another IP of the provider: the conversation decides, not the address
	n1.Host = "10.0.0.2"

	assert.Empty(t, feed(t, tr, n, other))
	out := feed(t, tr, n1)
	require.Len(t, out, 1)
	assert.Equal(t, request.ToolOutcomeNextRequest, out[0].InferredTool.Outcome)
	assert.Equal(t, request.ToolWindowEndNextRequest, out[0].InferredTool.WindowEnd)
	assert.Empty(t, out[0].InferredTool.Result)
	assert.Equal(t, 110*ms, out[0].End)
}

func TestNextRequestByHistoryWithoutIDs(t *testing.T) {
	tr := testTracker(testConfig())
	c := newChat("summarise")
	// Ollama never sends ids; the result was cut by the capture
	n := inference(0, 100*ms, c.history(), call("", "fetch", `{}`))
	n1 := inference(110*ms, 150*ms, c.say(`{"role":"assistant","content":""}`, `{"role":"user","content":"go on"}`).history())

	out := feed(t, tr, n, n1)
	require.Len(t, out, 1)
	assert.Equal(t, request.ToolOutcomeNextRequest, out[0].InferredTool.Outcome)
}

func TestNestedModelCallDoesNotCloseWindow(t *testing.T) {
	tr := testTracker(testConfig())
	c := newChat("research this")
	n := inference(0, 100*ms, c.history(), call("call_summ4rize0001", "summarize", `{}`))
	// the tool calls a model of its own, in a conversation of its own
	sub := inference(150*ms, 900*ms, newChat("summarize this").history())
	n1 := inference(1000*ms, 1100*ms, c.answer("call_summ4rize0001").history())

	out := feed(t, tr, n, sub, n1)
	require.Len(t, out, 1)
	assert.Equal(t, 1000*ms, out[0].End, "the window ends when the result was sent")
	assert.Equal(t, request.ToolOutcomeResult, out[0].InferredTool.Outcome)
}

func TestConcurrentConversationsCloseOnTheirOwnResults(t *testing.T) {
	tr := testTracker(testConfig())
	ca, cb := newChat("conversation A"), newChat("conversation B")
	a := inference(0, 100*ms, ca.history(), call("call_aaaaaaaaaaaa", "read_file", `{"path":"x"}`))
	b := inference(10*ms, 120*ms, cb.history(), call("call_bbbbbbbbbbbb", "run_command", `{"cmd":"ls"}`))
	ra := inference(200*ms, 300*ms, ca.answer("call_aaaaaaaaaaaa").history())
	rb := inference(210*ms, 300*ms, cb.answer("call_bbbbbbbbbbbb").history())

	out := feed(t, tr, a, b, ra, rb)
	require.Len(t, out, 2)
	for _, s := range out {
		assert.Equal(t, request.ToolOutcomeResult, s.InferredTool.Outcome)
	}
	assert.Equal(t, 200*ms, out[0].End)
	assert.Equal(t, 210*ms, out[1].End)
}

func TestReusedCallIDsAcrossResponses(t *testing.T) {
	tr := testTracker(testConfig())
	// local OpenAI-compatible servers number calls per response
	c := newChat("read both")
	n := inference(0, 100*ms, c.history(), call("call_0", "read_file", `{"path":"a"}`))
	n1 := inference(110*ms, 200*ms, c.answer("call_0").history(), call("call_0", "read_file", `{"path":"b"}`))
	n2 := inference(300*ms, 400*ms, c.answer("call_0").history())

	out := feed(t, tr, n, n1, n2)
	require.Len(t, out, 2, "a reused id opens a new call")
	assert.Equal(t, 110*ms, out[0].End)
	assert.Contains(t, out[0].InferredTool.Arguments, `"a"`)
	assert.Equal(t, 300*ms, out[1].End)
	assert.Contains(t, out[1].InferredTool.Arguments, `"b"`)
}

func TestReusedCallIDsOfAnotherConversation(t *testing.T) {
	tr := testTracker(testConfig())
	ca, cb := newChat("conversation A"), newChat("conversation B")
	a := inference(0, 100*ms, ca.history(), call("call_0", "read_file", `{}`))
	// B's result for its own call_0, whose response OBI has not seen yet
	rb := inference(150*ms, 200*ms, cb.answer("call_0").history())

	assert.Empty(t, feed(t, tr, a, rb), "a short id alone does not pair across conversations")
}

func TestUnparsedRequestIDScan(t *testing.T) {
	cfg := testConfig()
	cfg.CloseOnNextRequest = false
	tr := testTracker(cfg)
	n := inference(0, 100*ms, noTools, call("call_hole_7f3a9c21", "read_file", `{}`))
	n1 := inference(110*ms, 150*ms, "")
	n1.GenAI.UnparsedRequest = []byte(`{"messages":[{"role":"tool","tool_call_id":"call_hole_7f3a9c21","con` + "\x00\x00" + `tent":"x"}`)

	out := feed(t, tr, n, n1)
	require.Len(t, out, 1)
	assert.Equal(t, request.ToolOutcomeResult, out[0].InferredTool.Outcome)
}

func TestUnparsedRequestNeedsTheWholeID(t *testing.T) {
	cfg := testConfig()
	cfg.CloseOnNextRequest = false
	tr := testTracker(cfg)
	n := inference(0, 100*ms, noTools, call("call_hole_7f3a9c21", "read_file", `{}`))
	n1 := inference(110*ms, 150*ms, "")
	n1.GenAI.UnparsedRequest = []byte(`{"messages":[{"role":"tool","tool_call_id":"call_hole_7f3a9c210","con` + "\x00")

	assert.Empty(t, feed(t, tr, n, n1))
}

func TestUnmatchedTimeout(t *testing.T) {
	for _, tc := range []struct {
		name   string
		adjust func(*config.GenAIToolCallsConfig)
		emits  bool
	}{
		{name: "emitted", adjust: func(*config.GenAIToolCallsConfig) {}, emits: true},
		{name: "on_timeout drop", adjust: func(c *config.GenAIToolCallsConfig) { c.OnTimeout = config.ToolCallTimeoutDrop }},
		{name: "emit_unmatched off", adjust: func(c *config.GenAIToolCallsConfig) { c.EmitUnmatched = false }},
		{name: "below unmatched_min_activity", adjust: func(c *config.GenAIToolCallsConfig) { c.UnmatchedMinActivity = 1 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := testConfig()
			cfg.MaxWindow = time.Second
			tc.adjust(cfg)
			tr := testTracker(cfg)

			assert.Empty(t, feed(t, tr, inference(0, 100*ms, noTools, call("call_f1nalAnsw3r", "final_answer", `{}`))))
			assert.Empty(t, toolSpans(tr.tick(900*ms, time.Now())))

			out := toolSpans(tr.tick(1200*ms, time.Now()))
			if !tc.emits {
				assert.Empty(t, out)
				return
			}
			require.Len(t, out, 1)
			assert.Equal(t, request.ToolOutcomeUnmatchedTimeout, out[0].InferredTool.Outcome)
			assert.Equal(t, request.ToolWindowEndTimeout, out[0].InferredTool.WindowEnd)
			assert.Equal(t, 1100*ms, out[0].End)
			assert.Empty(t, out[0].Links[1:], "nothing answered the call")
		})
	}
}

func TestUnmatchedCallEndsAtTheNextModelCall(t *testing.T) {
	cfg := testConfig()
	cfg.MaxWindow = time.Second
	tr := testTracker(cfg)

	// the result was lost and nothing ties the next request to the call
	assert.Empty(t, feed(t, tr,
		inference(0, 100*ms, newChat("one").history(), call("call_l0stResult01", "read_file", `{}`)),
		inference(250*ms, 300*ms, newChat("two").history())))

	out := toolSpans(tr.tick(1200*ms, time.Now()))
	require.Len(t, out, 1)
	assert.Equal(t, request.ToolOutcomeUnmatchedTimeout, out[0].InferredTool.Outcome)
	assert.Equal(t, request.ToolWindowEndNextInference, out[0].InferredTool.WindowEnd)
	assert.Equal(t, 250*ms, out[0].End)
}

func TestUnmatchedProcessExit(t *testing.T) {
	tr := testTracker(testConfig())
	assert.Empty(t, feed(t, tr, inference(0, 100*ms, noTools, call("call_f1nalAnsw3r", "final_answer", `{}`))))

	tr.onProcessTerminated(app.PID(agentPID), 300*ms)
	// the exit waits for the process's last spans to flush
	assert.Empty(t, toolSpans(tr.tick(301*ms, time.Now())))

	out := toolSpans(tr.tick(302*ms, time.Now().Add(minExitGrace)))
	require.Len(t, out, 1)
	assert.Equal(t, request.ToolOutcomeUnmatchedExit, out[0].InferredTool.Outcome)
	assert.Equal(t, request.ToolWindowEndProcessExit, out[0].InferredTool.WindowEnd)
	assert.Equal(t, 300*ms, out[0].End)

	// a response flushed after the process state was dropped waits for the
	// spans that may follow it, then closes at the exit
	assert.Empty(t, feed(t, tr, inference(150*ms, 200*ms, noTools, call("call_f1nalAnsw3r2", "final_answer", `{}`))))
	out = toolSpans(tr.tick(303*ms, time.Now().Add(minExitGrace)))
	require.Len(t, out, 1)
	assert.Equal(t, request.ToolOutcomeUnmatchedExit, out[0].InferredTool.Outcome)
	assert.Equal(t, 300*ms, out[0].End)
}

func TestShutdownFlushesExitedProcesses(t *testing.T) {
	cfg := testConfig()
	cfg.KernelActivity.Enabled = true
	tr := testTracker(cfg)
	feed(t, tr, inference(0, 100*ms, newChat("finish").history(), call("call_f1nalAnsw3r", "final_answer", `{}`)))
	feed(t, tr, inference(0, 100*ms, noTools, call("call_r3adFile0001", "read_file", `{}`)),
		inference(110*ms, 150*ms, results("call_r3adFile0001")))
	tr.onProcessTerminated(app.PID(agentPID), 300*ms)

	out := toolSpans(tr.flush())
	require.Len(t, out, 2, "the settling window and the call the exit left unanswered")
	outcomes := []string{out[0].InferredTool.Outcome, out[1].InferredTool.Outcome}
	assert.ElementsMatch(t, []string{request.ToolOutcomeUnmatchedExit, request.ToolOutcomeResult}, outcomes)
}

func TestReusedPIDDoesNotInheritExit(t *testing.T) {
	tr := testTracker(testConfig())
	feed(t, tr, inference(0, 100*ms, noTools))
	tr.onProcessTerminated(app.PID(agentPID), 200*ms)
	tr.tick(300*ms, time.Now().Add(minExitGrace))

	// discovery reports a new process with the same pid, running a 20s tool
	start := int64(3600 * time.Second)
	tr.store.forget(app.PID(agentPID))
	n := inference(start, start+100*ms, noTools, call("call_l0ngTool0001", "long_tool", `{}`))
	assert.Empty(t, feed(t, tr, n))
	assert.Empty(t, toolSpans(tr.tick(start+16*int64(time.Second), time.Now().Add(minExitGrace))),
		"a live process is not treated as exited")
}

func TestExitedPIDIsForgottenAfterAWhile(t *testing.T) {
	s := newStore()
	s.exited.Add(app.PID(agentPID), exitRecord{at: 1, seenAt: time.Now().Add(-exitedTTL)})
	assert.False(t, s.agent(request.PidInfo{HostPID: agentPID}).exited)
}

func TestSpansArrivingAfterExitStillPair(t *testing.T) {
	cfg := testConfig()
	cfg.KernelActivity.Enabled = true
	tr := testTracker(cfg)

	// the kernel reports the exit before the process's spans are flushed
	tr.onActivity([]agentact.Event{{Kind: agentact.KindExit, Time: 300 * ms, PID: agentPID, TID: agentPID, RootPID: agentPID}})
	n := inference(0, 100*ms, noTools, call("call_r3adFile0001", "read_file", `{}`))
	n1 := inference(110*ms, 150*ms, results("call_r3adFile0001"))
	assert.Empty(t, feed(t, tr, n, n1))

	out := toolSpans(tr.tick(0, time.Now().Add(minExitGrace)))
	require.Len(t, out, 1)
	assert.Equal(t, request.ToolOutcomeResult, out[0].InferredTool.Outcome)
}

func TestServerSideAndEmbeddingSpansIgnored(t *testing.T) {
	tr := testTracker(testConfig())
	server := call("ws_1", "web_search", `{}`)
	server.ServerSide = true
	n := inference(0, 100*ms, noTools, server)
	embedding := inference(0, 100*ms, noTools, call("call_e", "x", `{}`))
	embedding.SubType = request.HTTPSubtypeEmbedding

	assert.Empty(t, feed(t, tr, n, embedding))
	for _, p := range tr.store.procs.Values() {
		assert.Empty(t, p.pending)
	}
}

func TestIDLessCallsPairInOrderWithoutSynthesizedIDs(t *testing.T) {
	tr := testTracker(testConfig())
	c := newChat("weather")
	// Ollama never sends ids, Gemini may not: same-name calls pair in order
	n := inference(0, 100*ms, c.history(), call("", "get_weather", `{"city":"Paris"}`), call("", "get_weather", `{"city":"Rome"}`))
	n1 := inference(110*ms, 150*ms, c.say(`{"role":"assistant","tool_calls":[{"function":{"name":"get_weather"}}]}`,
		`{"role":"tool","tool_name":"get_weather","content":"rain"}`,
		`{"role":"tool","tool_name":"get_weather","content":"sun"}`).history())
	n1.SubType = request.HTTPSubtypeOllama
	n1.GenAI = &request.GenAI{Ollama: n1.GenAI.OpenAI}

	out := feed(t, tr, n, n1)
	require.Len(t, out, 3)
	for i, want := range []string{`"rain"`, `"sun"`} {
		tool := out[i+1].InferredTool
		assert.Empty(t, tool.CallID, "no id is synthesized")
		assert.Equal(t, request.ToolOutcomeResultByName, tool.Outcome)
		assert.JSONEq(t, want, tool.Result)
	}
	assert.Contains(t, out[1].InferredTool.Arguments, "Paris")
}

func TestLLMGatewayGetsNoToolSpans(t *testing.T) {
	tr := testTracker(testConfig())
	// a proxy forwards an agent's request, whose response asks for a tool
	served := request.Span{Type: request.EventTypeHTTP, Method: "POST", Path: "/v1/chat/completions",
		Pid: request.PidInfo{HostPID: agentPID}}
	n := inference(0, 100*ms, noTools, call("call_g4tewayCall1", "read_file", `{}`))
	n1 := inference(110*ms, 150*ms, results("call_g4tewayCall1"))

	out := toolSpans(tr.onSpans([]request.Span{n, served}))
	out = append(out, feed(t, tr, n1)...)
	assert.Empty(t, out)
}

func TestLLMRoutes(t *testing.T) {
	for path, want := range map[string]bool{
		"/v1/chat/completions":                              true,
		"/openai/deployments/gpt/chat/completions?api=2024": true,
		"/v1/messages":                                      true,
		"/v1beta/models/gemini-2.5-flash:generateContent":   true,
		"/model/anthropic.claude-3-haiku/converse-stream":   true,
		"/responses":           true,
		"/api/chat":            false,
		"/v1/messages/archive": false,
		"/converse":            false,
	} {
		s := request.Span{Type: request.EventTypeHTTP, Path: path}
		assert.Equal(t, want, servesLLMRoute(&s), path)
	}
	client := request.Span{Type: request.EventTypeHTTPClient, Path: "/v1/chat/completions"}
	assert.False(t, servesLLMRoute(&client))
}

func TestServiceSelection(t *testing.T) {
	for _, tc := range []struct {
		name             string
		include, exclude []string
		emits            bool
	}{
		{name: "everything by default", emits: true},
		{name: "included", include: []string{"agent*"}, emits: true},
		{name: "not included", include: []string{"worker"}},
		{name: "excluded", exclude: []string{"agent"}},
		{name: "exclude wins", include: []string{"*"}, exclude: []string{"age?t"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := testConfig()
			cfg.Services = config.GenAIToolCallsServices{Include: tc.include, Exclude: tc.exclude}
			tr := testTracker(cfg)
			out := feed(t, tr, inference(0, 100*ms, noTools, call("call_s3lect3d0001", "read_file", `{}`)),
				inference(110*ms, 150*ms, results("call_s3lect3d0001")))
			assert.Equal(t, tc.emits, len(out) == 1)
		})
	}
}

func TestTruncatedArgumentsAreMarked(t *testing.T) {
	tr := testTracker(testConfig())
	long := `{"text":"` + strings.Repeat("é", maxPayloadBytes) + `"}`
	out := feed(t, tr, inference(0, 100*ms, noTools, call("call_l0ngArgs0001", "write_note", long)),
		inference(110*ms, 150*ms, results("call_l0ngArgs0001")))
	require.Len(t, out, 1)
	tool := out[0].InferredTool
	assert.True(t, tool.PayloadTruncated)
	assert.LessOrEqual(t, len(tool.Arguments), maxPayloadBytes)
	assert.True(t, utf8Valid(tool.Arguments), "cut on a rune boundary")
}

func utf8Valid(s string) bool {
	return strings.ToValidUTF8(s, "�") == s
}

func TestConversationContinuation(t *testing.T) {
	tr := testTracker(testConfig())
	c := newChat("hello")
	conv := conversationOf(tr.seed, []byte(c.history()))
	require.True(t, conv.known())

	assert.True(t, conv.continuedBy(tr.seed, []byte(c.answer("call_x").history())))
	assert.False(t, conv.continuedBy(tr.seed, []byte(newChat("hello").history())), "the same history is no continuation")
	assert.False(t, conv.continuedBy(tr.seed, []byte(newChat("other").answer("call_x").history())))
	empty := conversationOf(tr.seed, []byte(`[]`))
	assert.False(t, empty.known(), "every conversation starts empty")

	cut := conversationOf(tr.seed, []byte(`[{"role":"user","content":"hel`))
	assert.True(t, cut.continuedBy(tr.seed, []byte(`[{"role":"user","content":"hel`)), "both cut by the capture")
}

func TestContainsToken(t *testing.T) {
	assert.True(t, containsToken([]byte(`[{'id': 'call_abc'}]`), "call_abc"))
	assert.False(t, containsToken([]byte(`"call_abc1"`), "call_abc"))
	assert.False(t, containsToken([]byte(`"xcall_abc"`), "call_abc"))
	assert.True(t, containsToken([]byte(`"xcall_abc" "call_abc"`), "call_abc"))
}

// BenchmarkOnSpansLargeHistory measures the stage's cost for a GenAI span
// with a long history: it runs in the span pipeline.
func BenchmarkOnSpansLargeHistory(b *testing.B) {
	c := newChat("hi")
	for len(c.history()) < 250<<10 {
		c.say(`{"role":"assistant","content":"lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor"}`,
			`{"role":"user","content":"more text more text more text more text more text more text"}`)
	}
	tr := testTracker(testConfig())
	s := inference(0, 100*ms, c.history())
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		tr.onSpans([]request.Span{s})
	}
}

// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package agenttools

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"go.opentelemetry.io/obi/pkg/appolly/app/request"
)

// annotated passes a span through the tracker and returns its annotation.
func annotated(t *testing.T, tr *tracker, span request.Span) *request.GenAIConversation {
	t.Helper()
	tr.onInference(&span, span.End)
	require.NotNil(t, span.Conversation)
	return span.Conversation
}

const (
	callA1 = "call_conversationA_0001"
	callA2 = "call_conversationA_0002"
	callB1 = "call_conversationB_0001"
)

func TestConversationSequentialAgent(t *testing.T) {
	tr := testTracker(testConfig())
	c := newChat("run the task")

	first := annotated(t, tr, inference(0, 10*ms, c.history(), call(callA1, "read_file", `{}`)))
	assert.Equal(t, request.ConversationLinkNew, first.Link)
	assert.Equal(t, request.TrailingTurnUser, first.TrailingTurn)
	assert.True(t, first.RequestsTools)

	second := annotated(t, tr, inference(20*ms, 30*ms, c.answer(callA1).history(), call(callA2, "run_command", `{}`)))
	assert.Equal(t, first.Key, second.Key)
	assert.Equal(t, request.ConversationLinkCallID, second.Link)
	assert.Equal(t, request.TrailingTurnToolResults, second.TrailingTurn)

	final := annotated(t, tr, inference(40*ms, 50*ms, c.answer(callA2).history()))
	assert.Equal(t, first.Key, final.Key)
	assert.False(t, final.RequestsTools)
}

func TestConversationHistoryPrefixWithoutIDs(t *testing.T) {
	tr := testTracker(testConfig())
	h1 := `[{"role":"system","content":"s"},{"role":"user","content":"task"}]`
	// the agent sends results as user text and the server numbers its calls
	h2 := `[{"role":"system","content":"s"},{"role":"user","content":"task"},` +
		`{"role":"assistant","content":"calling"},{"role":"user","content":"Observation: 42"}]`

	first := annotated(t, tr, inference(0, 10*ms, h1, call("call_0", "lookup", `{}`)))
	second := annotated(t, tr, inference(20*ms, 30*ms, h2))
	assert.Equal(t, first.Key, second.Key)
	assert.Equal(t, request.ConversationLinkHistoryPrefix, second.Link)
	assert.Equal(t, request.TrailingTurnUser, second.TrailingTurn)
}

func TestConversationConcurrentConversationsStayApart(t *testing.T) {
	tr := testTracker(testConfig())
	a, b := newChat("session A"), newChat("session B")

	a1 := annotated(t, tr, inference(0, 10*ms, a.history(), call(callA1, "read_file", `{}`)))
	b1 := annotated(t, tr, inference(5*ms, 15*ms, b.history(), call(callB1, "read_file", `{}`)))
	b2 := annotated(t, tr, inference(20*ms, 30*ms, b.answer(callB1).history()))
	a2 := annotated(t, tr, inference(25*ms, 35*ms, a.answer(callA1).history()))

	assert.NotEqual(t, a1.Key, b1.Key)
	assert.Equal(t, a1.Key, a2.Key)
	assert.Equal(t, b1.Key, b2.Key)
}

func TestConversationIdenticalStartsAreNotMerged(t *testing.T) {
	tr := testTracker(testConfig())
	a, b := newChat("hello"), newChat("hello")

	a1 := annotated(t, tr, inference(0, 10*ms, a.history(), call(callA1, "read_file", `{}`)))
	// the same history again starts another conversation: it appends nothing
	b1 := annotated(t, tr, inference(5*ms, 15*ms, b.history(), call(callB1, "read_file", `{}`)))
	assert.Equal(t, request.ConversationLinkNew, b1.Link)
	assert.NotEqual(t, a1.Key, b1.Key)

	// the call ids tell the continuations apart
	b2 := annotated(t, tr, inference(20*ms, 30*ms, b.answer(callB1).history()))
	a2 := annotated(t, tr, inference(25*ms, 35*ms, a.answer(callA1).history()))
	assert.Equal(t, b1.Key, b2.Key)
	assert.Equal(t, a1.Key, a2.Key)
}

func TestConversationAmbiguousPrefixStartsItsOwn(t *testing.T) {
	tr := testTracker(testConfig())
	h := `[{"role":"user","content":"hello"}]`
	next := `[{"role":"user","content":"hello"},{"role":"assistant","content":"hi"},{"role":"user","content":"more"}]`

	a := annotated(t, tr, inference(0, 10*ms, h))
	b := annotated(t, tr, inference(5*ms, 15*ms, h))
	c := annotated(t, tr, inference(20*ms, 30*ms, next))

	assert.Equal(t, request.ConversationLinkAmbiguous, c.Link)
	assert.NotEqual(t, a.Key, c.Key)
	assert.NotEqual(t, b.Key, c.Key)
}

func TestConversationLongestPrefixWins(t *testing.T) {
	tr := testTracker(testConfig())
	a, b := newChat("hello"), newChat("hello")

	annotated(t, tr, inference(0, 10*ms, a.history(), call(callA1, "read_file", `{}`)))
	b1 := annotated(t, tr, inference(5*ms, 15*ms, b.history(), call(callB1, "read_file", `{}`)))
	b2 := annotated(t, tr, inference(20*ms, 30*ms, b.answer(callB1).history()))

	// a new user turn after the answer: only the history tells, and only b's
	// longer history is a prefix of it
	b.turns = append(b.turns, `{"role":"assistant","content":"done"}`, `{"role":"user","content":"again"}`)
	b3 := annotated(t, tr, inference(40*ms, 50*ms, b.history()))
	assert.Equal(t, request.ConversationLinkHistoryPrefix, b3.Link)
	assert.Equal(t, b1.Key, b3.Key)
	assert.Equal(t, b2.Key, b3.Key)
}

func TestConversationPreviousResponse(t *testing.T) {
	tr := testTracker(testConfig())
	first := responsesInference(0, 10*ms, `[{"role":"user","content":"task"}]`, "")
	first.GenAI.OpenAI.ID = "resp_first_0123456789"
	first.GenAI.OpenAI.ToolCalls = []request.ToolCall{call("call_resp_0001", "read_file", `{}`)}
	a := annotated(t, tr, first)

	second := responsesInference(20*ms, 30*ms,
		`[{"type":"function_call_output","call_id":"call_resp_0001","output":"x"}]`, "resp_first_0123456789")
	b := annotated(t, tr, second)
	assert.Equal(t, a.Key, b.Key)
	assert.Equal(t, request.ConversationLinkPreviousResponse, b.Link)
	assert.Equal(t, request.TrailingTurnToolResults, b.TrailingTurn)
}

func responsesInference(start, end int64, input, prev string) request.Span {
	s := inference(start, end, "")
	s.GenAI.OpenAI.Request = request.OpenAIInput{InputItems: json.RawMessage(input), PreviousResponseID: prev}
	return s
}

func TestConversationReorderedPredecessorAdopts(t *testing.T) {
	tr := testTracker(testConfig())
	c := newChat("run the task")
	h1 := c.history()
	h2 := c.answer(callA1).history()
	h3 := c.answer(callA2).history()

	// the second request reaches the stage first: its predecessor is unknown
	second := annotated(t, tr, inference(20*ms, 30*ms, h2, call(callA2, "run_command", `{}`)))
	assert.Equal(t, request.ConversationLinkUnknown, second.Link)

	first := annotated(t, tr, inference(0, 10*ms, h1, call(callA1, "read_file", `{}`)))
	assert.Equal(t, request.ConversationLinkNew, first.Link)
	assert.Equal(t, []uint64{second.Key}, first.Adopted)

	third := annotated(t, tr, inference(40*ms, 50*ms, h3))
	assert.Equal(t, first.Key, third.Key)
}

func TestConversationCutHistoryLinksByItsCallIDs(t *testing.T) {
	tr := testTracker(testConfig())
	c := newChat("long task")
	full := c.answer(callA1).history()
	// both requests were cut by the capture, the second after more turns: its
	// captured part names the call the first response requested
	first := annotated(t, tr, inference(0, 10*ms, full[:len(full)-1], call(callA2, "run_command", `{}`)))
	more := c.answer(callA2).history()
	second := annotated(t, tr, inference(20*ms, 30*ms, more[:len(full)+80]))
	assert.Equal(t, first.Key, second.Key)
	assert.Equal(t, request.ConversationLinkCallID, second.Link)
}

// a cut history is no prefix evidence: conversations that start with the
// same document share every prefix the capture kept
func TestConversationCutHistoriesAreNotLinkedByPrefix(t *testing.T) {
	tr := testTracker(testConfig())
	doc := strings.Repeat("shared reference document ", 400)
	var keys []uint64
	for i, q := range []string{"question zero", "question one", "question two"} {
		h := `[{"role":"system","content":"s"},{"role":"user","content":"` + doc + q + `"}]`
		cut := h[:len(h)-len(q)-40]
		a := annotated(t, tr, inference(int64(i)*100*ms, int64(i)*100*ms+10*ms, cut, call(fmt.Sprintf("call_cutTwin%04dxyz", i), "read_doc", `{}`)))
		assert.Equal(t, request.ConversationLinkNew, a.Link, q)
		assert.NotContains(t, keys, a.Key, q)
		keys = append(keys, a.Key)
	}
}

// the part a cut request kept is not remembered as a whole history
func TestConversationCutRequestIsNotRemembered(t *testing.T) {
	tr := testTracker(testConfig())
	cut := `[{"role":"system","content":"s"},{"role":"user","content":"a long docu`
	a := annotated(t, tr, inference(0, 10*ms, cut))
	other := `[{"role":"system","content":"s"},{"role":"user","content":"another task"}]`
	b := annotated(t, tr, inference(20*ms, 30*ms, other))
	assert.NotEqual(t, a.Key, b.Key)
	assert.Equal(t, request.ConversationLinkNew, b.Link)
}

// servers that number calls by position issue the same ids in every
// conversation: they are no evidence, the histories tell
func TestConversationPositionalCallIDs(t *testing.T) {
	tr := testTracker(testConfig())
	a, b := newChat("user zero asks"), newChat("user one asks")
	const id = "functions.read_doc:0"

	a1 := annotated(t, tr, inference(0, 10*ms, a.history(), call(id, "read_doc", `{}`)))
	b1 := annotated(t, tr, inference(5*ms, 15*ms, b.history(), call(id, "read_doc", `{}`)))
	a2 := annotated(t, tr, inference(20*ms, 30*ms, a.answer(id).history()))
	b2 := annotated(t, tr, inference(25*ms, 35*ms, b.answer(id).history()))

	assert.NotEqual(t, a1.Key, b1.Key)
	assert.Equal(t, a1.Key, a2.Key)
	assert.Equal(t, request.ConversationLinkHistoryPrefix, a2.Link)
	assert.Equal(t, b1.Key, b2.Key)
	assert.Equal(t, request.ConversationLinkHistoryPrefix, b2.Link)
}

// an id that two conversations' responses issued points to neither
func TestConversationSharedCallIDIsNoEvidence(t *testing.T) {
	tr := testTracker(testConfig())
	a, b := newChat("hello"), newChat("hello")
	const id = "call_r3peatedId0001"

	a1 := annotated(t, tr, inference(0, 10*ms, a.history(), call(id, "read_file", `{}`)))
	b1 := annotated(t, tr, inference(5*ms, 15*ms, b.history(), call(id, "read_file", `{}`)))
	// the same history and the same id: nothing tells which conversation it is
	b2 := annotated(t, tr, inference(20*ms, 30*ms, b.answer(id).history()))
	assert.NotEqual(t, a1.Key, b1.Key)
	assert.Equal(t, request.ConversationLinkAmbiguous, b2.Link)
	assert.NotEqual(t, a1.Key, b2.Key)
	assert.NotEqual(t, b1.Key, b2.Key)
}

func TestConversationCallID(t *testing.T) {
	for id, want := range map[string]bool{
		"call_conversationA_0001":            true,
		"call_r3adFile0001":                  true,
		"toolu_01A09q90qw90lq917835lq9":      true,
		"chatcmpl-tool-9f8e7d6c5b4a39281706": true,
		"call_aaaaaaaaaaaa":                  true,
		"functions.read_doc:0":               false,
		"functions.read_doc:12":              false,
		"tool_call_000001":                   false,
		"call_read_doc_17":                   false,
		"call_0_0":                           false,
	} {
		assert.Equal(t, want, conversationCallID(id, "read_doc"), id)
	}
}

func TestConversationNestedSubAgentIsAnotherConversation(t *testing.T) {
	tr := testTracker(testConfig())
	outer, inner := newChat("outer task"), newChat("inner question")

	o1 := annotated(t, tr, inference(0, 10*ms, outer.history(), call(callA1, "ask_expert", `{}`)))
	i1 := annotated(t, tr, inference(15*ms, 20*ms, inner.history(), call(callB1, "read_file", `{}`)))
	i2 := annotated(t, tr, inference(25*ms, 30*ms, inner.answer(callB1).history()))
	o2 := annotated(t, tr, inference(35*ms, 40*ms, outer.answer(callA1).history()))

	assert.Equal(t, request.ConversationLinkNew, i1.Link)
	assert.NotEqual(t, o1.Key, i1.Key)
	assert.Equal(t, i1.Key, i2.Key)
	assert.Equal(t, o1.Key, o2.Key)
}

func TestConversationOffersTools(t *testing.T) {
	tr := testTracker(testConfig())
	s := inference(0, 10*ms, noTools)
	s.GenAI.OpenAI.Request.Tools = json.RawMessage(`[{"type":"function","function":{"name":"read_file"}}]`)
	assert.True(t, annotated(t, tr, s).OffersTools)

	s = inference(20*ms, 30*ms, `[{"role":"user","content":"other"}]`)
	s.GenAI.OpenAI.Request.Tools = json.RawMessage(`[]`)
	assert.False(t, annotated(t, tr, s).OffersTools)
}

func TestConversationNotAnnotatedForGateways(t *testing.T) {
	tr := testTracker(testConfig())
	server := request.Span{Type: request.EventTypeHTTP, Path: "/v1/chat/completions", Pid: request.PidInfo{HostPID: agentPID}}
	client := inference(0, 10*ms, noTools)
	out := tr.onSpans([]request.Span{server, client})
	assert.Nil(t, out[1].Conversation)
}

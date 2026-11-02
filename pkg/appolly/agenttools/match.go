// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package agenttools // import "go.opentelemetry.io/obi/pkg/appolly/agenttools"

import (
	"cmp"
	"hash/maphash"
	"log/slog"
	"net/netip"
	"slices"
	"strconv"
	"time"
	"unicode/utf8"

	"go.opentelemetry.io/obi/pkg/appolly/app/agentact"
	"go.opentelemetry.io/obi/pkg/appolly/app/request"
	"go.opentelemetry.io/obi/pkg/config"
	"go.opentelemetry.io/obi/pkg/ebpf/timing"
)

// tracker pairs the tool calls of GenAI responses with the requests that
// carry their results, and builds the inferred spans once a window closes.
//
// GenAI spans reach it late and reordered: a client record is flushed when
// the next request starts on the same connection, when the connection closes,
// or after 10s idle, and the request that answers a response may use another
// connection. Matching is therefore symmetric: a response checks the requests
// already seen, and a request closes the responses already seen.
type tracker struct {
	cfg   *config.GenAIToolCallsConfig
	store *store
	seed  maphash.Seed
	// roots arms the kernel activity of agents; nil when every instrumented
	// process reports it
	roots     *agentact.RootRegistry
	paths     agentact.PathMatcher
	services  serviceSelection
	retention time.Duration
	exitGrace time.Duration
	probation int64
	windowIDs uint64
	// conversations created so far, which their keys derive from
	conversationKeys uint64
	now              func() int64
}

func newTracker(cfg *config.GenAIToolCallsConfig, maxTransactionTime time.Duration, roots *agentact.RootRegistry) *tracker {
	t := &tracker{
		cfg:       cfg,
		store:     newStore(),
		seed:      maphash.MakeSeed(),
		roots:     roots,
		paths:     agentact.NewPathMatcher(cfg.KernelActivity.PathExclude),
		services:  newServiceSelection(&cfg.Services),
		retention: max(minActivityRetention, maxTransactionTime+exitGraceLag),
		exitGrace: max(minExitGrace, maxTransactionTime+exitGraceLag),
		probation: cfg.KernelActivity.Probation.Nanoseconds(),
		now:       func() int64 { return int64(timing.MonoTimeNow()) },
	}
	t.store.onEvict = t.disarm
	return t
}

// isInference tells whether a span is an LLM call that can carry tool calls
// or tool results.
func isInference(span *request.Span) bool {
	if span.Type != request.EventTypeHTTPClient || span.GenAI == nil || !request.IsGenAISubtype(span.SubType) {
		return false
	}
	switch span.SubType {
	case request.HTTPSubtypeEmbedding, request.HTTPSubtypeRerank, request.HTTPSubtypeRetrieval:
		return false
	}
	return true
}

func spanEndpoint(span *request.Span) netip.AddrPort {
	addr, err := netip.ParseAddr(span.Host)
	if err != nil {
		return netip.AddrPort{}
	}
	return netip.AddrPortFrom(addr.Unmap(), uint16(span.HostPort))
}

// onInference handles one GenAI span. As a request it may close calls of
// earlier responses; as a response it may open new calls.
func (t *tracker) onInference(span *request.Span, now int64) []request.Span {
	p := t.store.agent(span.Pid)
	p.pid = span.Pid
	if !t.selected(p, span) {
		return nil
	}
	if !p.kernelRootKnown {
		t.findKernelRoot(p)
	}
	p.maxLag = max(p.maxLag, time.Duration(now-span.End))
	p.addEndpoint(spanEndpoint(span))
	t.noteInference(p, span)

	req := requestResults{
		ref:        refOf(span),
		results:    span.GenAI.ToolResults(),
		history:    span.GenAI.RequestHistory(),
		prevRespID: span.GenAI.PreviousResponseID(),
	}
	alog().Debug("inference", "pid", span.Pid.HostPID, "start", span.Start, "end", span.End,
		"results", len(req.results), "calls", len(span.GenAI.ToolCalls()), "pending", len(p.pending))

	t.annotateConversation(p, span, &req, now)

	unparsed := len(span.GenAI.UnparsedRequest) > 0
	out := t.applyRequest(p, &req, p.pending, unparsed)
	p.rememberRequest(req)

	if w := t.openWindow(p, span, now); w != nil {
		out = append(out, t.matchLateWindow(p, w)...)
	}

	return out
}

func alog() *slog.Logger {
	return slog.With("component", "agenttools.ToolSpans")
}

// noteInference records an LLM call of the process. The first one after a
// window opened bounds the window if its call never gets a result.
func (t *tracker) noteInference(p *procState, span *request.Span) {
	p.addInference(span)
	for _, w := range p.pendingWindows() {
		if span.SpanID != w.source.spanID && span.Start >= w.readyAt && (w.nextInference == 0 || span.Start < w.nextInference) {
			w.nextInference = span.Start
		}
	}
}

// openWindow registers the client-side tool calls of a response.
func (t *tracker) openWindow(p *procState, span *request.Span, now int64) *toolWindow {
	calls := slices.DeleteFunc(slices.Clone(span.GenAI.ToolCalls()), func(tc request.ToolCall) bool {
		return tc.ServerSide || tc.Name == ""
	})
	if len(calls) == 0 {
		return nil
	}

	response := responseKey{id: span.GenAI.ResponseID(), start: span.Start, end: span.End}
	if p.seenResponses.Contains(response) {
		return nil
	}
	p.seenResponses.Add(response, struct{}{})
	if len(p.pending)+len(calls) > maxPendingPerProcess {
		alog().Debug("too many pending tool calls, ignoring a response", "pid", span.Pid.HostPID, "pending", len(p.pending))
		return nil
	}

	t.windowIDs++
	w := &toolWindow{
		id:         t.windowIDs,
		readyAt:    span.End,
		source:     refOf(span),
		service:    span.Service,
		pid:        span.Pid,
		responseID: response.id,
		conv:       conversationOf(t.seed, span.GenAI.RequestHistory()),
	}
	provider := span.GenAIProviderName()
	for _, tc := range calls {
		args, truncated := truncate(tc.Arguments)
		tc.Arguments = args
		w.calls = append(w.calls, &pendingCall{call: tc, provider: provider, responseID: response.id, window: w, truncated: truncated})
	}
	w.open = len(w.calls)
	for i := range p.recent {
		if r := &p.recent[i]; r.ref.spanID != w.source.spanID && r.ref.start >= w.readyAt &&
			(w.nextInference == 0 || r.ref.start < w.nextInference) {
			w.nextInference = r.ref.start
		}
	}

	t.noteSharedThread(p, w, span.GenAI.RequestHistory())
	p.addPending(w)
	if t.cfg.KernelActivity.Enabled && p.caPaths == nil {
		p.caPaths = caBundlePaths(uint32(p.pid.HostPID))
	}
	t.arm(p, now)
	return w
}

// noteSharedThread remembers a thread that started the windows of two
// conversations open at the same time, as an event loop does. A window of the
// same conversation still open is one whose result was lost.
func (t *tracker) noteSharedThread(p *procState, w *toolWindow, history []byte) {
	tid := w.source.tid
	if tid == 0 {
		return
	}
	for _, o := range p.pendingWindows() {
		if o.source.tid == tid && !o.conv.continuedBy(t.seed, history) {
			p.multiplexed[tid] = struct{}{}
			return
		}
	}
	for _, s := range p.settling {
		if s.window.source.tid == tid && s.window.closeAt() > w.readyAt {
			p.multiplexed[tid] = struct{}{}
			return
		}
	}
	for i := range p.past {
		if o := &p.past[i]; slices.Contains(o.llmThreads, tid) && o.to > w.readyAt {
			p.multiplexed[tid] = struct{}{}
			return
		}
	}
}

// matchLateWindow closes a new window against the requests seen before its
// response arrived. A process's spans often arrive after its exit, so an
// exited process closes its open calls only after a grace period, in tick.
func (t *tracker) matchLateWindow(p *procState, w *toolWindow) []request.Span {
	// the earliest request after the response is the agent's next model call
	recent := slices.Clone(p.recent)
	slices.SortStableFunc(recent, func(a, b requestResults) int { return cmp.Compare(a.ref.start, b.ref.start) })

	var out []request.Span
	for i := range recent {
		if w.open == 0 {
			break
		}
		if recent[i].ref.spanID == w.source.spanID {
			continue
		}
		out = append(out, t.applyRequest(p, &recent[i], w.calls, false)...)
	}

	return out
}

// applyRequest closes the candidate calls a request answers: by call id, then
// id-less results by name and order, then ids found by scanning a body that
// did not parse, and finally the next-request rule.
func (t *tracker) applyRequest(p *procState, req *requestResults, candidates []*pendingCall, unparsed bool) []request.Span {
	if len(candidates) == 0 {
		return nil
	}
	// candidates may be p.pending, which closing windows rewrites
	candidates = append([]*pendingCall(nil), candidates...)

	var touched []*toolWindow
	closeWith := func(c *pendingCall, outcome, endSource string, result request.ToolResult) {
		t.closeCall(c, req, outcome, endSource, result)
		touched = append(touched, c.window)
	}

	var idless []request.ToolResult
	for _, r := range req.results {
		if r.ID == "" {
			idless = append(idless, r)
			continue
		}
		if c := t.callForResult(candidates, req, r.ID); c != nil {
			closeWith(c, request.ToolOutcomeResult, request.ToolWindowEndToolResult, r)
		}
	}

	if t.cfg.MatchByName && len(idless) > 0 {
		for _, m := range matchByName(t.continuedWindow(candidates, req), idless) {
			closeWith(m.call, request.ToolOutcomeResultByName, request.ToolWindowEndToolResult, m.result)
		}
	}

	if unparsed && len(req.results) == 0 {
		for _, c := range candidates {
			if !c.closed && c.window.source.spanID != req.ref.spanID && distinctiveID(c.call.ID) && containsToken(req.history, c.call.ID) {
				closeWith(c, request.ToolOutcomeResult, request.ToolWindowEndToolResult, request.ToolResult{ID: c.call.ID})
			}
		}
	}

	if t.cfg.CloseOnNextRequest {
		for _, c := range candidates {
			if !c.closed && t.continuesCall(c, req) {
				closeWith(c, request.ToolOutcomeNextRequest, request.ToolWindowEndNextRequest, request.ToolResult{})
			}
		}
	}

	return t.flushWindows(p, touched)
}

// callForResult returns the open call a result id answers. Ids repeat across
// responses on some servers and across conversations, so a short id, or one
// several open calls carry, only pairs with a call of the conversation the
// request continues.
func (t *tracker) callForResult(candidates []*pendingCall, req *requestResults, id string) *pendingCall {
	var matches []*pendingCall
	for _, c := range candidates {
		if !c.closed && c.call.ID == id {
			matches = append(matches, c)
		}
	}
	if len(matches) == 1 && distinctiveID(id) {
		return matches[0]
	}

	var continued *pendingCall
	for _, c := range matches {
		if !t.continues(c.window, req) {
			continue
		}
		if continued != nil {
			return nil
		}
		continued = c
	}
	return continued
}

// continues tells whether req belongs to the conversation of a window: it
// continues the stateful response, or resends the history that asked for
// the calls with turns appended.
func (t *tracker) continues(w *toolWindow, req *requestResults) bool {
	if req.ref.spanID == w.source.spanID {
		return false
	}
	if req.prevRespID != "" && req.prevRespID == w.responseID {
		return true
	}
	return w.conv.continuedBy(t.seed, req.history)
}

// continuesCall tells whether a later request of the call's conversation
// went out without a parsable result for it: the capture cut it, or the
// agent encodes results as text. Anthropic and OpenAI require every result
// before the next model call of a conversation, so the call was done by then.
// Requests of other conversations, such as a tool's own model calls, never
// close a call.
func (t *tracker) continuesCall(c *pendingCall, req *requestResults) bool {
	if req.ref.start < c.window.readyAt {
		return false
	}
	if t.continues(c.window, req) {
		return true
	}
	// the request replays the turn that asked for the call
	return req.ref.spanID != c.window.source.spanID && distinctiveID(c.call.ID) && containsToken(req.history, c.call.ID)
}

// continuedWindow returns the newest window with open calls, ready before
// the request, whose conversation the request continues.
func (t *tracker) continuedWindow(candidates []*pendingCall, req *requestResults) *toolWindow {
	var latest *toolWindow
	for _, c := range candidates {
		w := c.window
		if c.closed || w.readyAt > req.ref.start || (latest != nil && w.readyAt <= latest.readyAt) {
			continue
		}
		if t.continues(w, req) {
			latest = w
		}
	}
	return latest
}

type nameMatch struct {
	call   *pendingCall
	result request.ToolResult
}

// matchByName pairs id-less results with the open calls of one window: by
// tool name in order of appearance, or by position when the result names no
// tool. Same-name parallel calls pair in order.
func matchByName(w *toolWindow, results []request.ToolResult) []nameMatch {
	if w == nil {
		return nil
	}

	used := map[*pendingCall]bool{}
	var matches []nameMatch
	for _, r := range results {
		c := findOpen(w.calls, func(c *pendingCall) bool {
			return !used[c] && (r.Name == "" || r.Name == c.call.Name)
		})
		if c == nil {
			continue
		}
		used[c] = true
		matches = append(matches, nameMatch{call: c, result: r})
	}
	return matches
}

func findOpen(calls []*pendingCall, match func(*pendingCall) bool) *pendingCall {
	for _, c := range calls {
		if !c.closed && match(c) {
			return c
		}
	}
	return nil
}

func (t *tracker) closeCall(c *pendingCall, req *requestResults, outcome, endSource string, result request.ToolResult) {
	alog().Debug("tool call closed", "pid", c.window.pid.HostPID, "tool", c.call.Name, "outcome", outcome)
	sink := req.ref
	c.closed = true
	// a request reusing a connection opened earlier starts before the response
	// ended; the window cannot close before it opened
	c.closeAt = max(sink.start, c.window.readyAt)
	c.outcome = outcome
	c.endSource = endSource
	c.result = result
	raw, truncated := truncate(string(result.Raw))
	c.result.Raw = []byte(raw)
	c.truncated = c.truncated || truncated
	c.sink = &sink
	c.window.open--
}

// closeUnmatched closes calls that never got a result, at the given time or
// at the next LLM call of the process, whichever comes first. Nothing proves
// the call ran until then: the outcome says so.
func (t *tracker) closeUnmatched(p *procState, calls []*pendingCall, at int64, outcome, endSource string) []request.Span {
	var touched []*toolWindow
	for _, c := range calls {
		if c.closed {
			continue
		}
		alog().Debug("tool call closed", "pid", c.window.pid.HostPID, "tool", c.call.Name, "outcome", outcome)
		end, source := at, endSource
		if next := c.window.nextInference; next != 0 && next < end {
			end, source = next, request.ToolWindowEndNextInference
		}
		c.closed = true
		c.closeAt = max(end, c.window.readyAt)
		c.outcome = outcome
		c.endSource = source
		c.window.open--
		touched = append(touched, c.window)
	}
	return t.flushWindows(p, touched)
}

// flushWindows emits the windows whose calls are all closed. With kernel
// activity, they first wait for the events of their window still in flight.
func (t *tracker) flushWindows(p *procState, windows []*toolWindow) []request.Span {
	var out []request.Span
	seen := map[*toolWindow]bool{}
	for _, w := range windows {
		if seen[w] || w.open > 0 {
			continue
		}
		seen[w] = true
		p.removeWindow(w)
		if t.cfg.KernelActivity.Enabled {
			p.settling = append(p.settling, settlingWindow{window: w, deadline: time.Now().Add(activitySettle)})
			continue
		}
		spans, _ := t.windowSpans(p, w, true)
		out = append(out, spans...)
	}
	return out
}

// truncate cuts a payload at maxPayloadBytes, on a rune boundary.
func truncate(s string) (string, bool) {
	if len(s) <= maxPayloadBytes {
		return s, false
	}
	cut := maxPayloadBytes
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return s[:cut], true
}

func pidString(pid uint32) string {
	return strconv.FormatUint(uint64(pid), 10)
}

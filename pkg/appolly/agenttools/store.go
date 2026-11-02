// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package agenttools // import "go.opentelemetry.io/obi/pkg/appolly/agenttools"

import (
	"math"
	"net/netip"
	"slices"
	"time"

	lru "github.com/hashicorp/golang-lru/v2"
	"go.opentelemetry.io/otel/trace"

	"go.opentelemetry.io/obi/pkg/appolly/app"
	"go.opentelemetry.io/obi/pkg/appolly/app/agentact"
	"go.opentelemetry.io/obi/pkg/appolly/app/request"
	"go.opentelemetry.io/obi/pkg/appolly/app/svc"
)

const (
	// bound memory, not correctness: an evicted process loses its open calls
	maxProcesses = 1024
	// instrumented processes that made no LLM call yet, kept for the kernel
	// activity of their probation period
	maxCandidates = 256
	// responses already seen, so a response reaching the stage twice cannot
	// open its calls again
	seenResponsesPerProcess = 1024
	// requests kept for responses that reach the stage after the request that
	// answers them: GenAI spans flush late and out of order. A server running
	// many conversations sends dozens per second.
	recentRequestsPerProcess = 256
	// conversation bytes kept for those requests, to recognize continuations
	maxRecentHistoryBytes = 1 << 20
	// kernel activity kept per process while no window is known to need it,
	// at least: spans held for their parents arrive up to the maximum
	// transaction time late
	minActivityRetention  = 30 * time.Second
	maxActivityPerProcess = 4096
	// processes that exited, for responses that arrive after the exit. Their
	// pids are reused, so they are forgotten after a while.
	exitedProcesses = 1024
	exitedTTL       = 10 * time.Minute
	// a process exit is acted on only after its last spans had time to
	// arrive: they flush when its connections close, or after 10s idle
	minExitGrace = 15 * time.Second
	exitGraceLag = 10 * time.Second
	// raw payloads kept per call
	maxPayloadBytes = 8 << 10
	// lineage entries kept per instrumented process
	maxLineagePerProcess = 4096
	// calls waiting for a result per process; later ones are not tracked
	maxPendingPerProcess = 1024
	// LLM endpoints remembered per process, to exclude its own LLM connects
	maxEndpointsPerProcess = 256
	// emitted windows kept to tell later windows they overlapped
	maxPastWindows = 512
	// LLM calls kept per process, to tell when a thread served another one
	maxInferencesPerProcess = 4096
	// spans of time with lost activity kept per process
	maxLossesPerProcess = 256
)

// spanRef identifies the inference span a tool call came from or went to.
type spanRef struct {
	traceID      trace.TraceID
	spanID       trace.SpanID
	parentSpanID trace.SpanID
	traceFlags   uint8
	start, end   int64
	tid          uint32
}

func refOf(span *request.Span) spanRef {
	return spanRef{
		traceID:      span.TraceID,
		spanID:       span.SpanID,
		parentSpanID: span.ParentSpanID,
		traceFlags:   span.TraceFlags,
		start:        span.Start,
		end:          span.End,
		tid:          span.TID,
	}
}

// responseKey identifies a response. Some servers reuse response ids, so the
// response timing is part of it; a span reaching the stage twice keeps both.
type responseKey struct {
	id         string
	start, end int64
}

// pendingCall is a tool call requested by a response and not yet emitted.
type pendingCall struct {
	call       request.ToolCall
	provider   string
	responseID string
	window     *toolWindow
	// the arguments or the result were cut at maxPayloadBytes
	truncated bool
	matcher   *argMatcher

	// filled when the call closes
	closed    bool
	closeAt   int64
	outcome   string
	endSource string
	result    request.ToolResult
	sink      *spanRef
}

func (c *pendingCall) argMatcher() argMatcher {
	if c.matcher == nil {
		m := newArgMatcher(c.call.Arguments)
		c.matcher = &m
	}
	return *c.matcher
}

func (c *pendingCall) unmatched() bool {
	return c.outcome == request.ToolOutcomeUnmatchedExit || c.outcome == request.ToolOutcomeUnmatchedTimeout
}

// toolWindow holds the calls of one response. They are available to the
// agent at the same time and, since all results of a response come back in
// one request, they usually close together.
type toolWindow struct {
	id         uint64
	calls      []*pendingCall
	readyAt    int64
	source     spanRef
	service    svc.Attrs
	pid        request.PidInfo
	open       int
	responseID string
	// the conversation of the request that asked for the calls
	conv conversation
	// start of the first other LLM call of the process after readyAt: the
	// end of a window that never gets a result
	nextInference int64
	// the window waited once for evidence that its loop thread served
	// other conversations
	deferred bool
}

func (w *toolWindow) closeAt() int64 {
	end := w.readyAt
	for _, c := range w.calls {
		end = max(end, c.closeAt)
	}
	return end
}

// end is the end of the window, unbounded while a call is open.
func (w *toolWindow) end() int64 {
	if w.open > 0 {
		return math.MaxInt64
	}
	return w.closeAt()
}

// unmatched tells that no call of the window got a result: its end is a
// bound OBI chose, not one the agent proved.
func (w *toolWindow) unmatched() bool {
	for _, c := range w.calls {
		if !c.unmatched() {
			return false
		}
	}
	return true
}

// loopThread is the thread of a sequential agent loop: a single call whose
// request and result went out from the same thread, which ran the tool in
// between. 0 when there is none.
func (w *toolWindow) loopThread() uint32 {
	if len(w.calls) != 1 || w.source.tid == 0 {
		return 0
	}
	if sink := w.calls[0].sink; sink == nil || sink.tid != w.source.tid {
		return 0
	}
	return w.source.tid
}

// llmThreads are the threads that performed the window's LLM calls.
func (w *toolWindow) llmThreads() []uint32 {
	tids := []uint32{w.source.tid}
	for _, c := range w.calls {
		if c.sink != nil {
			tids = append(tids, c.sink.tid)
		}
	}
	return tids
}

// requestResults is a GenAI request kept for responses that arrive later.
type requestResults struct {
	ref        spanRef
	results    []request.ToolResult
	history    []byte
	prevRespID string
}

type settlingWindow struct {
	window   *toolWindow
	deadline time.Time
}

// pastWindow is an emitted window, kept so that the windows emitted later
// know it overlapped them and which arguments its calls had.
type pastWindow struct {
	id         uint64
	from, to   int64
	loopThread uint32
	llmThreads []uint32
	calls      []pastCall
}

type pastCall struct {
	matcher argMatcher
	closeAt int64
}

// llmCall is an LLM call of the process.
type llmCall struct {
	spanID     trace.SpanID
	start, end int64
	tid        uint32
}

// activityRecord is a kernel event and the window that claimed it, if any.
type activityRecord struct {
	agentact.Event
	owner uint64
}

// loss is a span of time whose activity is incomplete: count events were
// dropped between from and to.
type loss struct {
	from, to int64
	count    int
}

// procNode is a descendant of an instrumented process, as reported by the
// kernel fork, exec and exit events.
type procNode struct {
	pid        uint32
	parent     uint32
	spawnerTID uint32
	forkAt     int64
	execAt     int64
	path       string
	args       []string
	exited     bool
	exitAt     int64
	exitCode   int
	// the window that claimed the process
	owner uint64
}

// procState is everything known about one instrumented process.
type procState struct {
	pid request.PidInfo
	// the process serves LLM API routes: a gateway or proxy, whose calls are
	// others' tool calls
	llmServer bool
	// the services selection leaves the process out
	excluded       bool
	selectionKnown bool
	// the kernel reports the process's activity since armedAt; armedPIDs are
	// the roots armed for it
	armed     bool
	armedAt   int64
	armedPIDs []uint32
	// the instrumented ancestor whose tree the kernel reports the process's
	// activity in, when it is not its own root: an agent started by another
	// instrumented process
	kernelRoot      uint32
	kernelRootKnown bool
	// CLOCK_MONOTONIC start of the process, from its kernel events
	rootStart int64
	// the longest delay seen between the end of one of its spans and its arrival
	maxLag time.Duration

	pending []*pendingCall
	// closed windows waiting for activity still in flight
	settling      []settlingWindow
	past          []pastWindow
	recent        []requestResults
	recentBytes   int
	seenResponses *lru.Cache[responseKey, struct{}]
	inferences    []llmCall
	// threads that served several conversations at once: being on one of
	// them does not tie an event to a call
	multiplexed map[uint32]struct{}
	// the conversations of the process's LLM calls, created at the first one
	conversations *conversations

	// server endpoints of the process's GenAI spans
	endpoints map[netip.AddrPort]struct{}
	// CA bundles the process's LLM clients reopen, from its environment
	caPaths []string

	activity []activityRecord
	// the newest event dropped because it was older than the retention
	trimmedUntil int64
	losses       []loss
	lineage      map[uint32]*procNode

	exited     bool
	exitAt     int64
	exitSeenAt time.Time
}

func newProcState(pid request.PidInfo) *procState {
	seen, _ := lru.New[responseKey, struct{}](seenResponsesPerProcess)
	return &procState{
		pid:           pid,
		seenResponses: seen,
		multiplexed:   map[uint32]struct{}{},
		endpoints:     map[netip.AddrPort]struct{}{},
		lineage:       map[uint32]*procNode{},
	}
}

func (p *procState) markExited(at int64) {
	p.exited = true
	p.exitAt = at
	p.exitSeenAt = time.Now()
}

type exitRecord struct {
	at     int64
	seenAt time.Time
}

// store indexes process state by host pid. Processes that made LLM calls and
// candidates that so far only report kernel activity are kept apart, so that
// the activity of many instrumented services cannot evict an agent.
type store struct {
	procs      *lru.Cache[app.PID, *procState]
	candidates *lru.Cache[app.PID, *procState]
	exited     *lru.Cache[app.PID, exitRecord]
	onEvict    func(*procState)
}

func newStore() *store {
	s := &store{}
	evicted := func(_ app.PID, p *procState) {
		if s.onEvict != nil {
			s.onEvict(p)
		}
	}
	s.procs, _ = lru.NewWithEvict[app.PID, *procState](maxProcesses, evicted)
	s.candidates, _ = lru.NewWithEvict[app.PID, *procState](maxCandidates, evicted)
	s.exited, _ = lru.New[app.PID, exitRecord](exitedProcesses)
	return s
}

// agent returns the state of a process that made an LLM call, promoting the
// candidate state its kernel activity created.
func (s *store) agent(pid request.PidInfo) *procState {
	if p, ok := s.procs.Get(pid.HostPID); ok {
		return p
	}

	p, ok := s.candidates.Peek(pid.HostPID)
	if ok {
		s.candidates.Remove(pid.HostPID)
		p.pid = pid
	} else {
		p = s.fresh(pid)
	}
	s.procs.Add(pid.HostPID, p)
	return p
}

// activity returns the state that kernel activity of a root goes to. A root
// that made no LLM call gets a candidate state only while create holds.
func (s *store) activity(root app.PID, create bool) (*procState, bool) {
	if p, ok := s.procs.Get(root); ok {
		return p, true
	}
	if p, ok := s.candidates.Get(root); ok {
		return p, true
	}
	if !create {
		return nil, false
	}

	p := s.fresh(request.PidInfo{HostPID: root})
	s.candidates.Add(root, p)
	return p, true
}

func (s *store) peek(pid app.PID) (*procState, bool) {
	if p, ok := s.procs.Peek(pid); ok {
		return p, true
	}
	return s.candidates.Peek(pid)
}

func (s *store) fresh(pid request.PidInfo) *procState {
	p := newProcState(pid)
	if rec, ok := s.exited.Get(pid.HostPID); ok && time.Since(rec.seenAt) < exitedTTL {
		// a span arriving after the process state was dropped gets another
		// grace period for the spans that may follow it
		p.markExited(rec.at)
	}
	return p
}

// forget drops what is known about a pid, which a new process now uses.
func (s *store) forget(pid app.PID) {
	s.exited.Remove(pid)
	if p, ok := s.peek(pid); ok && p.exited {
		s.remove(p)
	}
}

func (s *store) remove(p *procState) {
	s.procs.Remove(p.pid.HostPID)
	s.candidates.Remove(p.pid.HostPID)
	if p.exited {
		s.exited.Add(p.pid.HostPID, exitRecord{at: p.exitAt, seenAt: p.exitSeenAt})
	}
}

func (s *store) all() []*procState {
	var out []*procState
	for _, pid := range s.procs.Keys() {
		if p, ok := s.procs.Peek(pid); ok {
			out = append(out, p)
		}
	}
	for _, pid := range s.candidates.Keys() {
		if p, ok := s.candidates.Peek(pid); ok {
			out = append(out, p)
		}
	}
	return out
}

// addPending registers the open calls of a window.
func (p *procState) addPending(w *toolWindow) {
	p.pending = append(p.pending, w.calls...)
}

// removeWindow forgets the calls of a closed window.
func (p *procState) removeWindow(w *toolWindow) {
	kept := p.pending[:0]
	for _, c := range p.pending {
		if c.window != w {
			kept = append(kept, c)
		}
	}
	clear(p.pending[len(kept):])
	p.pending = kept
}

func (p *procState) delaySettling(w *toolWindow, deadline time.Time) {
	for i := range p.settling {
		if p.settling[i].window == w {
			p.settling[i].deadline = deadline
		}
	}
}

func (p *procState) removeSettling(w *toolWindow) {
	p.settling = slices.DeleteFunc(p.settling, func(s settlingWindow) bool { return s.window == w })
}

// pendingWindows returns the windows with calls still open.
func (p *procState) pendingWindows() []*toolWindow {
	var windows []*toolWindow
	for _, c := range p.pending {
		if len(windows) == 0 || windows[len(windows)-1] != c.window {
			windows = append(windows, c.window)
		}
	}
	return windows
}

func (p *procState) rememberRequest(req requestResults) {
	if len(p.recent) == recentRequestsPerProcess {
		p.dropOldestRequest()
	}
	p.recent = append(p.recent, req)
	p.recentBytes += len(req.history)
	// older requests keep their results but lose their conversation bytes
	for i := 0; p.recentBytes > maxRecentHistoryBytes && i < len(p.recent)-1; i++ {
		p.recentBytes -= len(p.recent[i].history)
		p.recent[i].history = nil
	}
}

func (p *procState) dropOldestRequest() {
	p.recentBytes -= len(p.recent[0].history)
	copy(p.recent, p.recent[1:])
	p.recent[len(p.recent)-1] = requestResults{}
	p.recent = p.recent[:len(p.recent)-1]
}

func (p *procState) addInference(span *request.Span) {
	if len(p.inferences) == maxInferencesPerProcess {
		p.inferences = append(p.inferences[:0], p.inferences[1:]...)
	}
	p.inferences = append(p.inferences, llmCall{spanID: span.SpanID, start: span.Start, end: span.End, tid: span.TID})
}

func (p *procState) addEndpoint(endpoint netip.AddrPort) {
	if !endpoint.IsValid() || len(p.endpoints) >= maxEndpointsPerProcess {
		return
	}
	p.endpoints[endpoint] = struct{}{}
}

func (p *procState) addActivity(e *agentact.Event) {
	if len(p.activity) == maxActivityPerProcess {
		dropped := p.activity[:maxActivityPerProcess/4]
		p.addLoss(loss{from: dropped[0].Time, to: dropped[len(dropped)-1].Time, count: len(dropped)})
		p.activity = append(p.activity[:0], p.activity[len(dropped):]...)
	}
	p.activity = append(p.activity, activityRecord{Event: *e})
}

func (p *procState) addLoss(l loss) {
	if len(p.losses) == maxLossesPerProcess {
		// the two oldest merge: their span of time stays incomplete
		p.losses[1] = loss{from: p.losses[0].from, to: p.losses[1].to, count: p.losses[0].count + p.losses[1].count}
		p.losses = append(p.losses[:0], p.losses[1:]...)
	}
	p.losses = append(p.losses, l)
}

// lost reports how many events were lost inside [from, to], and whether any
// loss touches it. Losses that only partly overlap the window make it
// incomplete without adding to its count: their events may be outside.
func (p *procState) lost(from, to int64) (int, bool) {
	count, any := 0, from <= p.trimmedUntil
	for _, l := range p.losses {
		if l.to < from || l.from > to {
			continue
		}
		any = true
		if l.from >= from && l.to <= to {
			count += l.count
		}
	}
	return count, any
}

// trim drops what no open or future window can need: activity, windows and
// LLM calls older than the retention and than every pending window.
func (p *procState) trim(now int64, retention time.Duration) {
	keepFrom := now - retention.Nanoseconds()
	for _, c := range p.pending {
		keepFrom = min(keepFrom, c.window.readyAt)
	}
	for _, s := range p.settling {
		keepFrom = min(keepFrom, s.window.readyAt)
	}

	drop := 0
	for drop < len(p.activity) && p.activity[drop].Time < keepFrom {
		drop++
	}
	if drop > 0 {
		p.trimmedUntil = max(p.trimmedUntil, p.activity[drop-1].Time)
		p.activity = append(p.activity[:0], p.activity[drop:]...)
	}

	p.past = trimBefore(p.past, keepFrom, func(w *pastWindow) int64 { return w.to })
	p.inferences = trimBefore(p.inferences, keepFrom, func(i *llmCall) int64 { return i.end })
	p.losses = trimBefore(p.losses, keepFrom, func(l *loss) int64 { return l.to })
	for pid, n := range p.lineage {
		if n.exited && n.exitAt < keepFrom {
			delete(p.lineage, pid)
		}
	}
}

func trimBefore[T any](items []T, keepFrom int64, end func(*T) int64) []T {
	kept := items[:0]
	for i := range items {
		if end(&items[i]) >= keepFrom {
			kept = append(kept, items[i])
		}
	}
	clear(items[len(kept):])
	return kept
}

// eventsBetween returns the activity records of [from, to].
func (p *procState) eventsBetween(from, to int64) []*activityRecord {
	var events []*activityRecord
	for i := range p.activity {
		if t := p.activity[i].Time; t >= from && t <= to {
			events = append(events, &p.activity[i])
		}
	}
	return events
}

// trackLineage maintains the process tree below the instrumented process
// for its whole life, not only inside windows: pool workers are created
// before the first window and reused by every later one.
func (p *procState) trackLineage(e *agentact.Event) {
	switch e.Kind {
	case agentact.KindFork:
		if len(p.lineage) >= maxLineagePerProcess {
			return
		}
		p.lineage[e.PID] = &procNode{pid: e.PID, parent: e.ParentPID, spawnerTID: e.SpawnerTID, forkAt: e.Time}
	case agentact.KindExec:
		n := p.lineage[e.PID]
		if n == nil {
			if len(p.lineage) >= maxLineagePerProcess {
				return
			}
			n = &procNode{pid: e.PID, parent: e.ParentPID}
			p.lineage[e.PID] = n
		}
		n.execAt = e.Time
		n.path = e.Path
		n.args = e.Args
	case agentact.KindExit:
		if n := p.lineage[e.PID]; n != nil {
			n.exited = true
			n.exitAt = e.Time
			n.exitCode = e.ExitCode
		}
	}
}

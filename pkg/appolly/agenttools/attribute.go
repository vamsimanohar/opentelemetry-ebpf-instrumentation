// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package agenttools // import "go.opentelemetry.io/obi/pkg/appolly/agenttools"

import (
	"cmp"
	"encoding/json"
	"fmt"
	"math"
	"path"
	"slices"
	"strings"
	"time"

	"go.opentelemetry.io/otel/trace"

	"go.opentelemetry.io/obi/pkg/appolly/app/agentact"
	"go.opentelemetry.io/obi/pkg/appolly/app/request"
)

const (
	minExactToken     = 2
	minSubstringMatch = 6
	minSuffixSegments = 2

	scoreSubstring = 1
	scoreSuffix    = 2
	scoreExact     = 3
)

const unassigned = -1

// recurringWork is how long the activity of a worker outside tool windows
// must last to make it a background worker: pools and runtimes start their
// workers and executor threads in a short burst.
const recurringWork = 200 * time.Millisecond

// attributed is a kernel event and the rule that tied it to its span.
type attributed struct {
	rec  *activityRecord
	rule string
}

// callActivity is the kernel activity attributed to one span: an
// execute_tool span, or the window span for what no call can claim.
type callActivity struct {
	// opens and connects of the instrumented process, and of processes that
	// existed before the window
	events []attributed
	// processes started in the window, children of the instrumented process
	// or of a process that existed before it
	children []*procNode
	// the weakest rule of the claims
	rule  string
	first int64
	last  int64
}

func (a *callActivity) claim(rule string, at int64) {
	if a.rule == "" || rulePriority(rule) > rulePriority(a.rule) {
		a.rule = rule
	}
	if a.first == 0 || at < a.first {
		a.first = at
	}
	a.last = max(a.last, at)
}

func (a *callActivity) empty() bool {
	return len(a.events) == 0 && len(a.children) == 0
}

// rulePriority orders the rules from the strongest evidence to the weakest.
func rulePriority(rule string) int {
	switch rule {
	case request.ToolAttributionArgumentMatch:
		return 0
	case request.ToolAttributionLineage:
		return 1
	case request.ToolAttributionThread:
		return 2
	case request.ToolAttributionExclusive:
		return 3
	}
	return 4
}

// windowActivity splits the activity of a window among its calls.
type windowActivity struct {
	calls  []callActivity
	shared callActivity
	// processes started in the window, by pid, with their own events and the
	// rule that tied them to their span
	procs    map[uint32]*procNode
	events   map[uint32][]*activityRecord
	procRule map[uint32]string
	// activity no rule could tell apart from that of another open window
	overlap  int
	dropped  int
	complete bool
	// some activity went to the call only because of the thread it ran on,
	// which a conversation whose LLM calls are still in flight may share
	usedThreadRule bool
}

// windowView is a window the attribution of another one must account for:
// itself, or a window of the same process open at the same time.
type windowView struct {
	id         uint64
	from, to   int64
	loopThread uint32
	llmThreads []uint32
	calls      []int
	unmatched  bool
}

func (v *windowView) covers(at int64) bool {
	return at >= v.from && at <= v.to
}

// candidateCall is a call of a window view.
type candidateCall struct {
	view    int
	index   int
	matcher argMatcher
	closeAt int64
}

// attribution decides the activity of one window against all the windows of
// the process that overlap it, so that a kernel event or a process can only
// ever be claimed by one span.
type attribution struct {
	t *tracker
	p *procState
	// src holds the kernel activity of p: p itself, or the instrumented
	// ancestor p is reported under, whose tree only p's subtree is of p
	src   *procState
	w     *toolWindow
	root  uint32
	views []windowView
	calls []candidateCall

	threadSeeds  seeds
	processSeeds seeds
	llmThreads   map[uint32]struct{}
	// the loop thread of w served nothing else while w was open
	loopThreadFree bool
	// threads, and pool workers, with activity outside every tool window
	busy map[worker]struct{}
}

// worker is a thread of the instrumented process, or another process of its
// tree as a whole.
type worker struct {
	pid, tid uint32
}

// decision is where a piece of activity goes: a call, the window span of one
// window, or nowhere because windows overlap.
type decision struct {
	call    int
	view    int
	rule    string
	overlap bool
	// decided by the thread the activity ran on, not by what it touched
	byThread bool
}

func undecided() decision {
	return decision{call: unassigned, view: unassigned}
}

func (d decision) decided() bool {
	return d.call != unassigned
}

// attribute assigns the activity of a window to its calls. Evidence decides,
// in order: arguments that name the resource, descent from a process already
// tied to a call, the thread or the pool worker whose argument-matched work
// went to one call, the loop thread of a sequential agent that ran the call's
// tool, and, for the only window open, a thread or pool worker that does
// nothing outside tool windows. Activity of the window that none of them
// decides goes to the window span; activity that another window open at the
// same time could equally own goes nowhere and is counted.
func (t *tracker) attribute(p *procState, w *toolWindow) *windowActivity {
	at := t.newAttribution(p, w)
	act := &windowActivity{
		calls:    make([]callActivity, len(w.calls)),
		procs:    map[uint32]*procNode{},
		events:   map[uint32][]*activityRecord{},
		procRule: map[uint32]string{},
	}

	from, to := w.readyAt, w.closeAt()
	nodes := at.windowNodes(from, to)
	events, nodeEvents := at.windowEvents(from, to, nodes)

	nodeDecisions := make([]decision, len(nodes.topmost))
	for i, n := range nodes.topmost {
		nodeDecisions[i] = at.byArguments(execCandidates(n), n.startAt())
		if nodeDecisions[i].decided() && n.parent == at.root {
			at.threadSeeds.add(n.spawnerTID, nodeDecisions[i].call, at.llmThreads)
		}
	}
	eventDecisions := make([]decision, len(events))
	for i, e := range events {
		eventDecisions[i] = at.byArguments(eventCandidates(&e.Event), e.Time)
		if !eventDecisions[i].decided() {
			continue
		}
		if e.PID == at.root {
			at.threadSeeds.add(e.TID, eventDecisions[i].call, at.llmThreads)
		} else {
			at.processSeeds.add(e.PID, eventDecisions[i].call, nil)
		}
	}

	for i, n := range nodes.topmost {
		d := nodeDecisions[i]
		if !d.decided() {
			d = at.decideNode(n)
		}
		at.assignNode(act, n, nodes.subtree(n), nodeEvents, d)
	}
	for i, e := range events {
		d := eventDecisions[i]
		if !d.decided() {
			d = at.decideEvent(e)
		}
		at.assignEvent(act, e, d)
	}

	act.dropped, act.complete = t.completeness(at.src, from, to)
	return act
}

func (t *tracker) newAttribution(p *procState, w *toolWindow) *attribution {
	at := &attribution{
		t:            t,
		p:            p,
		src:          t.activitySource(p),
		w:            w,
		root:         uint32(p.pid.HostPID),
		threadSeeds:  seeds{owners: map[uint32]int{}},
		processSeeds: seeds{owners: map[uint32]int{}},
		llmThreads:   map[uint32]struct{}{},
	}
	at.addView(w.id, w.readyAt, w.closeAt(), w.loopThread(), w.llmThreads(), w.unmatched(), callsOf(w))

	from, to := w.readyAt, w.closeAt()
	for _, o := range p.pendingWindows() {
		if o != w && o.readyAt < to {
			at.addView(o.id, o.readyAt, math.MaxInt64, 0, o.llmThreads(), false, callsOf(o))
		}
	}
	for _, s := range p.settling {
		if o := s.window; o != w && o.readyAt < to && o.closeAt() > from {
			at.addView(o.id, o.readyAt, o.closeAt(), o.loopThread(), o.llmThreads(), o.unmatched(), callsOf(o))
		}
	}
	for i := range p.past {
		if o := &p.past[i]; o.from < to && o.to > from {
			at.addView(o.id, o.from, o.to, o.loopThread, o.llmThreads, false, o.calls)
		}
	}

	for _, v := range at.views {
		for _, tid := range v.llmThreads {
			at.llmThreads[tid] = struct{}{}
		}
	}
	at.loopThreadFree = t.loopThreadFree(p, w, at.views)
	at.busy = at.busyWorkers()
	return at
}

// busyWorkers returns the threads and processes that work outside the tool
// windows of the process: since w opened, or, for threads of the
// instrumented process, again and again before. They are background threads,
// such as a telemetry exporter, whose work inside a window is no evidence of
// the tool. A single burst before is not held against a thread, nor the start
// of a pool worker process: pools start their workers, and runtimes use
// their executor threads, before the first tool call.
func (at *attribution) busyWorkers() map[worker]struct{} {
	// windows still open count as not covering anything: what a worker does
	// while they are open may not be their work, and a worker found busy
	// only loses this rule
	type span struct{ from, to int64 }
	var windows []span
	for _, s := range at.p.settling {
		windows = append(windows, span{s.window.readyAt, s.window.closeAt()})
	}
	for i := range at.p.past {
		windows = append(windows, span{at.p.past[i].from, at.p.past[i].to})
	}
	windows = append(windows, span{at.w.readyAt, at.w.closeAt()})

	type outside struct{ first, last int64 }
	before := map[worker]outside{}
	busy := map[worker]struct{}{}
	for i := range at.src.activity {
		e := &at.src.activity[i]
		if !at.inScope(e.PID) || !at.t.toolWork(at.p, &e.Event) {
			continue
		}
		inWindow := false
		for _, s := range windows {
			if e.Time >= s.from && e.Time <= s.to {
				inWindow = true
				break
			}
		}
		if inWindow {
			continue
		}

		wk := at.workerOf(e.PID, e.TID)
		if e.Time >= at.w.readyAt {
			busy[wk] = struct{}{}
			continue
		}
		o, seen := before[wk]
		if !seen {
			o.first = e.Time
		}
		o.last = e.Time
		before[wk] = o
	}
	for wk, o := range before {
		if wk.pid == at.root && o.last-o.first >= recurringWork.Nanoseconds() {
			busy[wk] = struct{}{}
		}
	}
	return busy
}

func (at *attribution) workerOf(pid, tid uint32) worker {
	if pid == at.root {
		return worker{pid: pid, tid: tid}
	}
	return worker{pid: pid}
}

func callsOf(w *toolWindow) []pastCall {
	calls := make([]pastCall, len(w.calls))
	for i, c := range w.calls {
		closeAt := int64(math.MaxInt64)
		if c.closed {
			closeAt = c.closeAt
		}
		calls[i] = pastCall{matcher: c.argMatcher(), closeAt: closeAt}
	}
	return calls
}

func (at *attribution) addView(id uint64, from, to int64, loop uint32, llm []uint32, unmatched bool, calls []pastCall) {
	v := windowView{id: id, from: from, to: to, loopThread: loop, llmThreads: llm, unmatched: unmatched}
	for i, c := range calls {
		v.calls = append(v.calls, len(at.calls))
		at.calls = append(at.calls, candidateCall{view: len(at.views), index: i, matcher: c.matcher, closeAt: c.closeAt})
	}
	at.views = append(at.views, v)
}

// loopThreadFree tells whether the loop thread of w did nothing but run its
// tool while w was open. A thread that served another LLM call meanwhile, or
// that is the thread of another conversation's window, multiplexes several
// conversations, as an event loop does; that is remembered for the process.
func (t *tracker) loopThreadFree(p *procState, w *toolWindow, views []windowView) bool {
	tid := w.loopThread()
	if tid == 0 {
		return false
	}
	if _, known := p.multiplexed[tid]; known {
		return false
	}

	from, to := w.readyAt, w.closeAt()
	own := map[trace.SpanID]struct{}{w.source.spanID: {}}
	for _, c := range w.calls {
		if c.sink != nil {
			own[c.sink.spanID] = struct{}{}
		}
	}
	for _, inf := range p.inferences {
		if _, isOwn := own[inf.spanID]; isOwn || inf.tid != tid || inf.end <= from || inf.start >= to {
			continue
		}
		p.multiplexed[tid] = struct{}{}
		return false
	}
	for _, v := range views[1:] {
		if slices.Contains(v.llmThreads, tid) {
			p.multiplexed[tid] = struct{}{}
			return false
		}
	}
	return true
}

// windowNodes are the processes started in [from, to] that no window claimed.
type windowNodes struct {
	all      map[uint32]*procNode
	children map[uint32][]*procNode
	topmost  []*procNode
}

func (at *attribution) windowNodes(from, to int64) windowNodes {
	nodes := windowNodes{all: map[uint32]*procNode{}, children: map[uint32][]*procNode{}}
	for _, n := range at.src.lineage {
		if s := n.startAt(); s >= from && s <= to && n.owner == 0 && at.inScope(n.pid) {
			nodes.all[n.pid] = n
		}
	}
	for _, n := range nodes.all {
		if _, parentInWindow := nodes.all[n.parent]; parentInWindow && n.parent != n.pid {
			nodes.children[n.parent] = append(nodes.children[n.parent], n)
		} else {
			nodes.topmost = append(nodes.topmost, n)
		}
	}
	sortNodes(nodes.topmost)
	return nodes
}

// subtree returns a process started in the window and its descendants.
func (nodes *windowNodes) subtree(top *procNode) []*procNode {
	out := []*procNode{top}
	for i := 0; i < len(out) && len(out) <= len(nodes.all); i++ {
		out = append(out, nodes.children[out[i].pid]...)
	}
	return out
}

// windowEvents returns the opens and connects of [from, to] that no window
// claimed and that can be tool work, those of processes started in the
// window apart: they follow their process.
func (at *attribution) windowEvents(from, to int64, nodes windowNodes) ([]*activityRecord, map[uint32][]*activityRecord) {
	var events []*activityRecord
	nodeEvents := map[uint32][]*activityRecord{}
	for _, e := range at.src.eventsBetween(from, to) {
		if e.owner != 0 || !at.inScope(e.PID) || !at.t.toolWork(at.p, &e.Event) {
			continue
		}
		if _, inWindow := nodes.all[e.PID]; inWindow {
			nodeEvents[e.PID] = append(nodeEvents[e.PID], e)
			continue
		}
		events = append(events, e)
	}
	return events, nodeEvents
}

// inScope tells whether a process is p or one of its descendants, when p's
// activity is reported in the tree of another process.
func (at *attribution) inScope(pid uint32) bool {
	if at.src == at.p || pid == at.root {
		return true
	}
	for depth := 0; depth < maxLineagePerProcess; depth++ {
		n := at.src.lineage[pid]
		if n == nil || n.parent == pid {
			return false
		}
		if n.parent == at.root {
			return true
		}
		pid = n.parent
	}
	return false
}

// coveringViews returns the views open at a time.
func (at *attribution) coveringViews(t int64) []int {
	var out []int
	for i := range at.views {
		if at.views[i].covers(t) {
			out = append(out, i)
		}
	}
	return out
}

// byArguments matches activity with the arguments of the calls that were
// open when it happened. A unique best match decides; ties decide nothing.
func (at *attribution) byArguments(candidates []string, t int64) decision {
	best, bestScore, tie := unassigned, 0, false
	for _, vi := range at.coveringViews(t) {
		for _, ci := range at.views[vi].calls {
			c := &at.calls[ci]
			if t > c.closeAt {
				continue
			}
			score := c.matcher.score(candidates)
			switch {
			case score > bestScore:
				best, bestScore, tie = ci, score, false
			case score == bestScore && score > 0:
				tie = true
			}
		}
	}
	if best == unassigned || tie {
		return undecided()
	}
	return decision{call: best, view: at.calls[best].view, rule: request.ToolAttributionArgumentMatch}
}

// decideNode places a process started in the window that its arguments did
// not decide.
func (at *attribution) decideNode(n *procNode) decision {
	t := n.startAt()
	if owner, owned := at.ancestorOwner(n.parent); owned {
		return at.ownedElsewhere(owner, t)
	}
	if n.parent != at.root {
		// forked by a pool worker that existed before the window
		if d := at.fromSeed(&at.processSeeds, n.parent, t, request.ToolAttributionLineage); d.decided() {
			return d
		}
		return at.byQuietWorker(worker{pid: n.parent}, t)
	}
	if d := at.fromSeed(&at.threadSeeds, n.spawnerTID, t, request.ToolAttributionThread); d.decided() {
		return d
	}
	if d := at.byLoopThread(n.spawnerTID, t); d.decided() {
		return d
	}
	return at.byQuietWorker(worker{pid: at.root, tid: n.spawnerTID}, t)
}

// decideEvent places an open or connect that its arguments did not decide.
func (at *attribution) decideEvent(e *activityRecord) decision {
	if e.PID != at.root {
		if owner, owned := at.ancestorOwner(e.PID); owned {
			return at.ownedElsewhere(owner, e.Time)
		}
		if d := at.fromSeed(&at.processSeeds, e.PID, e.Time, request.ToolAttributionLineage); d.decided() {
			return d
		}
		return at.byQuietWorker(at.workerOf(e.PID, e.TID), e.Time)
	}
	if d := at.fromSeed(&at.threadSeeds, e.TID, e.Time, request.ToolAttributionThread); d.decided() {
		return d
	}
	if d := at.byLoopThread(e.TID, e.Time); d.decided() {
		return d
	}
	return at.byQuietWorker(at.workerOf(e.PID, e.TID), e.Time)
}

// ancestorOwner returns the window that claimed pid or its nearest ancestor.
func (at *attribution) ancestorOwner(pid uint32) (uint64, bool) {
	for depth := 0; depth < maxLineagePerProcess; depth++ {
		n := at.src.lineage[pid]
		if n == nil {
			return 0, false
		}
		if n.owner != 0 {
			return n.owner, true
		}
		pid = n.parent
	}
	return 0, false
}

// ownedElsewhere places activity of a process another window claimed: it is
// that window's while the window is open, else it is undecided here.
func (at *attribution) ownedElsewhere(owner uint64, t int64) decision {
	for i := range at.views {
		if at.views[i].id == owner && at.views[i].covers(t) {
			return decision{call: unassigned, view: i, rule: request.ToolAttributionLineage}
		}
	}
	return at.unclaimed(t)
}

// fromSeed gives activity to the call whose argument-matched work the same
// thread or pool worker did. Executors serve every conversation of a process,
// so this holds only while that call's window is the only one open.
func (at *attribution) fromSeed(s *seeds, key uint32, t int64, rule string) decision {
	call, ok := s.lookup(key)
	if !ok {
		return at.unclaimed(t)
	}
	c := &at.calls[call]
	if covering := at.coveringViews(t); len(covering) != 1 || covering[0] != c.view || t > c.closeAt {
		return at.unclaimed(t)
	}
	return decision{call: call, view: c.view, rule: rule, byThread: true}
}

// byLoopThread gives the activity of a sequential agent's loop thread to the
// single call whose tool the thread was running.
func (at *attribution) byLoopThread(tid uint32, t int64) decision {
	if !at.loopThreadFree || tid != at.views[0].loopThread {
		return at.unclaimed(t)
	}
	return decision{call: at.views[0].calls[0], view: 0, rule: request.ToolAttributionThread, byThread: true}
}

// byQuietWorker gives the activity of a thread or pool worker that does
// nothing outside tool windows to the single call of the only window open:
// a worker that runs the call's tool. A background thread is busy outside
// windows too, so its activity stays on nobody.
func (at *attribution) byQuietWorker(wk worker, t int64) decision {
	v := &at.views[0]
	if len(v.calls) != 1 || v.unmatched {
		return at.unclaimed(t)
	}
	if covering := at.coveringViews(t); len(covering) != 1 {
		return at.unclaimed(t)
	}
	if _, busy := at.busy[wk]; busy {
		return at.unclaimed(t)
	}
	return decision{call: v.calls[0], view: 0, rule: request.ToolAttributionExclusive, byThread: true}
}

// unclaimed places activity no rule decided: on the window span when one
// window was open, nowhere when several were.
func (at *attribution) unclaimed(t int64) decision {
	covering := at.coveringViews(t)
	if len(covering) == 1 {
		return decision{call: unassigned, view: covering[0], rule: request.ToolAttributionAmbiguous}
	}
	return decision{call: unassigned, view: unassigned, rule: request.ToolAttributionAmbiguous, overlap: true}
}

// target returns where a decision puts activity of the window being
// attributed, or nil when it is not that window's.
func (at *attribution) target(act *windowActivity, d decision) (*callActivity, string) {
	if d.view != 0 {
		return nil, ""
	}
	if !d.decided() {
		return &act.shared, request.ToolAttributionAmbiguous
	}
	rule := d.rule
	// a call that never got a result has no proven end: only activity that
	// names it, or descends from what does, is its
	if at.views[0].unmatched && rule != request.ToolAttributionArgumentMatch && rule != request.ToolAttributionLineage {
		return &act.shared, request.ToolAttributionAmbiguous
	}
	act.usedThreadRule = act.usedThreadRule || d.byThread
	return &act.calls[at.calls[d.call].index], rule
}

func (at *attribution) assignNode(act *windowActivity, top *procNode, subtree []*procNode, nodeEvents map[uint32][]*activityRecord, d decision) {
	if d.overlap {
		act.overlap += len(subtree)
		return
	}
	target, rule := at.target(act, d)
	if target == nil {
		return
	}

	target.children = append(target.children, top)
	target.claim(rule, top.startAt())
	if top.exited && top.exitAt <= at.w.closeAt() {
		target.claim(rule, top.exitAt)
	}
	act.procRule[top.pid] = rule
	for _, n := range subtree {
		act.procs[n.pid] = n
		act.events[n.pid] = nodeEvents[n.pid]
		if n != top {
			act.procRule[n.pid] = request.ToolAttributionLineage
		}
	}
}

func (at *attribution) assignEvent(act *windowActivity, e *activityRecord, d decision) {
	if d.overlap {
		act.overlap++
		return
	}
	target, rule := at.target(act, d)
	if target == nil {
		return
	}
	target.events = append(target.events, attributed{rec: e, rule: rule})
	target.claim(rule, e.Time)
}

// commit marks the activity of an emitted window as claimed.
func (act *windowActivity) commit(id uint64) {
	for _, n := range act.procs {
		n.owner = id
	}
	for _, events := range act.events {
		for _, e := range events {
			e.owner = id
		}
	}
	for _, a := range append(slices.Clone(act.calls), act.shared) {
		for _, e := range a.events {
			e.rec.owner = id
		}
	}
}

// seeds maps threads, or pool worker processes, whose argument-matched work
// went to one call. One that did work for several calls decides nothing.
type seeds struct {
	owners map[uint32]int
}

func (s *seeds) add(key uint32, call int, exclude map[uint32]struct{}) {
	if _, excluded := exclude[key]; excluded || key == 0 {
		return
	}
	if prev, ok := s.owners[key]; ok && prev != call {
		s.owners[key] = unassigned
		return
	}
	s.owners[key] = call
}

func (s *seeds) lookup(key uint32) (int, bool) {
	call, ok := s.owners[key]
	return call, ok && call != unassigned
}

// completeness reports the activity lost in [from, to] and whether the
// window saw all of it.
func (t *tracker) completeness(p *procState, from, to int64) (int, bool) {
	dropped, lost := p.lost(from, to)
	return dropped, !lost && t.collected(p, from, to)
}

// toolWork filters out the agent's own LLM traffic and runtime noise: the
// next model call's connect lands inside the window when the client does not
// keep its connection, and so does its CA bundle reload. Time is no filter:
// sequential agents send the next request microseconds after a tool's last
// file access.
func (t *tracker) toolWork(p *procState, e *agentact.Event) bool {
	ka := &t.cfg.KernelActivity
	switch e.Kind {
	case agentact.KindOpen:
		return ka.FileOpen && !t.paths.Matches(e.Path) && !underAny(e.Path, p.caPaths)
	case agentact.KindConnect:
		_, llm := p.endpoints[e.Remote]
		return ka.Connect && !(llm && ka.ExcludeLLMEndpoints)
	}
	return false
}

// underAny tells whether p is one of the paths, or inside one of them.
func underAny(p string, paths []string) bool {
	for _, base := range paths {
		if base != "" && (p == base || strings.HasPrefix(p, strings.TrimSuffix(base, "/")+"/")) {
			return true
		}
	}
	return false
}

func sortNodes(nodes []*procNode) {
	slices.SortFunc(nodes, func(a, b *procNode) int {
		return cmp.Or(cmp.Compare(a.startAt(), b.startAt()), cmp.Compare(a.pid, b.pid))
	})
}

func (n *procNode) startAt() int64 {
	if n.forkAt != 0 {
		return n.forkAt
	}
	return n.execAt
}

func execCandidates(n *procNode) []string {
	candidates := make([]string, 0, len(n.args)+2)
	if n.path != "" {
		candidates = append(candidates, n.path, path.Base(n.path))
	}
	return append(candidates, n.args...)
}

func eventCandidates(e *agentact.Event) []string {
	switch e.Kind {
	case agentact.KindOpen:
		return []string{e.Path, path.Base(e.Path)}
	case agentact.KindConnect:
		return []string{e.Remote.String()}
	}
	return nil
}

// argMatcher compares kernel activity with the string values of a call's
// arguments: exact tokens, then path suffixes of at least two segments, then
// substrings of at least six characters. Only the arguments held in memory
// are used; they are exported only when selected.
type argMatcher struct {
	tokens map[string]struct{}
	values []string
}

func newArgMatcher(arguments string) argMatcher {
	m := argMatcher{tokens: map[string]struct{}{}}
	for _, v := range argumentValues(arguments) {
		m.values = append(m.values, v)
		m.addToken(v)
		m.addToken(path.Base(v))
		for _, tok := range strings.Fields(v) {
			m.addToken(tok)
			m.addToken(path.Base(tok))
		}
	}
	return m
}

func (m *argMatcher) addToken(tok string) {
	if len(tok) >= minExactToken && tok != "." && tok != "/" {
		m.tokens[tok] = struct{}{}
	}
}

// score adds up the best match of each distinct candidate, so that a call
// whose arguments name more of the activity, such as the one argument that
// differs between two `echo` commands, wins over one that shares a word.
func (m *argMatcher) score(candidates []string) int {
	total := 0
	for i, c := range candidates {
		if c == "" || slices.Contains(candidates[:i], c) {
			continue
		}
		total += m.candidateScore(c)
	}
	return total
}

func (m *argMatcher) candidateScore(c string) int {
	if _, ok := m.tokens[c]; ok {
		return scoreExact
	}
	best := 0
	for _, v := range m.values {
		switch {
		case strings.Count(v, "/") >= minSuffixSegments-1 && strings.HasSuffix(c, "/"+strings.TrimPrefix(v, "/")):
			best = max(best, scoreSuffix)
		case len(v) >= minSubstringMatch && strings.Contains(c, v),
			len(c) >= minSubstringMatch && strings.Contains(v, c):
			best = max(best, scoreSubstring)
		}
	}
	return best
}

// argumentValues returns the string and number leaves of the arguments. A
// JSON-encoded string is decoded once more; text that is not JSON (custom
// tools) is one value.
func argumentValues(arguments string) []string {
	var decoded any
	if err := json.Unmarshal([]byte(arguments), &decoded); err != nil {
		if arguments == "" {
			return nil
		}
		return []string{arguments}
	}
	if s, ok := decoded.(string); ok {
		var inner any
		if json.Unmarshal([]byte(s), &inner) == nil {
			decoded = inner
		}
	}

	var values []string
	var walk func(v any)
	walk = func(v any) {
		switch x := v.(type) {
		case string:
			values = append(values, x)
		case float64:
			values = append(values, fmt.Sprint(x))
		case []any:
			for _, item := range x {
				walk(item)
			}
		case map[string]any:
			for _, item := range x {
				walk(item)
			}
		}
	}
	walk(decoded)
	return values
}

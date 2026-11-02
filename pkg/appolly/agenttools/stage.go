// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Package agenttools infers the execution of the tool calls an LLM requests
// from an agent: a tool call is available to the agent when the response
// that requests it ends, and done at the latest when the agent sends its
// result in a later request. The window between both is exported as an
// `execute_tool` span explicitly marked as inferred, with the kernel activity
// of the agent's process tree observed inside the window.
package agenttools // import "go.opentelemetry.io/obi/pkg/appolly/agenttools"

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/gobwas/glob"

	"go.opentelemetry.io/obi/pkg/appolly/app"
	"go.opentelemetry.io/obi/pkg/appolly/app/agentact"
	"go.opentelemetry.io/obi/pkg/appolly/app/request"
	"go.opentelemetry.io/obi/pkg/appolly/app/svc"
	"go.opentelemetry.io/obi/pkg/appolly/discover/exec"
	"go.opentelemetry.io/obi/pkg/config"
	"go.opentelemetry.io/obi/pkg/pipe/msg"
	"go.opentelemetry.io/obi/pkg/pipe/swarm"
)

const (
	tickInterval = time.Second
	// a closed window waits this long for activity events still in flight
	activitySettle = 2 * time.Second
	// a window whose activity was tied to the agent's loop thread waits this
	// much longer for the LLM calls that would show the thread serving
	// another conversation: a client record may only flush at the next
	// request on its connection
	loopThreadHold = 8 * time.Second
	// how long the spans still waiting may take to leave at shutdown
	shutdownFlush = time.Second
)

// caBundleEnv are the variables naming the CA bundles that LLM clients load,
// sometimes once per connection.
var caBundleEnv = [][]byte{
	[]byte("SSL_CERT_FILE="),
	[]byte("SSL_CERT_DIR="),
	[]byte("REQUESTS_CA_BUNDLE="),
	[]byte("CURL_CA_BUNDLE="),
	[]byte("AWS_CA_BUNDLE="),
	[]byte("NODE_EXTRA_CA_CERTS="),
}

// llmRouteSuffixes are the paths of LLM APIs. A process that serves them is a
// gateway or a proxy: the tool calls in its own LLM calls are its clients'.
var llmRouteSuffixes = []string{
	"/chat/completions", "/v1/completions", "/v1/responses", "/v1/messages",
	":generateContent", ":streamGenerateContent",
}

// bedrockRouteSuffixes follow /model/{modelId} in the Bedrock runtime API.
var bedrockRouteSuffixes = []string{"/converse", "/converse-stream", "/invoke", "/invoke-with-response-stream"}

// ToolSpans adds inferred tool-call spans to the spans that flow through it,
// which it forwards unchanged. activity, roots and processEvents may be nil.
func ToolSpans(
	cfg *config.GenAIToolCallsConfig,
	maxTransactionTime time.Duration,
	activity *msg.Queue[[]agentact.Event],
	roots *agentact.RootRegistry,
	processEvents *msg.Queue[exec.ProcessEvent],
	input, output *msg.Queue[[]request.Span],
) swarm.InstanceFunc {
	return func(_ context.Context) (swarm.RunFunc, error) {
		if !cfg.Enabled {
			return swarm.Bypass(input, output)
		}

		t := newTracker(cfg, maxTransactionTime, roots)
		in := input.Subscribe(msg.SubscriberName("agenttools.ToolSpans"))
		var activityCh <-chan []agentact.Event
		if activity != nil {
			activityCh = activity.Subscribe(msg.SubscriberName("agenttools.ToolSpans.activity"))
		}
		var processCh <-chan exec.ProcessEvent
		if processEvents != nil {
			processCh = processEvents.Subscribe(msg.SubscriberName("agenttools.ToolSpans.processes"))
		}

		return func(ctx context.Context) {
			defer output.Close()

			ticker := time.NewTicker(tickInterval)
			defer ticker.Stop()

			for {
				select {
				case <-ctx.Done():
					t.shutdown(output)
					return
				case spans, ok := <-in:
					if !ok {
						t.shutdown(output)
						return
					}
					output.SendCtx(ctx, t.onSpans(spans))
				case events, ok := <-activityCh:
					if !ok {
						activityCh = nil
						continue
					}
					t.onActivity(events)
				case pe, ok := <-processCh:
					if !ok {
						processCh = nil
						continue
					}
					t.onProcessEvent(&pe)
				case now := <-ticker.C:
					if out := t.tick(t.now(), now); len(out) > 0 {
						output.SendCtx(ctx, out)
					}
				}
			}
		}, nil
	}
}

func (t *tracker) onSpans(spans []request.Span) []request.Span {
	// a gateway's server span may follow, in its batch, the client span that
	// forwarded the call
	for i := range spans {
		if servesLLMRoute(&spans[i]) {
			t.onLLMServer(&spans[i])
		}
	}

	now := t.now()
	var emitted []request.Span
	for i := range spans {
		if isInference(&spans[i]) {
			emitted = append(emitted, t.onInference(&spans[i], now)...)
		}
	}
	if len(emitted) == 0 {
		return spans
	}
	// the input batch may share its backing array with other subscribers
	return append(slices.Clip(spans), emitted...)
}

// servesLLMRoute tells whether a span is an LLM API call served by the process.
func servesLLMRoute(span *request.Span) bool {
	if span.Type != request.EventTypeHTTP {
		return false
	}
	route, _, _ := strings.Cut(span.Path, "?")
	if route == "/responses" || hasAnySuffix(route, llmRouteSuffixes) {
		return true
	}
	return strings.HasPrefix(route, "/model/") && hasAnySuffix(route, bedrockRouteSuffixes)
}

func hasAnySuffix(s string, suffixes []string) bool {
	for _, suffix := range suffixes {
		if strings.HasSuffix(s, suffix) {
			return true
		}
	}
	return false
}

// onLLMServer leaves out, for good, a process that serves an LLM API: the tool
// calls of its LLM calls are executed by its clients, not by it.
func (t *tracker) onLLMServer(span *request.Span) {
	p := t.store.agent(span.Pid)
	if p.llmServer {
		return
	}
	alog().Debug("process serves an LLM API, not inferring its tool calls", "pid", span.Pid.HostPID, "path", span.Path)
	p.llmServer = true
	p.pending = nil
	p.settling = nil
	t.disarm(p)
}

// selected tells whether the tool calls of the process are inferred.
func (t *tracker) selected(p *procState, span *request.Span) bool {
	if p.llmServer {
		return false
	}
	if !p.selectionKnown {
		p.excluded = !t.services.includes(span.Service.UID.Name)
		p.selectionKnown = true
	}
	return !p.excluded
}

// serviceSelection matches service names with the configured globs.
type serviceSelection struct {
	include, exclude []glob.Glob
}

func newServiceSelection(cfg *config.GenAIToolCallsServices) serviceSelection {
	compile := func(patterns []string) []glob.Glob {
		var globs []glob.Glob
		for _, p := range patterns {
			if g, err := glob.Compile(p); err == nil {
				globs = append(globs, g)
			}
		}
		return globs
	}
	return serviceSelection{include: compile(cfg.Include), exclude: compile(cfg.Exclude)}
}

func (s *serviceSelection) includes(name string) bool {
	return !matchesAny(s.exclude, name) && (len(s.include) == 0 || matchesAny(s.include, name))
}

// explicit tells whether a service is selected by name, not by default.
func (s *serviceSelection) explicit(name string) bool {
	return len(s.include) > 0 && s.includes(name)
}

func matchesAny(globs []glob.Glob, name string) bool {
	for _, g := range globs {
		if g.Match(name) {
			return true
		}
	}
	return false
}

func (t *tracker) onActivity(events []agentact.Event) {
	for i := range events {
		e := &events[i]
		p, ok := t.store.activity(app.PID(e.RootPID), t.mayBeAgent(e))
		if !ok {
			continue
		}
		if e.RootStart != 0 {
			p.rootStart = e.RootStart
		}

		t.noteKernelRoot(e)

		switch {
		case e.Kind == agentact.KindExit && e.PID == e.RootPID:
			p.markExited(e.Time)
		case e.Kind == agentact.KindDrops:
			p.addLoss(loss{from: e.DroppedSince, to: e.Time, count: e.Dropped})
		case e.Kind == agentact.KindOpen && t.paths.Matches(e.Path):
		case e.Kind == agentact.KindOpen || e.Kind == agentact.KindConnect:
			p.addActivity(e)
		default:
			p.trackLineage(e)
		}
	}
}

// noteKernelRoot learns that an agent's activity is reported in the tree of
// another instrumented process, and arms that process for it.
func (t *tracker) noteKernelRoot(e *agentact.Event) {
	if e.PID == e.RootPID {
		return
	}
	pid := e.PID
	if e.Kind == agentact.KindFork {
		pid = e.ParentPID
	}
	p, ok := t.store.procs.Peek(app.PID(pid))
	if !ok || pid == e.RootPID || p.kernelRoot == e.RootPID {
		return
	}
	p.kernelRoot = e.RootPID
	p.kernelRootKnown = true
	if p.armed {
		t.armRoot(p, e.RootPID, t.now())
	}
}

// findKernelRoot looks, once, for the tree of another instrumented process
// that a new agent was already reported in.
func (t *tracker) findKernelRoot(p *procState) {
	p.kernelRootKnown = true
	pid := uint32(p.pid.HostPID)
	for _, other := range t.store.all() {
		// an exited descendant may have had the pid before
		if n := other.lineage[pid]; other != p && n != nil && !n.exited {
			p.kernelRoot = uint32(other.pid.HostPID)
			return
		}
	}
}

// activitySource returns the state holding the kernel activity of a process:
// its own, or that of the instrumented ancestor it is reported under.
func (t *tracker) activitySource(p *procState) *procState {
	if p.kernelRoot == 0 {
		return p
	}
	if src, ok := t.store.peek(app.PID(p.kernelRoot)); ok {
		return src
	}
	return p
}

// mayBeAgent tells whether an event of a root without state is worth a
// state: without arming, every root reports, and with it only armed roots
// and roots in their probation period do.
func (t *tracker) mayBeAgent(e *agentact.Event) bool {
	if t.roots == nil {
		return true
	}
	return t.inProbation(e.RootStart, e.Time)
}

func (t *tracker) inProbation(rootStart, at int64) bool {
	return t.probation > 0 && rootStart > 0 && at-rootStart <= t.probation
}

// collected tells whether the kernel reported the process's activity during
// all of [from, to]: during its probation period or since it was armed.
func (t *tracker) collected(p *procState, from, to int64) bool {
	if t.roots == nil {
		return true
	}
	if p.armed && from >= p.armedAt {
		return true
	}
	if p.rootStart == 0 || t.probation == 0 {
		return false
	}
	probationEnd := p.rootStart + t.probation
	return to <= probationEnd || (p.armed && p.armedAt <= probationEnd && from >= p.rootStart)
}

// arm has the kernel report the activity of a process known to run tool
// calls, in the tree it is reported in.
func (t *tracker) arm(p *procState, now int64) {
	if t.roots == nil || p.armed || p.exited || p.llmServer || p.excluded {
		return
	}
	root := uint32(p.pid.HostPID)
	if p.kernelRoot != 0 {
		root = p.kernelRoot
	}
	t.armRoot(p, root, now)
}

func (t *tracker) armRoot(p *procState, root uint32, now int64) {
	if !t.roots.Arm(root) {
		alog().Debug("could not arm the kernel activity of a process", "pid", p.pid.HostPID, "root", root)
		return
	}
	if !p.armed {
		p.armed = true
		p.armedAt = now
	}
	p.armedPIDs = append(p.armedPIDs, root)
	if src := t.activitySource(p); src != p && !src.armed {
		src.armed = true
		src.armedAt = now
	}
}

func (t *tracker) disarm(p *procState) {
	if t.roots == nil || !p.armed {
		return
	}
	for _, pid := range p.armedPIDs {
		t.roots.Disarm(pid)
	}
	p.armed = false
	p.armedPIDs = nil
}

func (t *tracker) onProcessEvent(pe *exec.ProcessEvent) {
	if pe.File == nil {
		return
	}
	pid := pe.File.Pid()
	switch pe.Type {
	case exec.ProcessEventCreated:
		// a new process may reuse the pid of one that exited
		t.store.forget(pid)
		service := pe.File.ServiceAttrs()
		t.onProcessCreated(pid, &service)
	case exec.ProcessEventTerminated:
		t.onProcessTerminated(pid, t.now())
	}
}

// onProcessCreated arms a service selected by name from the moment OBI
// instruments it, so that even its first tool call is observed.
func (t *tracker) onProcessCreated(pid app.PID, service *svc.Attrs) {
	if t.roots == nil || !t.services.explicit(service.UID.Name) {
		return
	}
	p, _ := t.store.activity(pid, true)
	p.selectionKnown = true
	t.arm(p, t.now())
}

func (t *tracker) onProcessTerminated(pid app.PID, now int64) {
	if p, ok := t.store.peek(pid); ok && !p.exited {
		p.markExited(now)
	}
}

// exitGraceOf is how long after its exit the spans of a process may still
// arrive: longer for processes whose spans were held for their parents.
func (t *tracker) exitGraceOf(p *procState) time.Duration {
	return min(t.exitGrace, max(minExitGrace, p.maxLag+exitGraceLag))
}

// tick expires calls that waited too long, emits settled windows and closes
// the calls of processes that exited.
func (t *tracker) tick(now int64, wall time.Time) []request.Span {
	var out []request.Span
	for _, p := range t.store.all() {
		// the process's last spans flush when its connections close, so its
		// exit is acted on only once they had time to arrive
		if p.exited && wall.Sub(p.exitSeenAt) >= t.exitGraceOf(p) {
			out = append(out, t.closeUnmatched(p, p.pending, p.exitAt, request.ToolOutcomeUnmatchedExit, request.ToolWindowEndProcessExit)...)
			out = append(out, t.emitSettled(p, wall, true)...)
			t.disarm(p)
			t.store.remove(p)
			continue
		}

		out = append(out, t.expire(p, now)...)
		out = append(out, t.emitSettled(p, wall, false)...)
		p.trim(now, t.retention)
		t.dropIdleCandidate(p, now)
	}
	return out
}

// dropIdleCandidate forgets a process that made no LLM call once its
// probation period and the retention of its activity are over.
func (t *tracker) dropIdleCandidate(p *procState, now int64) {
	if t.roots == nil || p.armed {
		return
	}
	if _, isAgent := t.store.procs.Peek(p.pid.HostPID); isAgent {
		return
	}
	if !t.inProbation(p.rootStart, now-t.retention.Nanoseconds()) {
		t.store.remove(p)
	}
}

func (t *tracker) expire(p *procState, now int64) []request.Span {
	maxWindow := t.cfg.MaxWindow.Nanoseconds()
	var expired []*pendingCall
	for _, c := range p.pending {
		if now-c.window.readyAt > maxWindow {
			expired = append(expired, c)
		}
	}
	if len(expired) == 0 {
		return nil
	}
	return t.closeUnmatched(p, expired, expired[0].window.readyAt+maxWindow, request.ToolOutcomeUnmatchedTimeout, request.ToolWindowEndTimeout)
}

// emitSettled emits the windows whose settle time passed before wall, or all
// of them when forced.
func (t *tracker) emitSettled(p *procState, wall time.Time, force bool) []request.Span {
	var out []request.Span
	// the windows still settling stay visible to the attribution of the
	// others, which they may overlap
	for _, s := range slices.Clone(p.settling) {
		if !force && wall.Before(s.deadline) {
			continue
		}
		spans, deferred := t.windowSpans(p, s.window, force)
		if deferred {
			p.delaySettling(s.window, wall.Add(loopThreadHold))
			continue
		}
		p.removeSettling(s.window)
		out = append(out, spans...)
	}
	return out
}

// shutdown sends, with what arrived, the windows that wait for late evidence
// and the calls of processes that exited without answering them.
func (t *tracker) shutdown(output *msg.Queue[[]request.Span]) {
	out := t.flush()
	if len(out) == 0 {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), shutdownFlush)
	defer cancel()
	output.SendCtx(ctx, out)
}

// flush emits every window waiting for late evidence, and closes the calls
// of processes that exited.
func (t *tracker) flush() []request.Span {
	var out []request.Span
	for _, p := range t.store.all() {
		if p.exited {
			out = append(out, t.closeUnmatched(p, p.pending, p.exitAt, request.ToolOutcomeUnmatchedExit, request.ToolWindowEndProcessExit)...)
		}
		out = append(out, t.emitSettled(p, time.Time{}, true)...)
	}
	return out
}

// caBundlePaths reads the CA bundle paths from the process environment. They
// are excluded from file activity: clients reopen them for each connection.
func caBundlePaths(pid uint32) []string {
	paths := []string{}
	environ, err := os.ReadFile(filepath.Join("/proc", pidString(pid), "environ"))
	if err != nil {
		return paths
	}
	for _, kv := range bytes.Split(environ, []byte{0}) {
		for _, name := range caBundleEnv {
			if value, ok := bytes.CutPrefix(kv, name); ok && len(value) > 0 {
				paths = append(paths, string(value))
			}
		}
	}
	return paths
}

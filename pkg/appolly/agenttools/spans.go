// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package agenttools // import "go.opentelemetry.io/obi/pkg/appolly/agenttools"

import (
	"cmp"
	"path"
	"slices"

	"go.opentelemetry.io/otel/attribute"
	semconv "go.opentelemetry.io/otel/semconv/v1.41.0"
	"go.opentelemetry.io/otel/trace"

	"go.opentelemetry.io/obi/pkg/appolly/app/agentact"
	"go.opentelemetry.io/obi/pkg/appolly/app/request"
	"go.opentelemetry.io/obi/pkg/config"
	"go.opentelemetry.io/obi/pkg/export/attributes"
	"go.opentelemetry.io/obi/pkg/export/otel/idgen"
)

const (
	linkKindKey      = attribute.Key("obi.link.kind")
	linkKindSource   = "tool_call_source"
	linkKindSink     = "tool_result_sink"
	attributionKey   = attribute.Key("obi.gen_ai.tool.attribution")
	eventFileOpen    = "obi.file.open"
	eventNetConnect  = "obi.net.connect"
	fileOpenFlags    = attribute.Key("obi.file.open.flags")
	fileOpenResult   = attribute.Key("obi.file.open.result")
	defaultToolType  = "function"
	unknownExecName  = "unknown"
	networkTransport = "tcp"
	// process spans per window: a build tool forks thousands of processes
	maxProcessSpansPerWindow = 256
)

// windowSpans builds the spans of a closed window. The spans sit next to the
// inference span that requested the calls, under its parent if it has one,
// and link to it and to the request that carried the results: OBI saw
// neither the tool run nor who ran it, so it asserts no parent.
//
// It reports deferred when the window should wait before it is emitted: some
// activity was tied to the call by its thread alone, and the LLM calls or
// windows that would show another conversation on that thread may still be
// in flight.
func (t *tracker) windowSpans(p *procState, w *toolWindow, force bool) (spans []request.Span, deferred bool) {
	if p.llmServer || p.excluded {
		return nil, false
	}

	var act *windowActivity
	if t.cfg.KernelActivity.Enabled {
		act = t.attribute(p, w)
		if act.usedThreadRule && !force && !w.deferred {
			w.deferred = true
			return nil, true
		}
		act.commit(w.id)
	}
	t.rememberWindow(p, w)

	emit := make([]bool, len(w.calls))
	anyEmitted := false
	for i, c := range w.calls {
		emit[i] = t.shouldEmit(c, act, i)
		anyEmitted = anyEmitted || emit[i]
	}
	if !anyEmitted {
		return nil, false
	}

	b := &spanBuilder{cfg: &t.cfg.KernelActivity, w: w, act: act, rootPID: uint32(p.pid.HostPID), traceID: w.source.traceID}
	if !b.traceID.IsValid() {
		// one trace for the whole window, so that its spans stay a tree
		b.traceID = idgen.RandomTraceID()
	}
	if act != nil {
		b.eventBudget = t.cfg.KernelActivity.MaxEventsPerWindow
		b.processBudget = maxProcessSpansPerWindow
		b.children = childIndex(act.procs)
	}

	parentID := w.source.parentSpanID
	var out []request.Span
	if len(w.calls) > 1 {
		ws := b.windowSpan()
		out = append(out, ws)
		parentID = ws.SpanID
		if act != nil {
			out = append(out, b.processSpans(ws.SpanID, act.shared.children, &out[0].InferredTool.Activity)...)
		}
	}

	for i, c := range w.calls {
		if !emit[i] {
			continue
		}
		s := b.toolSpan(c, i, parentID)
		out = append(out, s)
		if act != nil {
			out = append(out, b.processSpans(s.SpanID, act.calls[i].children, &out[len(out)-1].InferredTool.Activity)...)
		}
	}

	return out, false
}

// rememberWindow keeps the bounds and arguments of an emitted window, which
// windows emitted later may overlap.
func (t *tracker) rememberWindow(p *procState, w *toolWindow) {
	if len(p.past) == maxPastWindows {
		p.past = append(p.past[:0], p.past[1:]...)
	}
	p.past = append(p.past, pastWindow{
		id:         w.id,
		from:       w.readyAt,
		to:         w.closeAt(),
		loopThread: w.loopThread(),
		llmThreads: w.llmThreads(),
		calls:      callsOf(w),
	})
}

// shouldEmit applies the export options of calls that never got a result.
func (t *tracker) shouldEmit(c *pendingCall, act *windowActivity, i int) bool {
	switch c.outcome {
	case request.ToolOutcomeUnmatchedTimeout:
		if t.cfg.OnTimeout == config.ToolCallTimeoutDrop {
			return false
		}
	case request.ToolOutcomeUnmatchedExit:
	default:
		return true
	}

	if !t.cfg.EmitUnmatched {
		return false
	}
	if t.cfg.UnmatchedMinActivity == 0 {
		return true
	}
	return act != nil && len(act.calls[i].events)+len(act.calls[i].children) >= t.cfg.UnmatchedMinActivity
}

type spanBuilder struct {
	cfg     *config.GenAIToolKernelActivityConfig
	w       *toolWindow
	act     *windowActivity
	rootPID uint32
	traceID trace.TraceID
	// span events and process spans the window may still attach
	eventBudget   int
	processBudget int
	children      map[uint32][]*procNode
}

func (b *spanBuilder) base(typ request.EventType, parentID trace.SpanID, start, end int64) request.Span {
	s := request.Span{
		Type:         typ,
		RequestStart: start,
		Start:        start,
		End:          max(start, end),
		TraceID:      b.traceID,
		SpanID:       idgen.RandomSpanID(),
		ParentSpanID: parentID,
		TraceFlags:   b.w.source.traceFlags,
		Pid:          b.w.pid,
		Service:      b.w.service,
	}
	// these spans carry no request: no RED metrics or span metrics
	request.SetIgnoreMetrics(&s)
	return s
}

func (b *spanBuilder) toolSpan(c *pendingCall, i int, parentID trace.SpanID) request.Span {
	s := b.base(request.EventTypeGenAIToolExec, parentID, b.w.readyAt, c.closeAt)
	s.Links = []request.SpanLink{sourceLink(b.w)}
	if c.sink != nil {
		s.Links = append(s.Links, sinkLink(c.sink))
	}

	tool := &request.InferredTool{
		Name:             c.call.Name,
		CallID:           c.call.ID,
		Type:             cmp.Or(c.call.Type, defaultToolType),
		Provider:         c.provider,
		Arguments:        c.call.Arguments,
		IsError:          c.result.IsError,
		PayloadTruncated: c.truncated,
		WindowStart:      request.ToolWindowStartResponseEnd,
		WindowEnd:        c.endSource,
		Outcome:          c.outcome,
		ParallelCount:    len(b.w.calls),
	}
	if !c.result.IsError && (c.outcome == request.ToolOutcomeResult || c.outcome == request.ToolOutcomeResultByName) {
		tool.Result = string(c.result.Raw)
	}

	if b.act != nil {
		a := &b.act.calls[i]
		b.observe(tool, len(b.w.calls) == 1)
		tool.Attribution = a.rule
		s.Events = b.spanEvents(a.events, &tool.Activity)
		if a.last > a.first {
			tool.ActiveNs = a.last - a.first
		}
		if len(b.w.calls) == 1 {
			// a single call has no window span: the activity of its window
			// that no rule tied to it is only counted
			tool.Activity.Ambiguous = len(b.act.shared.events) + len(b.act.shared.children)
			if tool.Attribution == "" && tool.Activity.Ambiguous+tool.Activity.AmbiguousOverlap > 0 {
				tool.Attribution = request.ToolAttributionAmbiguous
			}
		}
		if len(a.children) == 1 && b.countProcesses(a.children) == 1 {
			tool.Command = b.observedProcess(a.children[0], c.closeAt)
			// a failing command is weaker evidence than a result flagged as an error
			tool.IsError = tool.IsError || (tool.Command.Exited && tool.Command.ExitCode != 0)
		}
	}

	s.InferredTool = tool
	return s
}

// observe records that the window's activity was collected. What concerns
// the window as a whole goes to the span that stands for it: the window span,
// or the span of its only call.
func (b *spanBuilder) observe(tool *request.InferredTool, standsForWindow bool) {
	tool.ActivityObserved = true
	tool.ActivityComplete = b.act.complete
	if standsForWindow {
		tool.Activity.Dropped = b.act.dropped
		tool.Activity.AmbiguousOverlap = b.act.overlap
	}
}

// windowSpan groups parallel calls. It holds the activity no call can claim.
func (b *spanBuilder) windowSpan() request.Span {
	end := b.w.closeAt()
	s := b.base(request.EventTypeGenAIToolWindow, b.w.source.parentSpanID, b.w.readyAt, end)
	s.Links = []request.SpanLink{sourceLink(b.w)}

	tool := &request.InferredTool{
		Provider:      b.w.calls[0].provider,
		WindowStart:   request.ToolWindowStartResponseEnd,
		ParallelCount: len(b.w.calls),
	}
	var sinks []trace.SpanID
	for _, c := range b.w.calls {
		if c.closeAt == end {
			tool.WindowEnd = c.endSource
		}
		if c.sink != nil && !slices.Contains(sinks, c.sink.spanID) {
			sinks = append(sinks, c.sink.spanID)
			s.Links = append(s.Links, sinkLink(c.sink))
		}
	}

	if b.act != nil {
		b.observe(tool, true)
		tool.Attribution = request.ToolAttributionAmbiguous
		s.Events = b.spanEvents(b.act.shared.events, &tool.Activity)
	}

	s.InferredTool = tool
	return s
}

func sourceLink(w *toolWindow) request.SpanLink {
	return request.SpanLink{
		TraceID:    w.source.traceID,
		SpanID:     w.source.spanID,
		TraceFlags: w.source.traceFlags,
		Attributes: []attribute.KeyValue{linkKindKey.String(linkKindSource)},
	}
}

func sinkLink(sink *spanRef) request.SpanLink {
	return request.SpanLink{
		TraceID:    sink.traceID,
		SpanID:     sink.spanID,
		TraceFlags: sink.traceFlags,
		Attributes: []attribute.KeyValue{linkKindKey.String(linkKindSink)},
	}
}

func childIndex(procs map[uint32]*procNode) map[uint32][]*procNode {
	children := map[uint32][]*procNode{}
	for _, n := range procs {
		if n.parent != n.pid {
			children[n.parent] = append(children[n.parent], n)
		}
	}
	for _, c := range children {
		sortNodes(c)
	}
	return children
}

// processSpans builds a span per process started in the window, following
// the CLI client span shape, with grandchildren under their parent. The
// activity of the subtree adds to counts, those of the owning tool span.
func (b *spanBuilder) processSpans(parentID trace.SpanID, nodes []*procNode, counts *request.ToolActivityCounts) []request.Span {
	var out []request.Span
	windowEnd := b.w.closeAt()
	for _, n := range nodes {
		counts.Execs++
		if b.processBudget == 0 {
			continue
		}
		b.processBudget--

		proc := b.observedProcess(n, windowEnd)
		end := windowEnd
		if !proc.OutlivedWindow {
			end = n.exitAt
		}

		s := b.base(request.EventTypeProcessExec, parentID, n.startAt(), end)
		events := make([]attributed, 0, len(b.act.events[n.pid]))
		for _, e := range b.act.events[n.pid] {
			events = append(events, attributed{rec: e, rule: request.ToolAttributionLineage})
		}
		s.Events = b.spanEvents(events, counts)
		s.Process = proc
		out = append(out, s)
		out = append(out, b.processSpans(s.SpanID, b.children[n.pid], counts)...)
	}
	return out
}

func (b *spanBuilder) observedProcess(n *procNode, windowEnd int64) *request.ObservedProcess {
	return &request.ObservedProcess{
		PID:            n.pid,
		ExecutableName: b.executableName(n),
		ExecutablePath: n.path,
		CommandArgs:    n.args,
		ExitCode:       n.exitCode,
		Exited:         n.exited,
		OutlivedWindow: !n.exited || n.exitAt > windowEnd,
		Attribution:    b.act.procRule[n.pid],
	}
}

// executableName names a process after its program; a process that forked
// without exec runs the program of its nearest ancestor that exec'd.
func (b *spanBuilder) executableName(n *procNode) string {
	for depth := 0; n != nil && depth < maxLineagePerProcess; depth++ {
		if n.path != "" {
			return path.Base(n.path)
		}
		n = b.act.procs[n.parent]
	}
	return unknownExecName
}

func (b *spanBuilder) countProcesses(nodes []*procNode) int {
	count := 0
	for _, n := range nodes {
		count += 1 + b.countProcesses(b.children[n.pid])
	}
	return count
}

// spanEvents converts opens and connects to span events, deduplicated and
// capped, strongest evidence first; counts holds every attributed event,
// including those beyond the caps.
func (b *spanBuilder) spanEvents(events []attributed, counts *request.ToolActivityCounts) []request.SpanEvent {
	type eventKey struct {
		kind         agentact.Kind
		pid          uint32
		path         string
		result       int32
		flags        uint32
		remote, peer string
	}

	ordered := slices.Clone(events)
	slices.SortStableFunc(ordered, func(a, c attributed) int {
		return cmp.Or(cmp.Compare(rulePriority(a.rule), rulePriority(c.rule)), cmp.Compare(a.rec.Time, c.rec.Time))
	})

	var out []request.SpanEvent
	seen := map[eventKey]struct{}{}
	for _, a := range ordered {
		e := &a.rec.Event
		switch e.Kind {
		case agentact.KindOpen:
			counts.FileOpens++
		case agentact.KindConnect:
			counts.Connects++
		default:
			continue
		}

		key := eventKey{kind: e.Kind, pid: e.PID, path: e.Path, result: e.Result, flags: e.OpenFlags, remote: e.Remote.String(), peer: e.Local.String()}
		if _, dup := seen[key]; dup {
			continue
		}
		seen[key] = struct{}{}

		if len(out) == b.cfg.MaxEventsPerSpan || b.eventBudget == 0 {
			continue
		}
		b.eventBudget--
		out = append(out, b.spanEvent(e, a.rule))
	}
	slices.SortStableFunc(out, func(a, c request.SpanEvent) int { return cmp.Compare(a.Time, c.Time) })
	return out
}

func (b *spanBuilder) spanEvent(e *agentact.Event, rule string) request.SpanEvent {
	var attrs []attribute.KeyValue
	name := eventFileOpen
	if e.Kind == agentact.KindOpen {
		attrs = append(attrs,
			semconv.FilePath(attributes.SanitizeUTF8(e.Path)),
			fileOpenFlags.Int64(int64(e.OpenFlags)),
			fileOpenResult.Int64(int64(e.Result)),
		)
	} else {
		name = eventNetConnect
		attrs = append(attrs,
			semconv.ServerAddress(e.Remote.Addr().String()),
			semconv.ServerPort(int(e.Remote.Port())),
			semconv.NetworkTransportKey.String(networkTransport),
		)
		if e.Local.IsValid() {
			attrs = append(attrs, semconv.NetworkLocalPort(int(e.Local.Port())))
		}
	}
	if e.PID != b.rootPID {
		attrs = append(attrs, semconv.ProcessPID(int(e.PID)))
	}
	attrs = append(attrs, attributionKey.String(rule))
	return request.SpanEvent{Name: name, Time: e.Time, Attributes: attrs}
}

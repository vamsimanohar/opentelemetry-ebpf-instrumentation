// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//go:build linux

package generictracer // import "go.opentelemetry.io/obi/pkg/internal/ebpf/generictracer"

import (
	"bytes"
	"context"
	"errors"
	"net/netip"
	"runtime"
	"sync"
	"time"
	"unsafe"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/asm"

	"go.opentelemetry.io/obi/pkg/appolly/app/agentact"
	"go.opentelemetry.io/obi/pkg/config"
	ebpfcommon "go.opentelemetry.io/obi/pkg/ebpf/common"
	"go.opentelemetry.io/obi/pkg/ebpf/ringbuf"
)

const (
	// mirror enum agent_activity_flag and enum agent_activity_sizes in
	// bpf/common/agent_activity.h
	agentActivityTruncated    = 1 << 0
	agentActivityRelativePath = 1 << 1
	agentActivityPathLen      = 256
	signalExitBase            = 128

	agentActivityDropsReport = 30 * time.Second
	// the smallest ring the kernel accepts, for a feature that is off
	disabledRingSize = 4096
	// arm64 kernels have BPF trampolines since 6.0
	arm64TrampolineMajor = 6
)

var agentActivityHeaderSize = int(unsafe.Offsetof(BpfAgentActivityEventT{}.Buf))

// agentActivityPrograms are only attached when kernel activity is observed.
var agentActivityPrograms = []string{
	BpfProgObiRawTpAgentFork,
	BpfProgObiRawTpAgentExec,
	BpfProgObiRawTpAgentExit,
	BpfProgObiFexitAgentOpenat2,
	BpfProgObiKprobeAgentOpenat2,
	BpfProgObiKretprobeAgentOpenat2,
}

// agentActivityMaps only need room when kernel activity is observed.
var agentActivityMaps = []string{
	BpfMapAgentLineage,
	BpfMapAgentRate,
	BpfMapAgentRoots,
	BpfMapAgentOpenArgs,
}

// agentActivity returns the kernel activity settings, or nil when inferred
// tool calls do not observe kernel activity.
func (p *Tracer) agentActivity() *config.GenAIToolKernelActivityConfig {
	toolCalls := &p.cfg.EBPF.PayloadExtraction.HTTP.GenAI.ToolCalls
	if !toolCalls.Enabled || !toolCalls.KernelActivity.Enabled {
		return nil
	}
	return &toolCalls.KernelActivity
}

// stubProgram reduces a program to an immediate return. A tracing program
// becomes a kprobe, so that loading it needs no attach target in the kernel.
func stubProgram(prog *ebpf.ProgramSpec) {
	prog.Instructions = asm.Instructions{asm.Mov.Imm(asm.R0, 0), asm.Return()}
	if prog.Type == ebpf.Tracing {
		prog.Type = ebpf.Kprobe
		prog.AttachType = ebpf.AttachNone
		prog.AttachTo = ""
	}
}

// disableAgentActivity stubs the programs, so that kernels the feature is not
// used on never verify them, and shrinks their maps.
func disableAgentActivity(spec *ebpf.CollectionSpec) {
	for _, name := range agentActivityPrograms {
		if prog, ok := spec.Programs[name]; ok {
			stubProgram(prog)
		}
	}
	if ring, ok := spec.Maps[BpfMapAgentActivityEvents]; ok {
		ring.MaxEntries = disabledRingSize
	}
	for _, name := range agentActivityMaps {
		if m, ok := spec.Maps[name]; ok {
			m.MaxEntries = 1
		}
	}
}

// openFexitSupported tells whether file opens are observed through the BPF
// trampoline, which costs the opens of every process on the host far less
// than the kprobe and kretprobe the kernels without one need.
var openFexitSupported = sync.OnceValue(func() bool {
	if runtime.GOARCH == "arm64" {
		if major, _ := ebpfcommon.KernelVersion(); major < arm64TrampolineMajor {
			return false
		}
	}
	found, err := ebpfcommon.KernelHasSymbol(ebpfcommon.KSymDoSysOpenat2)
	return err == nil && found
})

// prepareAgentOpen keeps one of the two ways to observe opens.
func (p *Tracer) prepareAgentOpen(spec *ebpf.CollectionSpec) {
	unused := []string{BpfProgObiFexitAgentOpenat2}
	if openFexitSupported() {
		unused = []string{BpfProgObiKprobeAgentOpenat2, BpfProgObiKretprobeAgentOpenat2}
	}
	for _, name := range unused {
		if prog, ok := spec.Programs[name]; ok {
			stubProgram(prog)
		}
	}
}

func (p *Tracer) agentActivityConstants(m map[string]any) {
	ka := p.agentActivity()
	if ka == nil {
		return
	}
	m[BpfVarAgentActivityEnabled] = true
	m[BpfVarAgentActivityExecEnabled] = ka.Exec
	m[BpfVarAgentActivityOpenEnabled] = ka.FileOpen
	m[BpfVarAgentActivityConnectEnabled] = ka.Connect
	m[BpfVarAgentActivityMaxDepth] = uint32(ka.MaxLineageDepth)
	m[BpfVarAgentActivityRateLimit] = uint32(ka.RateLimitPerProcess)
	m[BpfVarAgentActivityProbationNs] = uint64(ka.Probation.Nanoseconds())
}

func (p *Tracer) agentActivityKProbes(kp map[string]ebpfcommon.ProbeDesc) {
	if ka := p.agentActivity(); ka == nil || !ka.FileOpen || openFexitSupported() {
		return
	}
	kp["do_sys_openat2"] = ebpfcommon.ProbeDesc{
		Required: false,
		Start:    p.bpfObjects.ObiKprobeAgentOpenat2,
		End:      p.bpfObjects.ObiKretprobeAgentOpenat2,
	}
}

func (p *Tracer) agentActivityTracing() []*ebpfcommon.Tracing {
	if ka := p.agentActivity(); ka == nil || !ka.FileOpen || !openFexitSupported() {
		return nil
	}
	return []*ebpfcommon.Tracing{{Program: p.bpfObjects.ObiFexitAgentOpenat2, AttachAs: ebpf.AttachTraceFExit}}
}

func (p *Tracer) agentActivityTracepoints(tp map[string]ebpfcommon.ProbeDesc) {
	ka := p.agentActivity()
	if ka == nil {
		return
	}
	// raw tracepoints: the classic sched_process_fork lacks the child's tgid
	tp["sched_process_fork"] = ebpfcommon.ProbeDesc{Required: false, Start: p.bpfObjects.ObiRawTpAgentFork}
	tp["sched_process_exit"] = ebpfcommon.ProbeDesc{Required: false, Start: p.bpfObjects.ObiRawTpAgentExit}
	if ka.Exec {
		tp["sched_process_exec"] = ebpfcommon.ProbeDesc{Required: false, Start: p.bpfObjects.ObiRawTpAgentExec}
	}
}

// agentRoots arms processes in the kernel map the activity programs read.
type agentRoots struct {
	roots *ebpf.Map
}

func (r agentRoots) Arm(pid uint32) error {
	return r.roots.Update(pid, uint8(1), ebpf.UpdateAny)
}

func (r agentRoots) Disarm(pid uint32) {
	_ = r.roots.Delete(pid)
}

// forwardAgentActivity sends the kernel activity events to the queue the
// inferred tool calls read, and lets them arm the processes that run tool
// calls.
func (p *Tracer) forwardAgentActivity(ctx context.Context) {
	ka := p.agentActivity()
	if ka == nil || p.eventCtx == nil || p.eventCtx.AgentActivity == nil || p.bpfObjects.AgentActivityEvents == nil {
		return
	}

	how := "kprobe"
	if openFexitSupported() {
		how = "fexit"
	}
	p.log.Info("observing kernel activity for inferred tool calls", "file_open", how, "probation", ka.Probation)
	if p.eventCtx.AgentRoots != nil {
		if failed := p.eventCtx.AgentRoots.Attach(agentRoots{roots: p.bpfObjects.AgentRoots}); len(failed) > 0 {
			p.log.Warn("could not arm the kernel activity of agent processes", "pids", failed)
		}
	}

	paths := agentact.NewPathMatcher(ka.PathExclude)
	go p.reportAgentActivityDrops(ctx)
	go ebpfcommon.ForwardRingbuf(
		&p.cfg.EBPF,
		p.bpfObjects.AgentActivityEvents,
		func(record *ringbuf.Record) (agentact.Event, bool, error) {
			e, ignore, err := parseAgentActivityRecord(record)
			// excluded opens are dropped before they cost the pipeline anything
			return e, ignore || (e.Kind == agentact.KindOpen && paths.Matches(e.Path)), err
		},
		nil,
		p.log,
		p.metrics,
	)(ctx, p.eventCtx.AgentActivity)
}

func (p *Tracer) reportAgentActivityDrops(ctx context.Context) {
	ticker := time.NewTicker(agentActivityDropsReport)
	defer ticker.Stop()

	var reported uint64
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			var perCPU []uint64
			if err := p.bpfObjects.AgentActivityDrops.Lookup(uint32(0), &perCPU); err != nil {
				continue
			}
			var total uint64
			for _, n := range perCPU {
				total += n
			}
			if total > reported {
				p.log.Warn("kernel activity events dropped by the rate limit or a full ring",
					"dropped", total-reported, "total", total)
				reported = total
			}
		}
	}
}

var errShortAgentActivity = errors.New("agent activity record shorter than its header")

// parseAgentActivityRecord decodes a ring record. Records only carry the used
// part of the buffer: none for fork, exit, connect and drops, the path for an
// open, the path area and the argv for an exec.
func parseAgentActivityRecord(record *ringbuf.Record) (agentact.Event, bool, error) {
	if len(record.RawSample) < agentActivityHeaderSize {
		return agentact.Event{}, true, errShortAgentActivity
	}

	var raw BpfAgentActivityEventT
	copy(unsafe.Slice((*byte)(unsafe.Pointer(&raw)), unsafe.Sizeof(raw)), record.RawSample)

	e := agentact.Event{
		Kind:         agentact.Kind(raw.Kind),
		Time:         int64(raw.Ts),
		PID:          raw.Pid,
		TID:          raw.Tid,
		RootPID:      raw.RootPid,
		RootStart:    int64(raw.RootStart),
		ParentPID:    raw.ParentPid,
		SpawnerTID:   raw.SpawnerTid,
		Truncated:    raw.Flags&agentActivityTruncated != 0,
		PathRelative: raw.Flags&agentActivityRelativePath != 0,
	}

	switch e.Kind {
	case agentact.KindExec:
		e.Path = bufString(raw.Buf[:min(int(raw.PathLen), agentActivityPathLen)])
		e.Args = argv(raw.Buf[agentActivityPathLen : agentActivityPathLen+min(int(raw.ArgsLen), len(raw.Buf)-agentActivityPathLen)])
	case agentact.KindOpen:
		e.Path = bufString(raw.Buf[:min(int(raw.PathLen), agentActivityPathLen)])
		e.Result = raw.Ret
		e.OpenFlags = raw.OpenFlags
	case agentact.KindExit:
		// a wait status: the exit code in the second byte, the signal in the
		// low bits, reported the way shells do
		e.ExitCode = int(raw.Ret>>8) & 0xff
		if e.Signal = int(raw.Ret) & 0x7f; e.Signal != 0 {
			e.ExitCode = signalExitBase + e.Signal
		}
	case agentact.KindConnect:
		e.Local = netip.AddrPortFrom(netip.AddrFrom16(raw.Conn.S_addr).Unmap(), raw.Conn.S_port)
		e.Remote = netip.AddrPortFrom(netip.AddrFrom16(raw.Conn.D_addr).Unmap(), raw.Conn.D_port)
	case agentact.KindDrops:
		e.Dropped = int(raw.Ret)
		e.DroppedSince = int64(raw.DroppedSince)
	}

	return e, false, nil
}

func bufString(b []byte) string {
	if i := bytes.IndexByte(b, 0); i >= 0 {
		b = b[:i]
	}
	return string(b)
}

// argv splits the NUL-separated arguments copied from the process's memory.
func argv(b []byte) []string {
	b = bytes.TrimRight(b, "\x00")
	if len(b) == 0 {
		return nil
	}
	parts := bytes.Split(b, []byte{0})
	args := make([]string, len(parts))
	for i, part := range parts {
		args[i] = string(part)
	}
	return args
}

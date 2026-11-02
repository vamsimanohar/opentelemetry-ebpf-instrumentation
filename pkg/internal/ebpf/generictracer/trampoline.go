// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//go:build linux

package generictracer // import "go.opentelemetry.io/obi/pkg/internal/ebpf/generictracer"

import (
	"runtime"
	"strings"
	"sync"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/btf"

	ebpfcommon "go.opentelemetry.io/obi/pkg/ebpf/common"
	"go.opentelemetry.io/obi/pkg/internal/ebpf/kprobe"
)

// trampolineTwin is a kprobe or kretprobe whose body also runs from a BPF
// trampoline program, fentry or fexit. These probes run for every process on
// the host, instrumented or not, and a trampoline dispatches them at a fraction
// of the cost of a kprobe, and of a kretprobe even more so. One of the two is
// attached; the kprobe stays the fallback. Only functions without another
// kprobe of ours have twins: with both, ftrace takes a shared path that costs
// more than the kprobe alone (tcp_recvmsg, whose return tail calls, stays a
// kprobe pair).
type trampolineTwin struct {
	// the kernel function as KProbes() names it: syscalls without their
	// architecture prefix
	kfunc string
	// the trampoline program, and the kprobe it replaces
	name       string
	trampoline *ebpf.Program
	kprobe     *ebpf.Program
	exit       bool
	required   bool
}

// the programs are only set after the collection is loaded
func (p *Tracer) trampolineTwins() []trampolineTwin {
	o := &p.bpfObjects
	return []trampolineTwin{
		{"security_socket_accept", BpfProgObiFentrySecuritySocketAccept, o.ObiFentrySecuritySocketAccept, o.ObiKprobeSecuritySocketAccept, false, true},
		{"sys_accept4", BpfProgObiFexitSysAccept4, o.ObiFexitSysAccept4, o.ObiKretprobeSysAccept4, true, true},
		{"sys_accept", BpfProgObiFexitSysAccept, o.ObiFexitSysAccept, o.ObiKretprobeSysAccept4, true, true},
		{"sys_connect", BpfProgObiFentrySysConnect, o.ObiFentrySysConnect, o.ObiKprobeSysConnect, false, true},
		{"sys_connect", BpfProgObiFexitSysConnect, o.ObiFexitSysConnect, o.ObiKretprobeSysConnect, true, true},
		{"tcp_connect", BpfProgObiFentryTcpConnect, o.ObiFentryTcpConnect, o.ObiKprobeTcpConnect, false, true},
		{"tcp_close", BpfProgObiFentryTcpClose, o.ObiFentryTcpClose, o.ObiKprobeTcpClose, false, true},
		{"sock_def_error_report", BpfProgObiFentrySockDefErrorReport, o.ObiFentrySockDefErrorReport, o.ObiKprobeSockDefErrorReport, false, true},
		{"sys_clone", BpfProgObiFexitSysClone, o.ObiFexitSysClone, o.ObiKretprobeSysClone, true, true},
		{"sys_clone3", BpfProgObiFexitSysClone3, o.ObiFexitSysClone3, o.ObiKretprobeSysClone, true, false},
		{"sys_exit", BpfProgObiFentrySysExit, o.ObiFentrySysExit, o.ObiKprobeSysExit, false, true},
	}
}

// trampolinesSupported tells whether the kernel runs BPF trampolines: x86-64
// since 5.5, which OBI's minimum predates, and arm64 since 6.0.
var trampolinesSupported = sync.OnceValue(func() bool {
	if runtime.GOARCH == "arm64" {
		if major, _ := ebpfcommon.KernelVersion(); major < arm64TrampolineMajor {
			return false
		}
	}
	return true
})

// trampolineTarget is the function a trampoline twin attaches to, false when
// the kernel BTF has no such function.
func trampolineTarget(kfunc string) (string, bool) {
	target := kfunc
	if strings.HasPrefix(kfunc, "sys_") {
		target = kprobe.SyscallPrefix() + kfunc
	}

	spec, err := btf.LoadKernelSpec()
	if err != nil {
		return "", false
	}
	var fn *btf.Func
	if err := spec.TypeByName(target, &fn); err != nil {
		return "", false
	}

	return target, true
}

// prepareTrampolines points each trampoline twin at its kernel function, or
// stubs it where the kernel cannot run it, so that its kprobe is used instead.
func (p *Tracer) prepareTrampolines(spec *ebpf.CollectionSpec) {
	p.trampolined = map[string]bool{}
	for _, twin := range p.trampolineTwins() {
		prog, ok := spec.Programs[twin.name]
		if !ok {
			continue
		}

		target, ok := "", false
		if trampolinesSupported() {
			target, ok = trampolineTarget(twin.kfunc)
		}
		if !ok {
			stubProgram(prog)
			continue
		}

		prog.AttachTo = target
		p.trampolined[twin.name] = true
	}
}

// withoutTrampolined leaves out the kprobe halves a trampoline replaces
func (p *Tracer) withoutTrampolined(kp map[string]ebpfcommon.ProbeDesc) {
	for _, twin := range p.trampolineTwins() {
		desc, ok := kp[twin.kfunc]
		if !ok || !p.trampolined[twin.name] {
			continue
		}
		if twin.exit {
			desc.End = nil
		} else {
			desc.Start = nil
		}
		if desc.Start == nil && desc.End == nil {
			delete(kp, twin.kfunc)
			continue
		}
		kp[twin.kfunc] = desc
	}
}

func (p *Tracer) trampolineTracing() []*ebpfcommon.Tracing {
	var tracing []*ebpfcommon.Tracing
	for _, twin := range p.trampolineTwins() {
		if !p.trampolined[twin.name] {
			continue
		}
		attachAs := ebpf.AttachTraceFEntry
		if twin.exit {
			attachAs = ebpf.AttachTraceFExit
		}
		tracing = append(tracing, &ebpfcommon.Tracing{
			Program:  twin.trampoline,
			AttachAs: attachAs,
			Required: twin.required,
			Fallback: &ebpfcommon.KprobeFallback{Symbol: twin.kfunc, Program: twin.kprobe, Return: twin.exit},
		})
	}
	return tracing
}

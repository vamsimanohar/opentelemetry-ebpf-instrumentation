// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//go:build linux

package generictracer

import (
	"testing"

	"github.com/cilium/ebpf"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"go.opentelemetry.io/obi/pkg/obi"
)

// Each trampoline twin is either pointed at a function of the running kernel
// or stubbed, so that loading the collection never depends on the kernel.
func TestPrepareTrampolinesTargetsKernelFunctionsOrStubs(t *testing.T) {
	spec, err := LoadBpf()
	require.NoError(t, err)

	cfg := obi.DefaultConfig
	tracer := &Tracer{cfg: &cfg}
	tracer.prepareTrampolines(spec)

	for _, twin := range tracer.trampolineTwins() {
		prog := spec.Programs[twin.name]
		require.NotNil(t, prog, twin.name)
		if !tracer.trampolined[twin.name] {
			assert.Equal(t, ebpf.Kprobe, prog.Type, "%s is stubbed", twin.name)
			continue
		}
		target, ok := trampolineTarget(twin.kfunc)
		require.True(t, ok, twin.name)
		assert.Equal(t, target, prog.AttachTo, twin.name)
		assert.NotContains(t, prog.AttachTo, "/", twin.name)
	}
}

// A twin in use replaces only its own half of a kprobe pair, and is attached
// with that half as its fallback.
func TestTrampolineTwinsReplaceTheirKprobeHalf(t *testing.T) {
	cfg := obi.DefaultConfig
	tracer := &Tracer{cfg: &cfg}
	o := &tracer.bpfObjects
	o.ObiKprobeSysConnect, o.ObiKretprobeSysConnect = &ebpf.Program{}, &ebpf.Program{}
	o.ObiKprobeTcpClose = &ebpf.Program{}
	o.ObiKretprobeSysAccept4 = &ebpf.Program{}

	before := tracer.KProbes()
	require.NotNil(t, before["sys_connect"].Start)
	require.NotNil(t, before["sys_connect"].End)

	tracer.trampolined = map[string]bool{
		BpfProgObiFentrySysConnect: true,
		BpfProgObiFentryTcpClose:   true,
		BpfProgObiFexitSysAccept4:  true,
	}
	after := tracer.KProbes()

	assert.Nil(t, after["sys_connect"].Start, "the fentry twin replaces the entry kprobe")
	assert.Equal(t, before["sys_connect"].End, after["sys_connect"].End, "the return kprobe stays")
	assert.NotContains(t, after, "tcp_close", "the only half is replaced")
	assert.NotContains(t, after, "sys_accept4")
	assert.Contains(t, after, "sys_accept", "a twin not in use keeps its kprobe")

	tracing := tracer.trampolineTracing()
	require.Len(t, tracing, len(tracer.trampolined))
	for _, tr := range tracing {
		require.NotNil(t, tr.Fallback)
		assert.True(t, tr.Required)
		assert.Equal(t, tr.AttachAs == ebpf.AttachTraceFExit, tr.Fallback.Return)
	}
}

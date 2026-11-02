// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//go:build linux

package generictracer

import (
	"net/netip"
	"testing"
	"unsafe"

	"github.com/cilium/ebpf"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"go.opentelemetry.io/obi/pkg/appolly/app/agentact"
	"go.opentelemetry.io/obi/pkg/ebpf/ringbuf"
)

// record returns the bytes the kernel would send: the header and the used
// part of the buffer only.
func record(e *BpfAgentActivityEventT, bufLen int) *ringbuf.Record {
	raw := unsafe.Slice((*byte)(unsafe.Pointer(e)), unsafe.Sizeof(*e))
	return &ringbuf.Record{RawSample: append([]byte(nil), raw[:agentActivityHeaderSize+bufLen]...)}
}

func TestParseAgentActivityRecord(t *testing.T) {
	t.Run("exec", func(t *testing.T) {
		e := BpfAgentActivityEventT{Kind: uint8(agentact.KindExec), Ts: 42, Pid: 7, Tid: 7, RootPid: 1, ParentPid: 1}
		path := "/usr/bin/uname"
		copy(e.Buf[:], path)
		e.PathLen = uint16(len(path))
		args := "uname\x00-a\x00"
		copy(e.Buf[agentActivityPathLen:], args)
		e.ArgsLen = uint16(len(args))

		ev, ignore, err := parseAgentActivityRecord(record(&e, len(e.Buf)))
		require.NoError(t, err)
		require.False(t, ignore)
		assert.Equal(t, agentact.KindExec, ev.Kind)
		assert.Equal(t, int64(42), ev.Time)
		assert.Equal(t, path, ev.Path)
		assert.Equal(t, []string{"uname", "-a"}, ev.Args)
		assert.Equal(t, uint32(1), ev.RootPID)
	})

	t.Run("open carries only its path", func(t *testing.T) {
		e := BpfAgentActivityEventT{Kind: uint8(agentact.KindOpen), Ret: -2, OpenFlags: 0o100, Flags: agentActivityRelativePath}
		path := "fixtures/missing.txt"
		copy(e.Buf[:], path)
		e.PathLen = uint16(len(path))

		ev, _, err := parseAgentActivityRecord(record(&e, len(path)))
		require.NoError(t, err)
		assert.Equal(t, path, ev.Path)
		assert.Equal(t, int32(-2), ev.Result)
		assert.True(t, ev.PathRelative)
	})

	t.Run("exit status", func(t *testing.T) {
		for _, tc := range []struct {
			status       int32
			code, signal int
		}{
			{status: 3 << 8, code: 3},
			{status: 9, code: signalExitBase + 9, signal: 9},
		} {
			e := BpfAgentActivityEventT{Kind: uint8(agentact.KindExit), Ret: tc.status}
			ev, _, err := parseAgentActivityRecord(record(&e, 0))
			require.NoError(t, err)
			assert.Equal(t, tc.code, ev.ExitCode)
			assert.Equal(t, tc.signal, ev.Signal)
		}
	})

	t.Run("connect", func(t *testing.T) {
		e := BpfAgentActivityEventT{Kind: uint8(agentact.KindConnect)}
		mapped := func(ip string) [16]uint8 { return netip.MustParseAddr("::ffff:" + ip).As16() }
		e.Conn.S_addr, e.Conn.S_port = mapped("127.0.0.1"), 40000
		e.Conn.D_addr, e.Conn.D_port = mapped("127.0.0.1"), 28090

		ev, _, err := parseAgentActivityRecord(record(&e, 0))
		require.NoError(t, err)
		assert.Equal(t, netip.MustParseAddrPort("127.0.0.1:28090"), ev.Remote)
		assert.Equal(t, netip.MustParseAddrPort("127.0.0.1:40000"), ev.Local)
	})

	t.Run("drops", func(t *testing.T) {
		e := BpfAgentActivityEventT{Kind: uint8(agentact.KindDrops), Ts: 900, DroppedSince: 500, Ret: 40, RootPid: 1, RootStart: 7}
		ev, _, err := parseAgentActivityRecord(record(&e, 0))
		require.NoError(t, err)
		assert.Equal(t, 40, ev.Dropped)
		assert.Equal(t, int64(500), ev.DroppedSince)
		assert.Equal(t, int64(7), ev.RootStart)
	})

	t.Run("short record", func(t *testing.T) {
		_, ignore, err := parseAgentActivityRecord(&ringbuf.Record{RawSample: []byte{1, 2}})
		require.Error(t, err)
		assert.True(t, ignore)
	})
}

func TestDisabledAgentActivityNeedsNoKernelSupport(t *testing.T) {
	spec, err := LoadBpf()
	require.NoError(t, err)
	require.Equal(t, ebpf.Tracing, spec.Programs[BpfProgObiFexitAgentOpenat2].Type)

	disableAgentActivity(spec)

	for _, name := range agentActivityPrograms {
		prog := spec.Programs[name]
		assert.Len(t, prog.Instructions, 2, name)
		assert.NotEqual(t, ebpf.Tracing, prog.Type, "%s needs no attach target", name)
	}
	assert.Equal(t, uint32(disabledRingSize), spec.Maps[BpfMapAgentActivityEvents].MaxEntries)
	for _, name := range agentActivityMaps {
		assert.Equal(t, uint32(1), spec.Maps[name].MaxEntries, name)
	}
}

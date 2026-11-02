// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Package agentact defines the kernel activity events observed for
// instrumented processes and their descendants: process creation, program
// execution, exit, file opens and outgoing connections. They feed the
// inference of agent tool-call spans and never become spans themselves.
package agentact // import "go.opentelemetry.io/obi/pkg/appolly/app/agentact"

import "net/netip"

// Kind mirrors enum agent_activity_kind in bpf/common/agent_activity.h.
type Kind uint8

const (
	KindFork Kind = iota + 1
	KindExec
	KindExit
	KindOpen
	KindConnect
	// KindDrops reports events of a root lost to the rate limit or a full ring.
	KindDrops
)

func (k Kind) String() string {
	switch k {
	case KindFork:
		return "fork"
	case KindExec:
		return "exec"
	case KindExit:
		return "exit"
	case KindOpen:
		return "open"
	case KindConnect:
		return "connect"
	case KindDrops:
		return "drops"
	default:
		return "unknown"
	}
}

// Event is one kernel activity record. Process and thread ids are host ids.
type Event struct {
	Kind Kind
	// Time is a CLOCK_MONOTONIC timestamp, the clock of span Start and End.
	Time int64
	PID  uint32
	TID  uint32
	// RootPID is the instrumented process the activity is reported for: the
	// process itself when it is instrumented, else its nearest instrumented
	// ancestor. RootStart is the CLOCK_MONOTONIC start time of that process.
	RootPID   uint32
	RootStart int64
	// ParentPID is the process that forked PID. For KindFork, PID is the new
	// child and SpawnerTID the thread that forked it.
	ParentPID  uint32
	SpawnerTID uint32

	// Path is the executable of KindExec and the opened file of KindOpen.
	Path string
	// Args is the argv of KindExec. Truncated tells that the path or the argv
	// did not fit the event.
	Args      []string
	Truncated bool
	// PathRelative marks a KindOpen path that is relative to the working
	// directory or to a directory fd.
	PathRelative bool

	// Result is the fd or the negated errno of KindOpen.
	Result    int32
	OpenFlags uint32
	// ExitCode and Signal decode the wait status of KindExit.
	ExitCode int
	Signal   int

	// Local and Remote are the endpoints of KindConnect.
	Local  netip.AddrPort
	Remote netip.AddrPort

	// Dropped and DroppedSince describe KindDrops: how many events of the
	// root were lost, from DroppedSince to Time.
	Dropped      int
	DroppedSince int64
}

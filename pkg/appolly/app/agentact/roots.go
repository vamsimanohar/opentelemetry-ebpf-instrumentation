// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package agentact // import "go.opentelemetry.io/obi/pkg/appolly/app/agentact"

import "sync"

// Roots selects the instrumented processes whose kernel activity is reported.
// An armed process and its descendants report their execs, file opens and
// connects. Other instrumented processes only report their process lineage,
// except during a probation period after they start.
type Roots interface {
	// Arm reports the activity of the process with this host pid.
	Arm(pid uint32) error
	// Disarm stops reporting it.
	Disarm(pid uint32)
}

// RootRegistry connects the stage that learns which processes run tool calls
// with the tracer that reports their activity. The tracer loads after the
// stage starts, so arming requests made before are replayed when it attaches.
type RootRegistry struct {
	mu    sync.Mutex
	roots Roots
	armed map[uint32]struct{}
}

func NewRootRegistry() *RootRegistry {
	return &RootRegistry{armed: map[uint32]struct{}{}}
}

// Attach hands the registry the tracer's implementation and arms the
// processes armed so far. It returns the pids that could not be armed.
func (r *RootRegistry) Attach(roots Roots) []uint32 {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.roots = roots
	var failed []uint32
	for pid := range r.armed {
		if roots.Arm(pid) != nil {
			delete(r.armed, pid)
			failed = append(failed, pid)
		}
	}
	return failed
}

// Arm arms a process. It reports whether the process is known to be armed:
// false when the tracer refused it.
func (r *RootRegistry) Arm(pid uint32) bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, ok := r.armed[pid]; ok {
		return true
	}
	if r.roots != nil && r.roots.Arm(pid) != nil {
		return false
	}
	r.armed[pid] = struct{}{}
	return true
}

// Disarm disarms a process.
func (r *RootRegistry) Disarm(pid uint32) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, ok := r.armed[pid]; !ok {
		return
	}
	delete(r.armed, pid)
	if r.roots != nil {
		r.roots.Disarm(pid)
	}
}

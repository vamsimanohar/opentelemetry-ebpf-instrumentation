// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package agentact

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
)

type mapRoots struct {
	armed map[uint32]bool
	full  bool
}

func (m *mapRoots) Arm(pid uint32) error {
	if m.full {
		return errors.New("map full")
	}
	m.armed[pid] = true
	return nil
}

func (m *mapRoots) Disarm(pid uint32) {
	delete(m.armed, pid)
}

func TestRootRegistryReplaysArmsWhenTheTracerAttaches(t *testing.T) {
	r := NewRootRegistry()
	assert.True(t, r.Arm(10), "armed before the tracer loads")
	assert.True(t, r.Arm(11))
	r.Disarm(11)

	roots := &mapRoots{armed: map[uint32]bool{}}
	assert.Empty(t, r.Attach(roots))
	assert.Equal(t, map[uint32]bool{10: true}, roots.armed)

	assert.True(t, r.Arm(12))
	r.Disarm(10)
	assert.Equal(t, map[uint32]bool{12: true}, roots.armed)
}

func TestRootRegistryReportsRefusedArms(t *testing.T) {
	r := NewRootRegistry()
	r.Arm(10)
	roots := &mapRoots{armed: map[uint32]bool{}, full: true}
	assert.Equal(t, []uint32{10}, r.Attach(roots))
	assert.False(t, r.Arm(11))
}

// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//go:build linux

package ebpf

import (
	"errors"
	"io"
	"log/slog"
	"os"
	"sync"
	"testing"
	"time"

	cebpf "github.com/cilium/ebpf"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"go.opentelemetry.io/obi/pkg/appolly/app"
	"go.opentelemetry.io/obi/pkg/appolly/discover/exec"
	ebpfcommon "go.opentelemetry.io/obi/pkg/ebpf/common"
	"go.opentelemetry.io/obi/pkg/export/imetrics"
	"go.opentelemetry.io/obi/pkg/obi"
)

type closeTrackingTracer struct {
	stubTracer
	closes int
}

func (t *closeTrackingTracer) Close() error {
	t.closes++
	return nil
}

type initProbeTracer struct {
	closeTrackingTracer
	probes map[string]ebpfcommon.ProbeDesc
}

func (t *initProbeTracer) KProbes() map[string]ebpfcommon.ProbeDesc {
	return t.probes
}

func (t *initProbeTracer) Required() bool { return true }

func TestCloseInstrumentersReleasesEveryExecutableOnce(t *testing.T) {
	module := &countingCloser{}
	shared := &instrumenter{
		closables:      []io.Closer{&countingCloser{}, &countingCloser{}},
		processModules: map[app.PID]map[uint64][]io.Closer{42: {7: {module}}},
	}
	single := &instrumenter{
		closables: []io.Closer{&countingCloser{}},
	}
	pt := &ProcessTracer{
		log:      slog.Default(),
		Programs: []Tracer{&stubTracer{}},
		Instrumentables: map[ExecutableKey]*instrumenter{
			{Dev: 1, Ino: 1}: shared,
			{Dev: 1, Ino: 2}: shared,
			{Dev: 1, Ino: 3}: single,
		},
		instrumentableGenerations: map[ExecutableKey]uint64{
			{Dev: 1, Ino: 1}: 1,
			{Dev: 1, Ino: 2}: 2,
			{Dev: 1, Ino: 3}: 3,
		},
	}

	pt.closeInstrumenters()

	for _, i := range []*instrumenter{shared, single} {
		for _, c := range i.closables {
			assert.Equal(t, int32(1), c.(*countingCloser).closes.Load())
		}
	}
	assert.Equal(t, int32(1), module.closes.Load())
	assert.Empty(t, pt.Instrumentables)
	assert.Empty(t, pt.instrumentableGenerations)
}

func TestCloseInstrumentersWithoutExecutables(t *testing.T) {
	pt := &ProcessTracer{log: slog.Default()}

	pt.closeInstrumenters()

	assert.Empty(t, pt.Instrumentables)
}

func TestCloseReleasesProgramsBeforeRun(t *testing.T) {
	program := &closeTrackingTracer{}
	pt := &ProcessTracer{log: slog.Default(), Programs: []Tracer{program}}

	require.NoError(t, pt.Close())
	assert.Equal(t, 1, program.closes)
	require.ErrorIs(t, pt.NewExecutable(nil, nil), errTracerStopped)

	require.NoError(t, pt.Close())
	assert.Equal(t, 1, program.closes)
}

func TestInitClosesPartialInitProbesOnRequiredProbeFailure(t *testing.T) {
	originalKprobe := attachKprobe
	originalKretprobe := attachKretprobe
	t.Cleanup(func() {
		attachKprobe = originalKprobe
		attachKretprobe = originalKretprobe
	})

	attached := &countingCloser{}
	attachKprobe = func(string, *cebpf.Program) (io.Closer, error) {
		return attached, nil
	}
	attachKretprobe = func(string, *cebpf.Program) (io.Closer, error) {
		return nil, errors.New("required probe failed")
	}

	program := &initProbeTracer{
		probes: map[string]ebpfcommon.ProbeDesc{
			"required_probe": {
				Start:    &cebpf.Program{},
				End:      &cebpf.Program{},
				Required: true,
			},
		},
	}
	unattemptedProgram := &closeTrackingTracer{}
	cfg := &obi.Config{}
	cfg.EBPF.BPFFSPath = t.TempDir()
	pt := NewProcessTracer(Generic, []Tracer{program, unattemptedProgram}, cfg, imetrics.NoopReporter{})

	err := pt.Init(&ebpfcommon.EBPFEventContext{}, cfg)

	require.Error(t, err)
	require.NoError(t, pt.Close())
	assert.Equal(t, int32(1), attached.closes.Load())
	assert.Equal(t, 1, program.closes)
	assert.Equal(t, 1, unattemptedProgram.closes)
}

// an executable that fails to attach is never committed, so its probes and the
// module probes of its processes are only ever released here
func TestUnlinkInstrumenterReleasesProbesAndModules(t *testing.T) {
	baseline := &countingCloser{}
	group := &reverseCloser{closers: []io.Closer{&countingCloser{}}}
	module := &countingCloser{}
	pt := &ProcessTracer{log: slog.Default(), Programs: []Tracer{&stubTracer{}}}
	i := &instrumenter{
		closables:      []io.Closer{baseline, group},
		processModules: map[app.PID]map[uint64][]io.Closer{42: {11: {module}}},
	}

	pt.unlinkInstrumenter(i)

	assert.Equal(t, int32(1), baseline.closes.Load())
	assert.Equal(t, int32(1), group.closers[0].(*countingCloser).closes.Load())
	assert.Equal(t, int32(1), module.closes.Load())
	assert.Empty(t, i.processModules)
}

// blocks every close until all of them are running at once
type gatedCloser struct {
	entered *sync.WaitGroup
	release chan struct{}
}

func (c gatedCloser) Close() error {
	c.entered.Done()
	<-c.release
	return nil
}

// each released probe waits for kernel grace periods, so the module probes of an
// executable's processes are released in parallel with each other and with its own probes
func TestUnlinkInstrumenterReleasesModulesInParallel(t *testing.T) {
	var entered sync.WaitGroup
	release := make(chan struct{})
	gated := gatedCloser{entered: &entered, release: release}
	i := &instrumenter{
		closables: []io.Closer{gated},
		processModules: map[app.PID]map[uint64][]io.Closer{
			1: {1: {gated}, 2: {gated}},
			2: {1: {gated}},
		},
	}
	entered.Add(4)
	pt := &ProcessTracer{log: slog.Default(), Programs: []Tracer{&stubTracer{}}}

	unlinked := make(chan struct{})
	go func() {
		pt.unlinkInstrumenter(i)
		close(unlinked)
	}()

	allEntered := make(chan struct{})
	go func() {
		entered.Wait()
		close(allEntered)
	}()
	select {
	case <-allEntered:
	case <-time.After(5 * time.Second):
		close(release)
		require.Fail(t, "the module probes of one executable were released one at a time")
	}

	close(release)
	<-unlinked
}

// a process that ends while others run its executable releases only its own module probes
func TestUnlinkProcessesReleasesOnlyTheirModules(t *testing.T) {
	ended, child, other, executable := &countingCloser{}, &countingCloser{}, &countingCloser{}, &countingCloser{}
	key := ExecutableKey{Dev: 1, Ino: 1}
	i := &instrumenter{
		key:       key,
		closables: []io.Closer{executable},
		processModules: map[app.PID]map[uint64][]io.Closer{
			10: {7: {ended}},
			11: {7: {child}},
			20: {7: {other}},
		},
	}
	pt := &ProcessTracer{
		log:             slog.Default(),
		Programs:        []Tracer{&stubTracer{}},
		Instrumentables: map[ExecutableKey]*instrumenter{key: i},
	}

	pt.UnlinkProcesses(exec.New(exec.Init{Dev: 1, Ino: 1}), []app.PID{10, 11})
	pt.instanceReleases.Wait()

	assert.Equal(t, int32(1), ended.closes.Load())
	assert.Equal(t, int32(1), child.closes.Load())
	assert.Equal(t, int32(0), other.closes.Load())
	assert.Equal(t, int32(0), executable.closes.Load())
	assert.Equal(t, map[app.PID]map[uint64][]io.Closer{20: {7: {other}}}, i.processModules)
}

// shutdown may not clear the committed set while an attachment is between its
// probe setup and its commit, or those probes are left to the kernel
func TestCloseInstrumentersWaitsForInFlightAttachment(t *testing.T) {
	pt := &ProcessTracer{
		log:                       slog.Default(),
		Instrumentables:           map[ExecutableKey]*instrumenter{},
		instrumentableGenerations: map[ExecutableKey]uint64{},
	}
	attached := &countingCloser{}
	holdsLock := make(chan struct{})
	commitDone := make(chan struct{})

	// mirrors NewExecutable: the mutex is held from before the probes are
	// attached until after the instrumenter is committed
	go func() {
		defer close(commitDone)
		pt.instrumentablesMu.Lock()
		defer pt.instrumentablesMu.Unlock()
		close(holdsLock)
		time.Sleep(20 * time.Millisecond)
		pt.commitInstrumenter(&instrumenter{
			key:       ExecutableKey{Dev: 1, Ino: 1},
			closables: []io.Closer{attached},
		}, &Instrumentable{})
	}()

	<-holdsLock
	pt.closeInstrumenters()
	<-commitDone

	assert.Equal(t, int32(1), attached.closes.Load())
	assert.Empty(t, pt.Instrumentables)
}

func TestNewExecutableRejectedAfterShutdown(t *testing.T) {
	pt := &ProcessTracer{log: slog.Default(), Instrumentables: map[ExecutableKey]*instrumenter{}}

	pt.closeInstrumenters()

	// nothing is dereferenced because no probe is attached after shutdown
	require.ErrorIs(t, pt.NewExecutable(nil, &Instrumentable{}), errTracerStopped)
	require.ErrorIs(t, pt.NewExecutableInstance(&Instrumentable{
		FileInfo: exec.New(exec.Init{Dev: 1, Ino: 2}),
	}), errTracerStopped)
	assert.Empty(t, pt.Instrumentables)
}

// the module uprobes of a later process are attached in the background, for that process
func TestInstanceUprobesAttachInTheBackground(t *testing.T) {
	pt, ie := instanceTracer(t)

	pt.instrumentablesMu.Lock()
	pt.attachInstanceUprobes(pt.Instrumentables[ie.key], ie.instrumentable, makeProcMaps("/irrelevant"))
	pt.instrumentablesMu.Unlock()
	pt.instanceAttaches.Wait()

	assert.Len(t, pt.Instrumentables[ie.key].processModules[ie.pid], 1)
	assert.Empty(t, pt.pendingUprobes)
}

// an attachment in progress does not hold up the next process of the executable
func TestInstanceUprobesDoNotBlockTheNextProcess(t *testing.T) {
	pt, ie := instanceTracer(t)
	i := pt.Instrumentables[ie.key]

	i.attachMu.Lock() // an attachment in progress
	done := make(chan struct{})
	go func() {
		pt.instrumentablesMu.Lock()
		pt.attachInstanceUprobes(i, ie.instrumentable, makeProcMaps("/irrelevant"))
		pt.instrumentablesMu.Unlock()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		require.Fail(t, "queuing a process waited for another process's attachment")
	}
	i.attachMu.Unlock()
	pt.instanceAttaches.Wait()

	assert.Len(t, i.processModules[ie.pid], 1)
}

// a released executable attaches no more process uprobes
func TestInstanceUprobesSkipAReleasedExecutable(t *testing.T) {
	pt, ie := instanceTracer(t)
	i := pt.Instrumentables[ie.key]

	i.attachMu.Lock() // the attachment waits its turn
	pt.instrumentablesMu.Lock()
	pt.attachInstanceUprobes(i, ie.instrumentable, makeProcMaps("/irrelevant"))
	pt.instrumentablesMu.Unlock()
	i.closed = true // what unlinkInstrumenter does
	i.attachMu.Unlock()
	pt.instanceAttaches.Wait()

	assert.Empty(t, i.processModules)
	assert.Empty(t, pt.pendingUprobes)
}

// a process that ends before its turn, whose pid may then name another
// process, gets no uprobes
func TestInstanceUprobesSkipAProcessThatEnded(t *testing.T) {
	pt, ie := instanceTracer(t)

	i := pt.Instrumentables[ie.key]
	i.attachMu.Lock() // the attachment waits its turn
	pt.instrumentablesMu.Lock()
	pt.attachInstanceUprobes(i, ie.instrumentable, makeProcMaps("/irrelevant"))
	pt.instrumentablesMu.Unlock()
	pt.takeUprobesPending(ie.pid) // what UnlinkProcesses does when the process ends
	i.attachMu.Unlock()
	pt.instanceAttaches.Wait()

	assert.Empty(t, pt.Instrumentables[ie.key].processModules[ie.pid])
	assert.Empty(t, pt.pendingUprobes)
}

type instanceCase struct {
	key            ExecutableKey
	pid            app.PID
	instrumentable *Instrumentable
}

// a tracer with one committed executable and a later process of it: this test
// process, with a uprobe library it does not map, which resolves to its own
// executable
func instanceTracer(t *testing.T) (*ProcessTracer, instanceCase) {
	t.Helper()
	pid := app.PID(os.Getpid())
	key := ExecutableKey{Dev: 1, Ino: 1}
	tracer := &stubTracer{uprobes: map[string]map[string][]*ebpfcommon.ProbeDesc{
		"missing-library": {"missing_symbol": {{}}},
	}}
	pt := &ProcessTracer{
		log:             slog.Default(),
		Programs:        []Tracer{tracer},
		Instrumentables: map[ExecutableKey]*instrumenter{key: {key: key}},
	}
	ie := &Instrumentable{FileInfo: exec.New(exec.Init{Dev: key.Dev, Ino: key.Ino, Pid: pid})}
	return pt, instanceCase{key: key, pid: pid, instrumentable: ie}
}

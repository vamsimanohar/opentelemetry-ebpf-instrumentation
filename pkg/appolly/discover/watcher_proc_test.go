// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package discover

import (
	"bytes"
	"context"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"go.opentelemetry.io/obi/pkg/appolly/app"
	"go.opentelemetry.io/obi/pkg/appolly/services"
	ebpfcommon "go.opentelemetry.io/obi/pkg/ebpf/common"
	"go.opentelemetry.io/obi/pkg/internal/ebpf/watcher"
	"go.opentelemetry.io/obi/pkg/internal/testutil"
	"go.opentelemetry.io/obi/pkg/obi"
	"go.opentelemetry.io/obi/pkg/pipe/msg"
)

const testTimeout = 5 * time.Second

func TestWatcher_Poll(t *testing.T) {
	// mocking a fake listProcesses method
	p1_1 := ProcessAttrs{pid: 1, openPorts: []uint32{3030}}
	p1_2 := ProcessAttrs{pid: 1, openPorts: []uint32{3030, 3031}}
	p2 := ProcessAttrs{pid: 2, openPorts: []uint32{123}}
	p3 := ProcessAttrs{pid: 3, openPorts: []uint32{456}}
	p4 := ProcessAttrs{pid: 4, openPorts: []uint32{789}}
	p5 := ProcessAttrs{pid: 10}
	invocation := 0
	ctx, cancel := context.WithCancel(t.Context())
	// GIVEN a pollAccounter
	acc := pollAccounter{
		interval: time.Microsecond,
		cfg:      &obi.Config{},
		pidPorts: map[pidPort]ProcessAttrs{},
		listProcesses: func(bool) (map[app.PID]ProcessAttrs, error) {
			invocation++
			switch invocation {
			case 1:
				return map[app.PID]ProcessAttrs{p1_1.pid: p1_1, p2.pid: p2, p3.pid: p3}, nil
			case 2:
				// p1_2 simulates that a new connection has been created for an existing process
				return map[app.PID]ProcessAttrs{p1_2.pid: p1_2, p3.pid: p3, p4.pid: p4}, nil
			case 3:
				return map[app.PID]ProcessAttrs{p2.pid: p2, p3.pid: p3, p4.pid: p4}, nil
			default:
				// new processes with no connections (p5) should be also reported
				return map[app.PID]ProcessAttrs{p5.pid: p5, p2.pid: p2, p3.pid: p3, p4.pid: p4}, nil
			}
		},
		executableReady: func(app.PID) (string, bool) {
			return "", true
		},
		loadBPFWatcher: func(context.Context, *ebpfcommon.EBPFEventContext, *obi.Config, chan<- watcher.Event) error {
			return nil
		},
		loadBPFLogger: func(context.Context, *ebpfcommon.EBPFEventContext, *obi.Config) error {
			return nil
		},
		output: msg.NewQueue[[]Event[ProcessAttrs]](msg.ChannelBufferLen(1)),
	}
	accounterOutput := acc.output.Subscribe()
	accounterExited := make(chan struct{})
	go func() {
		acc.run(ctx)
		close(accounterExited)
	}()

	// WHEN it polls the process for the first time
	// THEN it returns the creation of all the events
	out := testutil.ReadChannel(t, accounterOutput, testTimeout)
	assert.Equal(t, []Event[ProcessAttrs]{
		{Type: EventCreated, Obj: p1_1},
		{Type: EventCreated, Obj: p2},
		{Type: EventCreated, Obj: p3},
	}, sort(out))

	// WHEN it polls the process for the successive times
	// THEN it returns the creation of the new processes/connections
	// AND the deletion of the old processes
	out = testutil.ReadChannel(t, accounterOutput, testTimeout)
	assert.Equal(t, []Event[ProcessAttrs]{
		{Type: EventCreated, Obj: p1_2},
		{Type: EventDeleted, Obj: p2},
		{Type: EventCreated, Obj: p4},
	}, sort(out))
	out = testutil.ReadChannel(t, accounterOutput, testTimeout)
	assert.Equal(t, []Event[ProcessAttrs]{
		{Type: EventDeleted, Obj: p1_2},
		{Type: EventCreated, Obj: p2},
	}, sort(out))

	// WHEN a new process with no connections is created
	// THEN it should be also reported
	// (use case: we want to later match by executable path a client process with short-lived connections)
	out = testutil.ReadChannel(t, accounterOutput, testTimeout)
	assert.Equal(t, []Event[ProcessAttrs]{
		{Type: EventCreated, Obj: p5},
	}, sort(out))

	// WHEN no changes in the process, it doesn't send anything
	select {
	case procs := <-accounterOutput:
		assert.Failf(t, "no output expected", "got %v", procs)
	default:
		// ok!
	}

	// WHEN its context is cancelled
	cancel()
	// THEN the main loop exits
	select {
	case <-accounterExited:
	// ok!
	case <-time.After(testTimeout):
		assert.Fail(t, "expected to exit the main loop")
	}
}

func TestProcessNotReady(t *testing.T) {
	// mocking a fake listProcesses method
	p1 := ProcessAttrs{pid: 1, openPorts: []uint32{3030, 3031}}
	p2 := ProcessAttrs{pid: 2, openPorts: []uint32{123}}
	p3 := ProcessAttrs{pid: 3, openPorts: []uint32{456}}
	p4 := ProcessAttrs{pid: 4, openPorts: []uint32{789}}
	p5 := ProcessAttrs{pid: 10}

	acc := pollAccounter{
		interval: time.Microsecond,
		cfg:      &obi.Config{},
		pidPorts: map[pidPort]ProcessAttrs{},
		listProcesses: func(bool) (map[app.PID]ProcessAttrs, error) {
			return map[app.PID]ProcessAttrs{p1.pid: p1, p5.pid: p5, p2.pid: p2, p3.pid: p3, p4.pid: p4}, nil
		},
		executableReady: func(pid app.PID) (string, bool) {
			return "", pid >= 3
		},
		loadBPFWatcher: func(context.Context, *ebpfcommon.EBPFEventContext, *obi.Config, chan<- watcher.Event) error {
			return nil
		},
		loadBPFLogger: func(context.Context, *ebpfcommon.EBPFEventContext, *obi.Config) error {
			return nil
		},
	}

	procs, err := acc.listProcesses(true)
	require.NoError(t, err)
	assert.Len(t, procs, 5)
	events := acc.snapshot(procs)
	assert.Len(t, events, 3)       // 2 are not ready
	assert.Len(t, acc.pids, 3)     // this should equal the first invocation of snapshot
	assert.Len(t, acc.pidPorts, 2) // only 2 ports opened, p5 has no ports

	eventsNext := acc.snapshot(procs)
	assert.Empty(t, eventsNext) // 0 new events
	assert.Len(t, acc.pids, 3)  // this should equal the first invocation of snapshot, no changes

	acc.executableReady = func(pid app.PID) (string, bool) { // we change so that pid=1 becomes ready
		return "", pid != 2
	}

	eventsNextNext := acc.snapshot(procs)
	assert.Len(t, eventsNextNext, 1) // 1 net new event
	assert.Len(t, acc.pids, 4)       // this should increase by one since we have one more app.PID we are caching now
	assert.Len(t, acc.pidPorts, 4)   // this is now 4 because pid=1 has 2 port mappings
}

func TestPortsFetchRequired(t *testing.T) {
	synctest.Test(t, testPortsFetchRequired)
}

func testPortsFetchRequired(t *testing.T) {
	userConfig := bytes.NewBufferString("channel_buffer_len: 33")
	t.Setenv("OTEL_EBPF_OPEN_PORT", "8080-8089")

	cfg, err := obi.LoadConfig(userConfig)
	require.NoError(t, err)

	channelReturner := make(chan chan<- watcher.Event)

	ctx, cancel := context.WithCancel(t.Context())

	acc := pollAccounter{
		cfg:      cfg,
		interval: time.Hour, // don't let the inner loop mess with our test
		pidPorts: map[pidPort]ProcessAttrs{},
		listProcesses: func(bool) (map[app.PID]ProcessAttrs, error) {
			return nil, nil
		},
		executableReady: func(_ app.PID) (string, bool) {
			return "", true
		},
		loadBPFWatcher: func(_ context.Context, _ *ebpfcommon.EBPFEventContext, _ *obi.Config, events chan<- watcher.Event) error {
			channelReturner <- events
			return nil
		},
		loadBPFLogger: func(context.Context, *ebpfcommon.EBPFEventContext, *obi.Config) error {
			return nil
		},
		stateMux:          sync.Mutex{},
		bpfWatcherEnabled: false,
		fetchPorts:        true,
		findingCriteria:   FindingCriteria(cfg),
		output:            msg.NewQueue[[]Event[ProcessAttrs]](msg.ChannelBufferLen(1)),
	}

	accounterExited := make(chan struct{})
	go func() {
		acc.run(ctx)
		close(accounterExited)
	}()

	eventsChan := testutil.ReadChannel(t, channelReturner, testTimeout)

	assert.True(t, acc.portFetchRequired()) // initial state means poll all ports until we are ready to look for binds in bpf
	eventsChan <- watcher.Event{Type: watcher.NewPort}
	assert.True(t, acc.portFetchRequired())
	eventsChan <- watcher.Event{Type: watcher.Ready}
	assert.True(t, acc.portFetchRequired()) // we must see it true one more time
	assert.EventuallyWithTf(t, func(c *assert.CollectT) {
		assert.False(c, acc.portFetchRequired()) // eventually we'll see this being false
	}, 5*time.Second, 100*time.Millisecond, "eventsChan was never set")
	assert.False(t, acc.portFetchRequired()) // another false after that

	// we send new port watcher event which matches the port range
	eventsChan <- watcher.Event{Type: watcher.NewPort, Payload: 8080}
	assert.EventuallyWithTf(t, func(c *assert.CollectT) {
		assert.True(c, acc.portFetchRequired()) // eventually we'll see this being true
	}, 5*time.Second, 100*time.Millisecond, "eventsChan was never set")
	assert.False(t, acc.portFetchRequired()) // once we see it true, next time it's false

	// we send port that's not in our port range
	eventsChan <- watcher.Event{Type: watcher.NewPort, Payload: 8090}
	// 5 seconds should be enough to have the channel send something
	for range 5 {
		assert.False(t, acc.portFetchRequired()) // once we see it true, next time it's false
		time.Sleep(1 * time.Second)
	}

	// WHEN its context is cancelled
	cancel()
	// THEN the main loop exits
	select {
	case <-accounterExited:
	// ok!
	case <-time.After(testTimeout):
		assert.Fail(t, "expected to exit the main loop")
	}
}

// auxiliary function just to allow comparing slices whose order is not deterministic
func sort(events []Event[ProcessAttrs]) []Event[ProcessAttrs] {
	slices.SortFunc(events, func(a, b Event[ProcessAttrs]) int {
		return int(a.Obj.pid) - int(b.Obj.pid)
	})
	return events
}

func TestMinProcessAge(t *testing.T) {
	count := 1
	processAgeFunc = func(pid app.PID) time.Duration {
		if pid == 3 {
			return time.Duration(0)
		}
		count++
		return time.Duration(count * 1000000 * 1000)
	}

	processPidsFunc = func() ([]int32, error) {
		return []int32{1, 2, 3}, nil
	}

	userConfig := bytes.NewBufferString("channel_buffer_len: 33")
	t.Setenv("OTEL_EBPF_OPEN_PORT", "8080-8089")

	cfg, err := obi.LoadConfig(userConfig)
	require.NoError(t, err)

	channelReturner := make(chan chan<- watcher.Event)

	acc := pollAccounter{
		cfg:      cfg,
		interval: time.Hour, // don't let the inner loop mess with our test
		pidPorts: map[pidPort]ProcessAttrs{},
		listProcesses: func(bool) (map[app.PID]ProcessAttrs, error) {
			return nil, nil
		},
		executableReady: func(_ app.PID) (string, bool) {
			return "", true
		},
		loadBPFWatcher: func(_ context.Context, _ *ebpfcommon.EBPFEventContext, _ *obi.Config, events chan<- watcher.Event) error {
			channelReturner <- events
			return nil
		},
		loadBPFLogger: func(context.Context, *ebpfcommon.EBPFEventContext, *obi.Config) error {
			return nil
		},
		stateMux:          sync.Mutex{},
		bpfWatcherEnabled: false,
		fetchPorts:        true,
		findingCriteria:   FindingCriteria(cfg),
		output:            msg.NewQueue[[]Event[ProcessAttrs]](msg.ChannelBufferLen(1)),
	}

	procs, err := fetchProcessPorts(false)
	require.NoError(t, err)
	process, ok := procs[app.PID(1)]

	assert.True(t, ok)
	assert.True(t, acc.processTooNew(process))

	// Pid 3 has 0 duration meaning we had to scan it without checking duration
	// it's never too new
	process, ok = procs[app.PID(3)]

	assert.True(t, ok)
	assert.False(t, acc.processTooNew(process))

	for range 10 {
		procs, err = fetchProcessPorts(false)
		require.NoError(t, err)
	}

	process, ok = procs[app.PID(1)]

	assert.True(t, ok)
	assert.False(t, acc.processTooNew(process))
}

// TestForgetPIDs_ReemitsExistingProcess verifies that when a PID was already seen by the watcher,
// forgetting it causes the next poll to emit EventCreated again.
// This supports the use case of adding an existing process to the dynamic selector.
func TestForgetPIDs_ReemitsExistingProcess(t *testing.T) {
	p1 := ProcessAttrs{pid: 1, openPorts: []uint32{3030}}
	p2 := ProcessAttrs{pid: 2, openPorts: []uint32{123}}
	acc := pollAccounter{
		interval: time.Hour,
		cfg:      &obi.Config{},
		pids:     map[app.PID]ProcessAttrs{},
		pidPorts: map[pidPort]ProcessAttrs{},
		listProcesses: func(bool) (map[app.PID]ProcessAttrs, error) {
			return map[app.PID]ProcessAttrs{p1.pid: p1, p2.pid: p2}, nil
		},
		executableReady: func(app.PID) (string, bool) {
			return "", true
		},
		loadBPFWatcher: func(context.Context, *ebpfcommon.EBPFEventContext, *obi.Config, chan<- watcher.Event) error {
			return nil
		},
		loadBPFLogger: func(context.Context, *ebpfcommon.EBPFEventContext, *obi.Config) error {
			return nil
		},
	}
	procs, err := acc.listProcesses(false)
	require.NoError(t, err)
	events := acc.snapshot(procs)
	require.Len(t, events, 2)
	assert.Equal(t, EventCreated, sort(events)[0].Type)
	assert.Equal(t, EventCreated, sort(events)[1].Type)
	// Second snapshot: no new events (already seen)
	events2 := acc.snapshot(procs)
	assert.Empty(t, events2)
	// Forget p1 only
	acc.forgetPIDs([]app.PID{1})
	// Next snapshot: p1 should appear as created again
	events3 := acc.snapshot(procs)
	require.Len(t, events3, 1)
	assert.Equal(t, EventCreated, events3[0].Type)
	assert.Equal(t, app.PID(1), events3[0].Obj.pid)
	// p2 still not re-emitted
	events4 := acc.snapshot(procs)
	assert.Empty(t, events4)
}

// A process that runs for less than a poll interval is reported from its exec
// event, ahead of the next poll and regardless of MinProcessAge.
func TestExecEventsReportProcessesBeforeThePoll(t *testing.T) {
	cfg := &obi.Config{}
	cfg.Discovery.ExecEvents = true
	cfg.Discovery.MinProcessAge = time.Hour

	var mu sync.Mutex
	alive := map[app.PID]bool{100: true}
	listed := map[app.PID]ProcessAttrs{100: {pid: 100, processAge: 2 * time.Hour}}
	watcherEvents := make(chan chan<- watcher.Event, 1)

	acc := pollAccounter{
		cfg:      cfg,
		interval: time.Hour, // only the first poll runs
		pids:     map[app.PID]ProcessAttrs{},
		pidPorts: map[pidPort]ProcessAttrs{},
		listProcesses: func(bool) (map[app.PID]ProcessAttrs, error) {
			return listed, nil
		},
		executableReady: func(app.PID) (string, bool) { return "/usr/bin/agent", true },
		execMayMatch: func(pid app.PID) bool {
			mu.Lock()
			defer mu.Unlock()
			return alive[pid]
		},
		loadBPFWatcher: func(_ context.Context, _ *ebpfcommon.EBPFEventContext, _ *obi.Config, events chan<- watcher.Event) error {
			watcherEvents <- events
			return nil
		},
		loadBPFLogger: func(context.Context, *ebpfcommon.EBPFEventContext, *obi.Config) error {
			return nil
		},
		execPIDs: make(chan app.PID, execQueueLen),
		output:   msg.NewQueue[[]Event[ProcessAttrs]](msg.ChannelBufferLen(10)),
	}
	out := acc.output.Subscribe()
	go acc.run(t.Context())

	events := testutil.ReadChannel(t, watcherEvents, testTimeout)
	assert.Equal(t, []Event[ProcessAttrs]{{Type: EventCreated, Obj: listed[100]}},
		testutil.ReadChannel(t, out, testTimeout), "the first poll")

	mu.Lock()
	alive[200] = true
	mu.Unlock()
	events <- watcher.Event{Type: watcher.NewExec, Payload: 200}
	// it exited before its event was read
	events <- watcher.Event{Type: watcher.NewExec, Payload: 300}

	got := testutil.ReadChannel(t, out, testTimeout)
	require.Len(t, got, 1)
	assert.Equal(t, EventCreated, got[0].Type)
	assert.Equal(t, app.PID(200), got[0].Obj.pid)

	// a known process that runs another program is evaluated again
	events <- watcher.Event{Type: watcher.NewExec, Payload: 100}
	got = testutil.ReadChannel(t, out, testTimeout)
	require.Len(t, got, 1)
	assert.Equal(t, Event[ProcessAttrs]{Type: EventCreated, Obj: ProcessAttrs{pid: 100, openPorts: []uint32{}}}, got[0])
}

func TestExecSnapshotIsSeenByTheNextPoll(t *testing.T) {
	acc := pollAccounter{
		cfg:             &obi.Config{},
		pids:            map[app.PID]ProcessAttrs{},
		pidPorts:        map[pidPort]ProcessAttrs{},
		executableReady: func(app.PID) (string, bool) { return "/usr/bin/agent", true },
		execMayMatch:    func(app.PID) bool { return true },
	}

	created := acc.execSnapshot([]app.PID{7})
	require.Len(t, created, 1)

	// still running at the next poll: not reported twice
	assert.Empty(t, acc.snapshot(map[app.PID]ProcessAttrs{7: {pid: 7, processAge: time.Millisecond}}))

	// gone by the poll after: reported deleted
	deleted := acc.snapshot(map[app.PID]ProcessAttrs{})
	require.Len(t, deleted, 1)
	assert.Equal(t, EventDeleted, deleted[0].Type)
	assert.Equal(t, app.PID(7), deleted[0].Obj.pid)
}

func TestDrainExecPIDsDeduplicates(t *testing.T) {
	acc := pollAccounter{execPIDs: make(chan app.PID, 4)}
	for _, pid := range []app.PID{2, 1, 2} {
		acc.queueExec(pid)
	}

	assert.Equal(t, []app.PID{1, 2}, acc.drainExecPIDs(1))
	assert.Empty(t, acc.execPIDs)
}

func TestQueueExecNeverBlocks(t *testing.T) {
	acc := pollAccounter{execPIDs: make(chan app.PID, 1)}
	acc.queueExec(1)
	acc.queueExec(2) // dropped: the next poll finds it

	assert.Equal(t, []app.PID{1}, acc.drainExecPIDs(<-acc.execPIDs))

	// exec events off: no queue, nothing to block on
	(&pollAccounter{}).queueExec(3)
}

func TestExecMayMatch(t *testing.T) {
	agent := execImage{exePath: "/opt/py/bin/python3.12", cmdArgs: "/srv/agents/run.py --task x"}
	read := func(pid app.PID) (execImage, error) {
		if pid == 404 {
			return execImage{}, os.ErrNotExist
		}
		return agent, nil
	}
	glob := services.NewGlob
	agentLabel := glob("agent")

	cases := []struct {
		name     string
		selector services.GlobAttributes
		want     bool
	}{
		{"executable matches", services.GlobAttributes{Path: glob("*/python3*")}, true},
		{"executable differs", services.GlobAttributes{Path: glob("*/node")}, false},
		{"executable and arguments match", services.GlobAttributes{Path: glob("*/python3*"), CmdArgs: glob("/srv/agents/*")}, true},
		{"arguments differ", services.GlobAttributes{Path: glob("*/python3*"), CmdArgs: glob("*-m http.server*")}, false},
		{"ports are not known at exec time", services.GlobAttributes{OpenPorts: services.IntEnum{Ranges: []services.IntRange{{Start: 8080}}}}, false},
		{"target pid", services.GlobAttributes{PIDs: []uint32{7}}, true},
		{"other target pid", services.GlobAttributes{PIDs: []uint32{8}, Path: glob("*/python3*")}, false},
		{"languages are decided later", services.GlobAttributes{Languages: glob("python")}, true},
		{"containers are decided later", services.GlobAttributes{Path: glob("*/node"), ContainersOnly: true}, true},
		{"pod labels are decided later", services.GlobAttributes{PodLabels: map[string]*services.GlobAttr{"app": &agentLabel}}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sel := tc.selector
			assert.Equal(t, tc.want, execMayMatch([]services.Selector{&sel}, false, 7, read))
		})
	}

	none := []services.Selector{&services.GlobAttributes{Path: glob("*/node")}}
	assert.True(t, execMayMatch(none, true, 7, read), "dynamic selection decides later")
	assert.False(t, execMayMatch(none, true, 404, read), "a process that is gone")
}

func TestReadExecImage(t *testing.T) {
	image, err := readExecImage(app.PID(os.Getpid()))
	require.NoError(t, err)

	exe, err := os.Executable()
	require.NoError(t, err)
	assert.Equal(t, exe, image.exePath)
	assert.Equal(t, strings.Join(os.Args[1:], " "), image.cmdArgs)
}

// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package agenttools

import (
	"net/netip"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"go.opentelemetry.io/obi/pkg/appolly/app/agentact"
	"go.opentelemetry.io/obi/pkg/appolly/app/request"
)

const externalAPI = "10.1.2.3:8080"

func kernelConfigTracker() *tracker {
	cfg := testConfig()
	cfg.KernelActivity.Enabled = true
	return testTracker(cfg)
}

func open(at int64, pid, tid uint32, path string) agentact.Event {
	return agentact.Event{Kind: agentact.KindOpen, Time: at, PID: pid, TID: tid, RootPID: agentPID, Path: path, Result: 3}
}

func connect(at int64, tid uint32, remote string) agentact.Event {
	return agentact.Event{
		Kind: agentact.KindConnect, Time: at, PID: agentPID, TID: tid, RootPID: agentPID,
		Remote: netip.MustParseAddrPort(remote), Local: netip.MustParseAddrPort("127.0.0.1:40000"),
	}
}

func fork(at int64, child, parent, spawner uint32) agentact.Event {
	return agentact.Event{Kind: agentact.KindFork, Time: at, PID: child, TID: child, RootPID: agentPID, ParentPID: parent, SpawnerTID: spawner}
}

func execve(at int64, pid uint32, path string, args ...string) agentact.Event {
	return agentact.Event{Kind: agentact.KindExec, Time: at, PID: pid, TID: pid, RootPID: agentPID, Path: path, Args: args}
}

func exit(at int64, pid uint32, code int) agentact.Event {
	return agentact.Event{Kind: agentact.KindExit, Time: at, PID: pid, TID: pid, RootPID: agentPID, ExitCode: code}
}

// settle emits the windows waiting for in-flight activity, including those
// that wait for evidence about the agent's loop thread.
func settle(tr *tracker) []request.Span {
	out := toolSpans(tr.tick(0, time.Now().Add(activitySettle)))
	return append(out, toolSpans(tr.tick(0, time.Now().Add(activitySettle+loopThreadHold+time.Second)))...)
}

func spansByName(spans []request.Span) map[string]request.Span {
	byName := map[string]request.Span{}
	for _, s := range spans {
		byName[s.TraceName()] = s
	}
	return byName
}

func eventPaths(s request.Span) []string {
	var paths []string
	for _, e := range s.Events {
		for _, a := range e.Attributes {
			if a.Key == "file.path" || a.Key == "server.address" {
				paths = append(paths, a.Value.AsString())
			}
		}
	}
	return paths
}

func eventAttribution(s request.Span) []string {
	var rules []string
	for _, e := range s.Events {
		for _, a := range e.Attributes {
			if a.Key == attributionKey {
				rules = append(rules, a.Value.AsString())
			}
		}
	}
	return rules
}

func TestAttributionOfParallelCalls(t *testing.T) {
	tr := kernelConfigTracker()
	n := inference(0, 100*ms, noTools,
		call("call_r3adFile0001", "read_file", `{"path":"beta.txt"}`),
		call("call_runC0mmand01", "run_command", `"{\"cmd\": \"uname -a\"}"`),
		call("call_httpG3t00001", "http_get", `{"url":"http://`+externalAPI+`/api/weather?run=t1"}`))
	n1 := inference(200*ms, 250*ms, results("call_r3adFile0001", "call_runC0mmand01", "call_httpG3t00001"))

	tr.onActivity([]agentact.Event{
		// read_file on a pool thread: the path decides, the thread follows
		open(110*ms, agentPID, 11, "/h/fixtures/beta.txt"),
		open(111*ms, agentPID, 11, "/h/fixtures/beta.meta"),
		// run_command spawns uname from another thread
		fork(112*ms, 2001, agentPID, 12),
		execve(113*ms, 2001, "/usr/bin/uname", "uname", "-a"),
		open(114*ms, 2001, 2001, "/etc/os-release"),
		exit(115*ms, 2001, 0),
		// http_get connects to the external API
		connect(116*ms, 13, externalAPI),
		// the next model call reconnects to the LLM inside the window
		connect(198*ms, 10, "10.0.0.1:443"),
		// interpreter noise and work no rule decides
		open(117*ms, agentPID, 14, "/x/lib/python3.12/json/__init__.py"),
		open(118*ms, agentPID, 14, "/tmp/unclaimed"),
	})

	require.Empty(t, feed(t, tr, n, n1), "windows wait for in-flight activity")
	out := settle(tr)
	byName := spansByName(out)
	require.Len(t, out, 5)

	window := byName[request.ToolWindowSpanName]
	assert.Equal(t, []string{"/tmp/unclaimed"}, eventPaths(window))
	assert.Equal(t, request.ToolAttributionAmbiguous, window.InferredTool.Attribution)
	assert.True(t, window.InferredTool.ActivityComplete)

	read := byName["execute_tool read_file"]
	assert.ElementsMatch(t, []string{"/h/fixtures/beta.txt", "/h/fixtures/beta.meta"}, eventPaths(read))
	assert.Equal(t, request.ToolAttributionThread, read.InferredTool.Attribution, "the weakest rule used")
	assert.ElementsMatch(t, []string{request.ToolAttributionArgumentMatch, request.ToolAttributionThread}, eventAttribution(read))
	assert.Equal(t, 2, read.InferredTool.Activity.FileOpens)

	run := byName["execute_tool run_command"]
	require.NotNil(t, run.InferredTool.Command)
	assert.Equal(t, "uname", run.InferredTool.Command.ExecutableName)
	assert.Equal(t, 1, run.InferredTool.Activity.Execs)
	assert.Equal(t, 1, run.InferredTool.Activity.FileOpens, "the child's activity counts on its tool")
	assert.Equal(t, request.ToolAttributionArgumentMatch, run.InferredTool.Attribution)

	proc := byName["uname"]
	assert.Equal(t, request.EventTypeProcessExec, proc.Type)
	assert.Equal(t, run.SpanID, proc.ParentSpanID)
	assert.Equal(t, 112*ms, proc.Start)
	assert.Equal(t, 115*ms, proc.End)
	assert.Equal(t, []string{"/etc/os-release"}, eventPaths(proc))
	assert.Equal(t, request.ToolAttributionArgumentMatch, proc.Process.Attribution)
	assert.False(t, proc.Process.OutlivedWindow)

	get := byName["execute_tool http_get"]
	assert.Equal(t, []string{"10.1.2.3"}, eventPaths(get))
	assert.Equal(t, 1, get.InferredTool.Activity.Connects)
}

func TestAttributionByLineageAndThread(t *testing.T) {
	tr := kernelConfigTracker()
	n := inference(0, 100*ms, noTools,
		call("call_l1stDir00001", "list_dir", `{"cmd":"sh -c 'ls fixtures | wc -l'"}`),
		call("call_wr1teNote001", "write_note", `{"text":"hello"}`))
	n1 := inference(300*ms, 350*ms, results("call_l1stDir00001", "call_wr1teNote001"))

	tr.onActivity([]agentact.Event{
		fork(110*ms, 3001, agentPID, 21),
		execve(111*ms, 3001, "/bin/sh", "sh", "-c", "ls fixtures | wc -l"),
		// grandchildren follow their parent
		fork(112*ms, 3002, 3001, 3001),
		execve(113*ms, 3002, "/usr/bin/ls", "ls", "fixtures"),
		exit(114*ms, 3002, 0),
		// thread 21 matched list_dir, so its later work follows
		open(120*ms, agentPID, 21, "/tmp/listing.cache"),
		// a child that outlives the window
		fork(130*ms, 3003, agentPID, 22),
		execve(131*ms, 3003, "/usr/bin/sleep", "sleep", "60"),
	})

	feed(t, tr, n, n1)
	out := settle(tr)
	byName := spansByName(out)

	list := byName["execute_tool list_dir"]
	assert.Equal(t, request.ToolAttributionThread, list.InferredTool.Attribution)
	assert.Equal(t, []string{"/tmp/listing.cache"}, eventPaths(list))
	assert.Equal(t, 2, list.InferredTool.Activity.Execs)
	assert.Nil(t, list.InferredTool.Command, "two processes ran")

	sh, ls := byName["sh"], byName["ls"]
	assert.Equal(t, list.SpanID, sh.ParentSpanID)
	assert.Equal(t, sh.SpanID, ls.ParentSpanID)
	assert.Equal(t, request.ToolAttributionLineage, ls.Process.Attribution)
	assert.True(t, sh.Process.OutlivedWindow, "sh never exited")

	sleep := byName["sleep"]
	assert.Equal(t, byName[request.ToolWindowSpanName].SpanID, sleep.ParentSpanID, "nothing ties sleep to a call")
	assert.True(t, sleep.Process.OutlivedWindow)
	assert.Equal(t, 300*ms, sleep.End)
}

func TestSequentialLoopThreadOwnsItsWindow(t *testing.T) {
	tr := kernelConfigTracker()
	c := newChat("look up the user")
	n := inference(0, 100*ms, c.history(), call("call_l00kupUser01", "lookup_user", `{"user_id":"u-1001"}`))
	n1 := inference(200*ms, 250*ms, c.answer("call_l00kupUser01").history())

	tr.onActivity([]agentact.Event{
		// the loop thread runs the tool: nothing in its arguments names users.db
		open(110*ms, agentPID, 10, "/srv/users.db"),
		fork(120*ms, 4001, agentPID, 10),
		execve(121*ms, 4001, "/usr/bin/sqlite3", "sqlite3"),
		exit(130*ms, 4001, 0),
		// a telemetry thread works in the background, inside windows or not
		open(140*ms, agentPID, 77, "/tmp/metrics.json"),
		connect(150*ms, 77, "10.9.9.9:4318"),
		open(240*ms, agentPID, 77, "/tmp/metrics.json"),
	})

	feed(t, tr, n, n1)
	assert.Empty(t, toolSpans(tr.tick(0, time.Now().Add(activitySettle))), "waits for evidence that the thread is shared")
	byName := spansByName(settle(tr))

	tool := byName["execute_tool lookup_user"]
	assert.Equal(t, []string{"/srv/users.db"}, eventPaths(tool))
	assert.Equal(t, request.ToolAttributionThread, tool.InferredTool.Attribution)
	assert.Equal(t, 1, tool.InferredTool.Activity.Execs)
	assert.Equal(t, 2, tool.InferredTool.Activity.Ambiguous, "the telemetry thread's work is not the tool's")
	assert.Equal(t, tool.SpanID, byName["sqlite3"].ParentSpanID)
}

func TestQuietPoolThreadOwnsTheOnlyWindow(t *testing.T) {
	tr := kernelConfigTracker()
	c := newChat("write it down")
	n := inference(0, 100*ms, c.history(), call("call_wr1teNote001", "write_note", `{"text":"hello"}`))
	n1 := inference(200*ms, 250*ms, c.answer("call_wr1teNote001").history())
	n1.TID = 11
	tr.onActivity([]agentact.Event{
		// a pool thread writes a note whose name nothing in the arguments gives
		open(120*ms, agentPID, 31, "/tmp/notes/note-1000-1.txt"),
		// the pool thread did other work before the first call
		open(20*ms, agentPID, 31, "/etc/os-release"),
		// a background thread that keeps working outside windows
		open(130*ms, agentPID, 32, "/tmp/cache.db"),
		open(220*ms, agentPID, 32, "/tmp/cache.db"),
	})

	feed(t, tr, n, n1)
	out := settle(tr)
	require.Len(t, out, 1)
	assert.Equal(t, []string{"/tmp/notes/note-1000-1.txt"}, eventPaths(out[0]))
	assert.Equal(t, request.ToolAttributionExclusive, out[0].InferredTool.Attribution)
	assert.Equal(t, 1, out[0].InferredTool.Activity.Ambiguous)
}

func TestBackgroundThreadQuietAfterTheWindowOpened(t *testing.T) {
	tr := kernelConfigTracker()
	c := newChat("slow job")
	start := int64(time.Second)
	n := inference(start, start+100*ms, c.history(), call("call_sl0wJob00001", "slow_job", `{"seconds":"1"}`))
	n1 := inference(start+1200*ms, start+1250*ms, c.answer("call_sl0wJob00001").history())
	n1.TID = 11
	var evs []agentact.Event
	// a telemetry thread that wrote every 25ms since the agent started, and
	// only happens to be quiet between the windows
	for at := int64(0); at < start+1200*ms; at += 25 * ms {
		evs = append(evs, open(at, agentPID, 77, "/tmp/metrics.json"))
	}
	tr.onActivity(evs)

	feed(t, tr, n, n1)
	out := settle(tr)
	require.Len(t, out, 1)
	assert.Empty(t, out[0].Events)
	assert.Positive(t, out[0].InferredTool.Activity.Ambiguous)
}

func TestSharedExecutorThreadDecidesNothingAcrossConversations(t *testing.T) {
	tr := kernelConfigTracker()
	ca, cb := newChat("conversation A"), newChat("conversation B")
	a := inference(0, 100*ms, ca.history(), call("call_aaaaaaaaaaaa", "read_doc", `{"path":"sess/a.txt"}`))
	b := inference(10*ms, 120*ms, cb.history(), call("call_bbbbbbbbbbbb", "lookup", `{"id":"b"}`))
	feed(t, tr, a, b)
	tr.onActivity([]agentact.Event{
		// one executor thread serves both conversations
		open(130*ms, agentPID, 31, "/srv/sess/a.txt"),
		open(140*ms, agentPID, 31, "/srv/users.db"),
	})
	feed(t, tr, inference(200*ms, 300*ms, ca.answer("call_aaaaaaaaaaaa").history()),
		inference(210*ms, 300*ms, cb.answer("call_bbbbbbbbbbbb").history()))
	byName := spansByName(settle(tr))

	assert.Equal(t, []string{"/srv/sess/a.txt"}, eventPaths(byName["execute_tool read_doc"]))
	assert.Empty(t, eventPaths(byName["execute_tool lookup"]))
}

func TestConcurrentConversationsShareNoEvidence(t *testing.T) {
	tr := kernelConfigTracker()
	ca, cb := newChat("conversation A"), newChat("conversation B")
	// one event loop thread runs both conversations
	a := inference(0, 100*ms, ca.history(), call("call_aaaaaaaaaaaa", "read_doc", `{"path":"sess/a.txt"}`))
	b := inference(10*ms, 120*ms, cb.history(), call("call_bbbbbbbbbbbb", "run_command", `{"cmd":"echo b"}`))
	feed(t, tr, a, b)
	tr.onActivity([]agentact.Event{
		open(130*ms, agentPID, 31, "/srv/sess/a.txt"),
		fork(131*ms, 3001, agentPID, 10),
		execve(132*ms, 3001, "/usr/bin/echo", "echo", "b"),
		exit(140*ms, 3001, 0),
		// work nothing ties to either conversation
		open(150*ms, agentPID, 10, "/tmp/session.log"),
	})
	ra := inference(200*ms, 300*ms, ca.answer("call_aaaaaaaaaaaa").history())
	rb := inference(210*ms, 300*ms, cb.answer("call_bbbbbbbbbbbb").history())
	feed(t, tr, ra, rb)
	byName := spansByName(settle(tr))

	read := byName["execute_tool read_doc"]
	assert.Equal(t, []string{"/srv/sess/a.txt"}, eventPaths(read))
	assert.Equal(t, request.ToolAttributionArgumentMatch, read.InferredTool.Attribution)
	run := byName["execute_tool run_command"]
	assert.Empty(t, eventPaths(run))
	assert.Equal(t, 1, run.InferredTool.Activity.Execs)
	assert.Equal(t, run.SpanID, byName["echo"].ParentSpanID)

	overlap := read.InferredTool.Activity.AmbiguousOverlap + run.InferredTool.Activity.AmbiguousOverlap
	assert.Positive(t, overlap, "the session log is nobody's")
	for _, s := range []request.Span{read, run} {
		assert.NotContains(t, eventPaths(s), "/tmp/session.log")
	}
}

func TestStaleUnmatchedWindowClaimsNothing(t *testing.T) {
	cfg := testConfig()
	cfg.CloseOnNextRequest = false
	cfg.MaxWindow = 5 * time.Second
	cfg.KernelActivity.Enabled = true
	tr := testTracker(cfg)

	n := inference(0, 100*ms, newChat("one").history(), call("call_l0stResult01", "read_file", `{}`))
	// the result was not captured; the process carries on with a new response
	n1 := inference(200*ms, 300*ms, newChat("two").history(), call("call_runC0mmand01", "run_command", `{"cmd":"uname -a"}`))
	feed(t, tr, n, n1)
	tr.onActivity([]agentact.Event{
		fork(400*ms, 2001, agentPID, 10),
		execve(401*ms, 2001, "/usr/bin/uname", "uname", "-a"),
		exit(410*ms, 2001, 0),
		open(405*ms, 2001, 2001, "/etc/os-release"),
		open(450*ms, agentPID, 10, "/data/notes.txt"),
	})
	feed(t, tr, inference(500*ms, 600*ms, newChat("two").answer("call_runC0mmand01").history()))
	out := settle(tr)
	out = append(out, toolSpans(tr.tick(6*int64(time.Second), time.Now().Add(time.Hour)))...)
	out = append(out, settle(tr)...)

	unameSpans := 0
	for _, s := range out {
		if s.Type == request.EventTypeProcessExec && s.Process.ExecutableName == "uname" {
			unameSpans++
		}
		if s.InferredTool != nil && s.InferredTool.Name == "read_file" {
			assert.Equal(t, request.ToolWindowEndNextInference, s.InferredTool.WindowEnd)
			assert.Empty(t, s.Events)
			assert.Zero(t, s.InferredTool.Activity.Execs)
		}
	}
	assert.Equal(t, 1, unameSpans, "a process yields at most one span")
}

func TestUnmatchedCallOnlyTakesActivityNamingIt(t *testing.T) {
	tr := kernelConfigTracker()
	c := newChat("finish")
	assert.Empty(t, feed(t, tr, inference(0, 100*ms, c.history(), call("call_f1nalAnsw3r", "final_answer", `{"answer":"done"}`))))
	tr.onActivity([]agentact.Event{
		// the harness writes its own files after the conversation ended
		open(150*ms, agentPID, 10, "/runs/run.json"),
		exit(200*ms, agentPID, 0),
	})
	tr.onActivity([]agentact.Event{{Kind: agentact.KindExit, Time: 200 * ms, PID: agentPID, TID: agentPID, RootPID: agentPID}})

	out := toolSpans(tr.tick(0, time.Now().Add(minExitGrace)))
	require.Len(t, out, 1)
	assert.Equal(t, request.ToolOutcomeUnmatchedExit, out[0].InferredTool.Outcome)
	assert.Empty(t, out[0].Events)
	assert.Equal(t, 1, out[0].InferredTool.Activity.Ambiguous)
}

func TestPoolWorkerServingTwoCallsDecidesNothing(t *testing.T) {
	tr := kernelConfigTracker()
	const worker = 5001
	tr.onActivity([]agentact.Event{fork(1*ms, worker, agentPID, 10)})
	n := inference(10*ms, 100*ms, noTools,
		call("call_l0adDocC0001", "load_doc", `{"path":"docC.txt"}`),
		call("call_l0adDocD0001", "load_doc", `{"path":"docD.txt"}`))
	n1 := inference(300*ms, 350*ms, results("call_l0adDocC0001", "call_l0adDocD0001"))
	tr.onActivity([]agentact.Event{
		open(110*ms, worker, worker, "/fixtures/docC.txt"),
		open(130*ms, worker, worker, "/tmp/cache/entry-1.bin"),
		open(150*ms, worker, worker, "/fixtures/docD.txt"),
		open(170*ms, worker, worker, "/tmp/cache/entry-2.bin"),
	})

	feed(t, tr, n, n1)
	byName := map[string][]string{}
	for _, s := range settle(tr) {
		byName[s.TraceName()+" "+s.InferredTool.CallID] = eventPaths(s)
	}

	assert.Equal(t, []string{"/fixtures/docC.txt"}, byName["execute_tool load_doc call_l0adDocC0001"])
	assert.Equal(t, []string{"/fixtures/docD.txt"}, byName["execute_tool load_doc call_l0adDocD0001"])
	assert.ElementsMatch(t, []string{"/tmp/cache/entry-1.bin", "/tmp/cache/entry-2.bin"}, byName[request.ToolWindowSpanName+" "])
}

func TestNestedModelCallSharesTheLoopThread(t *testing.T) {
	tr := kernelConfigTracker()
	outer := newChat("ask the expert")
	n := inference(0, 100*ms, outer.history(), call("call_askExp3rt001", "ask_expert", `{"question":"gamma"}`))
	// the tool runs a sub-agent on the loop thread, which reads a file
	inner := newChat("what does gamma say")
	sub := inference(110*ms, 150*ms, inner.history(), call("call_r3adGamma001", "read_file", `{"path":"gamma.txt"}`))
	tr.onActivity([]agentact.Event{
		open(160*ms, agentPID, 10, "/fixtures/gamma.txt"),
		open(170*ms, agentPID, 10, "/tmp/expert.cache"),
	})
	subResult := inference(180*ms, 220*ms, inner.answer("call_r3adGamma001").history())
	n1 := inference(300*ms, 350*ms, outer.answer("call_askExp3rt001").history())

	feed(t, tr, n, sub, subResult, n1)
	byName := spansByName(settle(tr))

	read := byName["execute_tool read_file"]
	assert.Equal(t, []string{"/fixtures/gamma.txt"}, eventPaths(read))
	ask := byName["execute_tool ask_expert"]
	assert.Equal(t, 300*ms, ask.End, "the outer window holds the sub-agent")
	assert.Equal(t, request.ToolOutcomeResult, ask.InferredTool.Outcome)
	assert.NotContains(t, eventPaths(ask), "/fixtures/gamma.txt")
	assert.NotContains(t, eventPaths(ask), "/tmp/expert.cache", "the loop thread served another LLM call")
}

func TestExclusiveCommandFailure(t *testing.T) {
	tr := kernelConfigTracker()
	n := inference(0, 100*ms, noTools, call("call_runC0mmand01", "run_command", `{"cmd":"cat missing"}`))
	n1 := inference(200*ms, 250*ms, results("call_runC0mmand01"))

	tr.onActivity([]agentact.Event{
		fork(110*ms, 4001, agentPID, 10),
		execve(111*ms, 4001, "/usr/bin/cat", "cat", "missing"),
		open(112*ms, 4001, 4001, "/h/missing"),
		exit(113*ms, 4001, 1),
		// a connect after the window closed belongs to no call
		connect(260*ms, 15, "10.0.0.9:443"),
	})

	feed(t, tr, n, n1)
	byName := spansByName(settle(tr))
	run := byName["execute_tool run_command"]
	assert.Equal(t, request.ToolAttributionArgumentMatch, run.InferredTool.Attribution)
	assert.True(t, run.InferredTool.IsError, "the only command exited non-zero")
	assert.Equal(t, 1, run.InferredTool.Command.ExitCode)
	assert.Zero(t, run.InferredTool.Activity.Connects)
	assert.Positive(t, run.InferredTool.ActiveNs)
}

func TestWindowArrivingAfterItsActivityExpiredIsIncomplete(t *testing.T) {
	tr := kernelConfigTracker()
	tr.onActivity([]agentact.Event{open(150*ms, agentPID, 21, "/data/alpha.txt")})
	// 40s later, spans held for their parents arrive
	tr.tick(40*int64(time.Second), time.Now())
	n := inference(0, 100*ms, noTools, call("call_r3adFile0001", "read_file", `{"path":"alpha.txt"}`))
	n1 := inference(200*ms, 250*ms, results("call_r3adFile0001"))
	feed(t, tr, n, n1)

	out := settle(tr)
	require.Len(t, out, 1)
	assert.True(t, out[0].InferredTool.ActivityObserved)
	assert.False(t, out[0].InferredTool.ActivityComplete)
}

func TestOverflowIsCountedOnSingleCalls(t *testing.T) {
	tr := kernelConfigTracker()
	n := inference(0, 100*ms, noTools, call("call_sc4nFiles001", "scan", `{}`))
	feed(t, tr, n)
	var evs []agentact.Event
	for i := range maxActivityPerProcess + 1000 {
		evs = append(evs, open(101*ms+int64(i), agentPID, 10, "/data/f"+pidString(uint32(i))))
	}
	tr.onActivity(evs)
	feed(t, tr, inference(200*ms, 250*ms, results("call_sc4nFiles001")))

	out := settle(tr)
	require.Len(t, out, 1)
	a := out[0].InferredTool.Activity
	assert.False(t, out[0].InferredTool.ActivityComplete)
	assert.Equal(t, maxActivityPerProcess+1000, a.FileOpens+a.Dropped, "every open is either counted or reported lost")
	assert.LessOrEqual(t, len(out[0].Events), 64)
}

func TestKernelDropsMarkWindowsIncomplete(t *testing.T) {
	tr := kernelConfigTracker()
	n := inference(0, 100*ms, noTools, call("call_r3adFile0001", "read_file", `{"path":"alpha.txt"}`))
	tr.onActivity([]agentact.Event{
		open(120*ms, agentPID, 10, "/data/alpha.txt"),
		{Kind: agentact.KindDrops, Time: 150 * ms, PID: agentPID, RootPID: agentPID, Dropped: 40, DroppedSince: 130 * ms},
	})
	feed(t, tr, n, inference(200*ms, 250*ms, results("call_r3adFile0001")))

	out := settle(tr)
	require.Len(t, out, 1)
	assert.False(t, out[0].InferredTool.ActivityComplete)
	assert.Equal(t, 40, out[0].InferredTool.Activity.Dropped)
}

func TestNotCollectedBeforeArming(t *testing.T) {
	cfg := testConfig()
	cfg.KernelActivity.Enabled = true
	cfg.KernelActivity.Probation = time.Second
	roots := &fakeRoots{}
	registry := agentact.NewRootRegistry()
	registry.Attach(roots)
	tr := newTracker(cfg, 0, registry)
	tr.now = func() int64 { return 10 * int64(time.Second) }

	// the process started long before its first tool call
	tr.onActivity([]agentact.Event{{Kind: agentact.KindFork, Time: 1, PID: 2, RootPID: agentPID, RootStart: 1}})
	start := 5 * int64(time.Second)
	n := inference(start, start+100*ms, noTools, call("call_r3adFile0001", "read_file", `{"path":"alpha.txt"}`))
	feed(t, tr, n, inference(start+200*ms, start+250*ms, results("call_r3adFile0001")))

	assert.Equal(t, []uint32{agentPID}, roots.armed, "a process that asks for a tool call is armed")
	out := settle(tr)
	require.Len(t, out, 1)
	assert.False(t, out[0].InferredTool.ActivityComplete, "nothing was collected before arming")
}

func TestLLMServerIsDisarmed(t *testing.T) {
	cfg := testConfig()
	cfg.KernelActivity.Enabled = true
	roots := &fakeRoots{}
	registry := agentact.NewRootRegistry()
	registry.Attach(roots)
	tr := newTracker(cfg, 0, registry)

	feed(t, tr, inference(0, 100*ms, noTools, call("call_g4tewayCall1", "read_file", `{}`)))
	require.Equal(t, []uint32{agentPID}, roots.armed)
	tr.onSpans([]request.Span{{Type: request.EventTypeHTTP, Path: "/v1/messages", Pid: request.PidInfo{HostPID: agentPID}}})
	assert.Equal(t, []uint32{agentPID}, roots.disarmed)
}

type fakeRoots struct {
	armed, disarmed []uint32
}

func (f *fakeRoots) Arm(pid uint32) error {
	f.armed = append(f.armed, pid)
	return nil
}

func (f *fakeRoots) Disarm(pid uint32) {
	f.disarmed = append(f.disarmed, pid)
}

func TestPathExclusion(t *testing.T) {
	m := agentact.NewPathMatcher([]string{"/proc/", "/dev/", "*/lib/python3*", "site-packages", "cacert.pem", "*ca-bundle*"})
	for path, excluded := range map[string]bool{
		"/dev/null":                                    true,
		"/home/alice/dev/project/main.py":              false,
		"/home/alice/sys/notes.txt":                    false,
		"/usr/lib/python3.12/json/__init__.py":         true,
		"/venv/lib/site-packages/requests/__init__.py": true,
		"/home/alice/my-site-packages-audit.txt":       false,
		"/venv/certifi/cacert.pem":                     true,
		"/etc/pki/tls/certs/ca-bundle.crt":             true,
	} {
		assert.Equal(t, excluded, m.Matches(path), path)
	}
}

func TestArgumentMatcher(t *testing.T) {
	m := newArgMatcher(`{"path":"notes/2026/todo.md","url":"https://api.example.com:8443/v1","n":3}`)
	assert.Equal(t, scoreSuffix+scoreExact, m.score([]string{"/home/u/notes/2026/todo.md", "todo.md"}))
	assert.Equal(t, scoreSuffix, m.score([]string{"/home/u/notes/2026/todo.md"}))
	assert.Equal(t, scoreSubstring, m.score([]string{"api.example.com:8443"}))
	assert.Zero(t, m.score([]string{"/etc/hosts", "3"}), "single characters never match")

	free := newArgMatcher("*** Begin Patch\n*** Update File: src/main.go")
	assert.Equal(t, scoreExact, free.score([]string{"src/main.go"}))

	// the argument that differs decides between two commands
	a, b := newArgMatcher(`{"cmd":"echo s01"}`), newArgMatcher(`{"cmd":"echo s02"}`)
	argv := []string{"/usr/bin/echo", "echo", "echo", "s01"}
	assert.Greater(t, a.score(argv), b.score(argv))
}

func TestAgentStartedByAnotherInstrumentedProcess(t *testing.T) {
	cfg := testConfig()
	cfg.KernelActivity.Enabled = true
	cfg.KernelActivity.Probation = time.Minute
	roots := &fakeRoots{}
	registry := agentact.NewRootRegistry()
	registry.Attach(roots)
	tr := newTracker(cfg, 0, registry)

	// `npm start` is instrumented too: the agent's activity is reported in its tree
	const launcher = 900
	under := func(e agentact.Event) agentact.Event {
		e.RootPID, e.RootStart = launcher, 1
		return e
	}
	tr.onActivity([]agentact.Event{
		under(fork(1*ms, agentPID, launcher, launcher)),
		under(open(2*ms, agentPID, agentPID, "/app/node_modules/x/index.js")),
	})
	c := newChat("read the readme")
	feed(t, tr, inference(10*ms, 100*ms, c.history(), call("call_r3adFile0001", "read_file", `{"path":"README.md"}`)))
	assert.Equal(t, []uint32{launcher}, roots.armed, "the tree the agent is reported in is armed")

	tr.onActivity([]agentact.Event{
		under(open(110*ms, agentPID, 10, "/app/README.md")),
		// another child of the launcher is not the agent's
		under(fork(115*ms, 950, launcher, launcher)),
		under(open(120*ms, 950, 950, "/app/other/README.md")),
	})
	feed(t, tr, inference(200*ms, 250*ms, c.answer("call_r3adFile0001").history()))

	out := settle(tr)
	require.Len(t, out, 1)
	assert.Equal(t, []string{"/app/README.md"}, eventPaths(out[0]))
	assert.Zero(t, out[0].InferredTool.Activity.Execs)
}

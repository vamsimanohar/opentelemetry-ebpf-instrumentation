// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package tracesgen

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/collector/pdata/ptrace"
	"go.opentelemetry.io/otel/attribute"
	semconv "go.opentelemetry.io/otel/semconv/v1.41.0"
	trace2 "go.opentelemetry.io/otel/trace"

	"go.opentelemetry.io/obi/pkg/appolly/app/request"
	attr "go.opentelemetry.io/obi/pkg/export/attributes/names"
	"go.opentelemetry.io/obi/pkg/export/instrumentations"
)

func inferredToolSpan() request.Span {
	return request.Span{
		Type:         request.EventTypeGenAIToolExec,
		RequestStart: 1_000,
		Start:        1_000,
		End:          9_000,
		TraceID:      trace2.TraceID{1},
		SpanID:       trace2.SpanID{2},
		ParentSpanID: trace2.SpanID{3},
		Links: []request.SpanLink{{
			TraceID:    trace2.TraceID{1},
			SpanID:     trace2.SpanID{4},
			Attributes: []attribute.KeyValue{attribute.String("obi.link.kind", "tool_call_source")},
		}},
		Events: []request.SpanEvent{{
			Name:       "obi.file.open",
			Time:       2_000,
			Attributes: []attribute.KeyValue{semconv.FilePath("/h/beta.txt")},
		}},
		InferredTool: &request.InferredTool{
			Name:             "run_command",
			CallID:           "call_1",
			Type:             "function",
			Provider:         "openai",
			Arguments:        `{"cmd":"uname -a"}`,
			Result:           `"Linux"`,
			WindowStart:      request.ToolWindowStartResponseEnd,
			WindowEnd:        request.ToolWindowEndToolResult,
			Outcome:          request.ToolOutcomeResult,
			Attribution:      request.ToolAttributionThread,
			ParallelCount:    1,
			ActivityObserved: true,
			ActivityComplete: true,
			Activity:         request.ToolActivityCounts{Execs: 1, FileOpens: 1},
			Command:          &request.ObservedProcess{ExecutableName: "uname", ExecutablePath: "/usr/bin/uname", Exited: true},
		},
	}
}

func TestInferredToolSpanExport(t *testing.T) {
	span := inferredToolSpan()
	assert.True(t, acceptSpan(instrumentations.NewInstrumentationSelection(nil), &span))

	exported := generateSingleTraceSpan(t, &span, map[attr.Name]struct{}{})
	assert.Equal(t, "execute_tool run_command", exported.Name())
	assert.Equal(t, ptrace.SpanKindInternal, exported.Kind())
	assert.Equal(t, ptrace.StatusCodeUnset, exported.Status().Code())

	assertSpanStringAttribute(t, exported, semconv.GenAIOperationNameKey, "execute_tool")
	assertSpanStringAttribute(t, exported, semconv.GenAIToolNameKey, "run_command")
	assertSpanStringAttribute(t, exported, semconv.GenAIToolCallIDKey, "call_1")
	assertSpanStringAttribute(t, exported, semconv.GenAIToolTypeKey, "function")
	assertSpanStringAttribute(t, exported, semconv.GenAIProviderNameKey, "openai")
	assertSpanStringAttribute(t, exported, obiToolOutcome, "result")
	assertSpanStringAttribute(t, exported, obiToolWindowEnd, "tool_result")
	assertSpanStringAttribute(t, exported, semconv.ProcessExecutableNameKey, "uname")
	inferred, ok := exported.Attributes().Get(string(obiToolInferred))
	require.True(t, ok)
	assert.True(t, inferred.Bool())
	execs, ok := exported.Attributes().Get(string(obiActivityExecs))
	require.True(t, ok)
	assert.Equal(t, int64(1), execs.Int())
	complete, ok := exported.Attributes().Get(string(obiActivityComplete))
	require.True(t, ok)
	assert.True(t, complete.Bool())
	assertSpanStringAttribute(t, exported, obiToolAttribution, "thread")
	assertSpanAttributeAbsent(t, exported, string(obiToolTruncated))

	// payloads are opt-in
	assertSpanAttributeAbsent(t, exported, string(attr.GenAIToolCallArguments))
	assertSpanAttributeAbsent(t, exported, string(attr.GenAIToolCallResult))

	require.Equal(t, 1, exported.Events().Len())
	assert.Equal(t, "obi.file.open", exported.Events().At(0).Name())
	path, ok := exported.Events().At(0).Attributes().Get(string(semconv.FilePathKey))
	require.True(t, ok)
	assert.Equal(t, "/h/beta.txt", path.Str())

	require.Equal(t, 1, exported.Links().Len())
	kind, ok := exported.Links().At(0).Attributes().Get("obi.link.kind")
	require.True(t, ok)
	assert.Equal(t, "tool_call_source", kind.Str())
}

func TestInferredToolSpanPayloadsWhenSelected(t *testing.T) {
	span := inferredToolSpan()
	exported := generateSingleTraceSpan(t, &span, map[attr.Name]struct{}{
		attr.GenAIToolCallArguments: {},
		attr.GenAIToolCallResult:    {},
	})
	assertSpanStringAttribute(t, exported, attribute.Key(attr.GenAIToolCallArguments), `{"cmd":"uname -a"}`)
	assertSpanStringAttribute(t, exported, attribute.Key(attr.GenAIToolCallResult), `"Linux"`)

	span.InferredTool.Arguments = "{\"text\":\"caf\xc3"
	span.InferredTool.PayloadTruncated = true
	exported = generateSingleTraceSpan(t, &span, map[attr.Name]struct{}{attr.GenAIToolCallArguments: {}})
	assertSpanStringAttribute(t, exported, attribute.Key(attr.GenAIToolCallArguments), `{"text":"caf`)
	truncated, ok := exported.Attributes().Get(string(obiToolTruncated))
	require.True(t, ok)
	assert.True(t, truncated.Bool())

	span.InferredTool.IsError = true
	exported = generateSingleTraceSpan(t, &span, map[attr.Name]struct{}{attr.ErrorType: {}})
	assert.Equal(t, ptrace.StatusCodeError, exported.Status().Code())
	assertSpanStringAttribute(t, exported, semconv.ErrorTypeKey, "_OTHER")
}

func TestObservedProcessSpanExport(t *testing.T) {
	span := request.Span{
		Type:         request.EventTypeProcessExec,
		RequestStart: 1_000,
		Start:        1_000,
		End:          2_000,
		TraceID:      trace2.TraceID{1},
		SpanID:       trace2.SpanID{5},
		Process: &request.ObservedProcess{
			PID:            2001,
			ExecutableName: "uname",
			ExecutablePath: "/usr/bin/un\xffame",
			CommandArgs:    []string{"uname", "-a"},
			OutlivedWindow: true,
			Attribution:    "lineage",
		},
	}

	exported := generateSingleTraceSpan(t, &span, map[attr.Name]struct{}{})
	assert.Equal(t, "uname", exported.Name())
	assert.Equal(t, ptrace.SpanKindClient, exported.Kind())
	assertSpanAttributeAbsent(t, exported, string(semconv.ProcessCommandArgsKey))
	assertSpanAttributeAbsent(t, exported, string(semconv.ProcessExitCodeKey))
	assertSpanStringAttribute(t, exported, semconv.ProcessExecutablePathKey, "/usr/bin/uname")
	assertSpanStringAttribute(t, exported, obiToolAttribution, "lineage")
	outlived, ok := exported.Attributes().Get(string(obiProcessOutlivedWin))
	require.True(t, ok)
	assert.True(t, outlived.Bool())

	exported = generateSingleTraceSpan(t, &span, map[attr.Name]struct{}{attr.ProcessCommandArgs: {}})
	args, ok := exported.Attributes().Get(string(semconv.ProcessCommandArgsKey))
	require.True(t, ok)
	assert.Equal(t, 2, args.Slice().Len())
}

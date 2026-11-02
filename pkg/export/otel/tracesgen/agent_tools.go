// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package tracesgen // import "go.opentelemetry.io/obi/pkg/export/otel/tracesgen"

import (
	"go.opentelemetry.io/collector/pdata/pcommon"
	"go.opentelemetry.io/collector/pdata/ptrace"
	"go.opentelemetry.io/otel/attribute"
	semconv "go.opentelemetry.io/otel/semconv/v1.41.0"

	"go.opentelemetry.io/obi/pkg/appolly/app/request"
	"go.opentelemetry.io/obi/pkg/export/attributes"
	attr "go.opentelemetry.io/obi/pkg/export/attributes/names"
)

// OBI-specific attributes of the inferred tool-call spans, declared in
// schemas/obi/groups/gen_ai.
const (
	obiToolInferred       = attribute.Key("obi.gen_ai.tool.inferred")
	obiToolWindowStart    = attribute.Key("obi.gen_ai.tool.window.start_source")
	obiToolWindowEnd      = attribute.Key("obi.gen_ai.tool.window.end_source")
	obiToolOutcome        = attribute.Key("obi.gen_ai.tool.outcome")
	obiToolParallelCount  = attribute.Key("obi.gen_ai.tool.parallel_count")
	obiToolAttribution    = attribute.Key("obi.gen_ai.tool.attribution")
	obiToolActiveNs       = attribute.Key("obi.gen_ai.tool.active_ns")
	obiActivityExecs      = attribute.Key("obi.activity.execs")
	obiActivityFileOpens  = attribute.Key("obi.activity.file_opens")
	obiActivityConnects   = attribute.Key("obi.activity.connects")
	obiActivityDropped    = attribute.Key("obi.activity.dropped")
	obiActivityComplete   = attribute.Key("obi.activity.complete")
	obiActivityAmbiguous  = attribute.Key("obi.activity.ambiguous")
	obiActivityOverlap    = attribute.Key("obi.activity.ambiguous_overlap")
	obiToolTruncated      = attribute.Key("obi.gen_ai.tool.payload_truncated")
	obiProcessOutlivedWin = attribute.Key("obi.process.outlived_window")
)

// inferredToolAttributes describes an execute_tool span, or the window span
// of parallel calls, which has no tool name and no GenAI operation.
func inferredToolAttributes(span *request.Span, optionalAttrs map[attr.Name]struct{}) []attribute.KeyValue {
	tool := span.InferredTool
	if tool == nil {
		return nil
	}

	attrs := []attribute.KeyValue{obiToolInferred.Bool(true)}
	if span.Type == request.EventTypeGenAIToolExec {
		attrs = append(attrs,
			semconv.GenAIOperationNameExecuteTool,
			semconv.GenAIToolName(tool.Name),
		)
		attrs = appendIfSet(attrs, semconv.GenAIToolCallID, tool.CallID)
		attrs = appendIfSet(attrs, semconv.GenAIToolType, tool.Type)
		_, withArgs := optionalAttrs[attr.GenAIToolCallArguments]
		if withArgs {
			attrs = appendIfSet(attrs, attribute.Key(attr.GenAIToolCallArguments).String, attributes.SanitizeUTF8(tool.Arguments))
		}
		_, withResult := optionalAttrs[attr.GenAIToolCallResult]
		if withResult {
			attrs = appendIfSet(attrs, attribute.Key(attr.GenAIToolCallResult).String, attributes.SanitizeUTF8(tool.Result))
		}
		if tool.PayloadTruncated && (withArgs || withResult) {
			attrs = append(attrs, obiToolTruncated.Bool(true))
		}
		attrs = appendIfSet(attrs, obiToolOutcome.String, tool.Outcome)
	}
	attrs = appendIfSet(attrs, semconv.GenAIProviderNameKey.String, tool.Provider)
	attrs = appendIfSet(attrs, obiToolWindowStart.String, tool.WindowStart)
	attrs = appendIfSet(attrs, obiToolWindowEnd.String, tool.WindowEnd)
	attrs = append(attrs, obiToolParallelCount.Int(tool.ParallelCount))

	if tool.ActivityObserved {
		attrs = appendIfSet(attrs, obiToolAttribution.String, tool.Attribution)
		attrs = append(attrs,
			obiActivityComplete.Bool(tool.ActivityComplete),
			obiActivityExecs.Int(tool.Activity.Execs),
			obiActivityFileOpens.Int(tool.Activity.FileOpens),
			obiActivityConnects.Int(tool.Activity.Connects),
			obiActivityDropped.Int(tool.Activity.Dropped),
			obiActivityAmbiguous.Int(tool.Activity.Ambiguous),
			obiActivityOverlap.Int(tool.Activity.AmbiguousOverlap),
		)
		if tool.ActiveNs > 0 {
			attrs = append(attrs, obiToolActiveNs.Int64(tool.ActiveNs))
		}
	}

	// the command-execution refinement: the tool ran exactly one process
	if cmd := tool.Command; cmd != nil {
		attrs = append(attrs, semconv.ProcessExecutableName(attributes.SanitizeUTF8(cmd.ExecutableName)))
		attrs = appendIfSet(attrs, semconv.ProcessExecutablePath, attributes.SanitizeUTF8(cmd.ExecutablePath))
		if cmd.Exited {
			attrs = append(attrs, semconv.ProcessExitCode(cmd.ExitCode))
		}
	}

	return attrs
}

// observedProcessAttributes follows the CLI client span: a process the agent
// started while a tool window was open.
func observedProcessAttributes(span *request.Span, optionalAttrs map[attr.Name]struct{}) []attribute.KeyValue {
	proc := span.Process
	if proc == nil {
		return nil
	}

	attrs := []attribute.KeyValue{
		semconv.ProcessExecutableName(attributes.SanitizeUTF8(proc.ExecutableName)),
		semconv.ProcessPID(int(proc.PID)),
	}
	attrs = appendIfSet(attrs, semconv.ProcessExecutablePath, attributes.SanitizeUTF8(proc.ExecutablePath))
	attrs = appendIfSet(attrs, obiToolAttribution.String, proc.Attribution)
	if proc.Exited {
		attrs = append(attrs, semconv.ProcessExitCode(proc.ExitCode))
	}
	if proc.OutlivedWindow {
		attrs = append(attrs, obiProcessOutlivedWin.Bool(true))
	}
	if _, ok := optionalAttrs[attr.ProcessCommandArgs]; ok && len(proc.CommandArgs) > 0 {
		args := make([]string, len(proc.CommandArgs))
		for i, a := range proc.CommandArgs {
			args[i] = attributes.SanitizeUTF8(a)
		}
		attrs = append(attrs, semconv.ProcessCommandArgs(args...))
	}
	return attrs
}

func appendSpanEvents(dst ptrace.Span, events []request.SpanEvent) {
	for i := range events {
		ev := dst.Events().AppendEmpty()
		ev.SetName(events[i].Name)
		ev.SetTimestamp(pcommon.NewTimestampFromTime(request.MonotonicTime(events[i].Time)))
		addAttrsToMap(events[i].Attributes, ev.Attributes())
	}
}

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

func TestAgentRunSpanExport(t *testing.T) {
	span := request.Span{
		Type:         request.EventTypeGenAIAgentRun,
		RequestStart: 1_000,
		Start:        1_000,
		End:          9_000,
		TraceID:      trace2.TraceID{1},
		SpanID:       trace2.SpanID{2},
		ExtraAttributes: []attribute.KeyValue{
			semconv.GenAIProviderNameOpenAI,
			attribute.Int("obi.gen_ai.agent.run.llm_calls", 3),
		},
	}
	assert.True(t, acceptSpan(instrumentations.NewInstrumentationSelection(nil), &span))

	exported := generateSingleTraceSpan(t, &span, map[attr.Name]struct{}{})
	assert.Equal(t, "invoke_agent", exported.Name())
	assert.Equal(t, ptrace.SpanKindInternal, exported.Kind())
	assert.True(t, exported.ParentSpanID().IsEmpty())
	assertSpanStringAttribute(t, exported, semconv.GenAIOperationNameKey, "invoke_agent")
	assertSpanStringAttribute(t, exported, semconv.GenAIProviderNameKey, "openai")
	calls, ok := exported.Attributes().Get("obi.gen_ai.agent.run.llm_calls")
	require.True(t, ok)
	assert.Equal(t, int64(3), calls.Int())
}

func TestExtraAttributesOnAnySpan(t *testing.T) {
	span := inferredToolSpan()
	span.ExtraAttributes = []attribute.KeyValue{attribute.String("obi.gen_ai.conversation.inferred_id", "c-1")}

	exported := generateSingleTraceSpan(t, &span, map[attr.Name]struct{}{})
	assertSpanStringAttribute(t, exported, "obi.gen_ai.conversation.inferred_id", "c-1")
	assertSpanStringAttribute(t, exported, semconv.GenAIOperationNameKey, "execute_tool")
}

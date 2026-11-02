// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package tracesgen

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"go.opentelemetry.io/otel/attribute"

	attr "go.opentelemetry.io/obi/pkg/export/attributes/names"
)

func TestAttributeValueLengthLimit(t *testing.T) {
	SetAttributeValueLengthLimit(4)
	t.Cleanup(func() { SetAttributeValueLengthLimit(0) })

	in := []attribute.KeyValue{
		attribute.String("short", "abc"),
		attribute.String("long", "héllo wörld"),
		attribute.StringSlice("list", []string{"ok", "toolong"}),
		attribute.Int("n", 123456),
	}
	out := limitAttributeValues(in)
	assert.Equal(t, "abc", out[0].Value.AsString())
	assert.Equal(t, "héll", out[1].Value.AsString(), "characters, not bytes, and never half of one")
	assert.Equal(t, []string{"ok", "tool"}, out[2].Value.AsStringSlice())
	assert.Equal(t, int64(123456), out[3].Value.AsInt64())
	assert.Equal(t, "héllo wörld", in[1].Value.AsString(), "the input is not changed")
}

func TestAttributeValueLengthLimitOnExport(t *testing.T) {
	SetAttributeValueLengthLimit(8)
	t.Cleanup(func() { SetAttributeValueLengthLimit(0) })

	span := inferredToolSpan()
	span.ExtraAttributes = []attribute.KeyValue{attribute.String("obi.test.long", strings.Repeat("x", 100))}
	exported := generateSingleTraceSpan(t, &span, map[attr.Name]struct{}{})
	v, ok := exported.Attributes().Get("obi.test.long")
	assert.True(t, ok)
	assert.Equal(t, "xxxxxxxx", v.Str())
}

func TestNoAttributeValueLengthLimitByDefault(t *testing.T) {
	in := []attribute.KeyValue{attribute.String("long", strings.Repeat("x", 1<<16))}
	assert.Equal(t, in, limitAttributeValues(in))
}

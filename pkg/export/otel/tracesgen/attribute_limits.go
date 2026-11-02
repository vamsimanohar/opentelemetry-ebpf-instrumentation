// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package tracesgen // import "go.opentelemetry.io/obi/pkg/export/otel/tracesgen"

import (
	"os"
	"strconv"
	"sync/atomic"

	"go.opentelemetry.io/otel/attribute"
)

// attributeValueLengthLimit caps the characters of string span attribute
// values, as the SDK variables OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT and
// OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT do. Zero means no limit. Backends reject
// whole spans whose attributes are too large to index, and GenAI messages
// grow with every turn.
var attributeValueLengthLimit atomic.Int64

func init() {
	for _, name := range []string{"OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT", "OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT"} {
		if n, err := strconv.Atoi(os.Getenv(name)); err == nil && n > 0 {
			attributeValueLengthLimit.Store(int64(n))
			return
		}
	}
}

// SetAttributeValueLengthLimit sets the limit for components that embed OBI
// and configure it in code. Zero removes it.
func SetAttributeValueLengthLimit(chars int) {
	attributeValueLengthLimit.Store(int64(max(chars, 0)))
}

// limitAttributeValues returns the attributes with their string values
// truncated to the limit. The input is copied before any change.
func limitAttributeValues(attrs []attribute.KeyValue) []attribute.KeyValue {
	limit := int(attributeValueLengthLimit.Load())
	if limit == 0 {
		return attrs
	}

	var out []attribute.KeyValue
	for i, a := range attrs {
		limited, changed := limitValue(a, limit)
		if !changed {
			continue
		}
		if out == nil {
			out = append([]attribute.KeyValue(nil), attrs...)
		}
		out[i] = limited
	}
	if out == nil {
		return attrs
	}
	return out
}

func limitValue(a attribute.KeyValue, limit int) (attribute.KeyValue, bool) {
	switch a.Value.Type() {
	case attribute.STRING:
		if s, cut := truncateChars(a.Value.AsString(), limit); cut {
			return a.Key.String(s), true
		}
	case attribute.STRINGSLICE:
		values := a.Value.AsStringSlice()
		changed := false
		for i, v := range values {
			if s, cut := truncateChars(v, limit); cut {
				values[i] = s
				changed = true
			}
		}
		if changed {
			return a.Key.StringSlice(values), true
		}
	}
	return a, false
}

// truncateChars cuts s after limit characters.
func truncateChars(s string, limit int) (string, bool) {
	if len(s) <= limit {
		return s, false
	}
	chars := 0
	for i := range s {
		if chars == limit {
			return s[:i], true
		}
		chars++
	}
	return s, false
}

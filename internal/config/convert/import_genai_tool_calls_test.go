// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package convert

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"go.opentelemetry.io/obi/internal/config/schema"
	"go.opentelemetry.io/obi/pkg/config"
)

func TestV2GenAIToolCallsRoundTrip(t *testing.T) {
	t.Parallel()

	cfg := defaultRuntimeConfig()
	tc := &cfg.EBPF.PayloadExtraction.HTTP.GenAI.ToolCalls
	tc.Enabled = true
	tc.MaxWindow = 3 * time.Minute
	tc.OnTimeout = config.ToolCallTimeoutDrop
	tc.MatchByName = false
	tc.UnmatchedMinActivity = 2
	tc.Services.Include = []string{"agent-*"}
	tc.Services.Exclude = []string{"agent-gateway"}
	tc.KernelActivity.Enabled = true
	tc.KernelActivity.Probation = 0
	tc.KernelActivity.PathExclude = []string{"/opt/agent/cache/"}

	_, ext := RuntimeToV2(&cfg)
	got, err := V2ToRuntime(ext)
	require.NoError(t, err)
	require.Equal(t, *tc, got.EBPF.PayloadExtraction.HTTP.GenAI.ToolCalls)
}

func TestV2GenAIToolCallsYAMLKeepsDefaults(t *testing.T) {
	t.Parallel()

	ext, err := schema.ParseReceiverYAML([]byte(`
version: '2.0'
instrumentation:
  http:
    buffer_size: 262144
    payload_extraction:
      enabled: [openai, anthropic]
      genai_tool_calls:
        enabled: true
        max_window: 2m
        services:
          exclude: [litellm-proxy]
        kernel_activity:
          enabled: true
          probation: 30s
          exclude_llm_endpoints: false
`))
	require.NoError(t, err)

	got, err := V2ToRuntime(ext)
	require.NoError(t, err)
	tc := got.EBPF.PayloadExtraction.HTTP.GenAI.ToolCalls
	require.True(t, tc.Enabled)
	require.Equal(t, 2*time.Minute, tc.MaxWindow)
	require.True(t, tc.MatchByName, "unset fields keep their defaults")
	require.True(t, tc.EmitUnmatched)
	require.True(t, tc.KernelActivity.Enabled)
	require.False(t, tc.KernelActivity.ExcludeLLMEndpoints)
	require.Equal(t, []string{"litellm-proxy"}, tc.Services.Exclude)
	require.Empty(t, tc.Services.Include)
	require.Equal(t, 30*time.Second, tc.KernelActivity.Probation)
	require.NotEmpty(t, tc.KernelActivity.PathExclude)
	require.NoError(t, tc.Validate(got.EBPF.BufferSizes.HTTP, &got.EBPF.PayloadExtraction.HTTP.GenAI))
}

func TestV2GenAIShapeDetection(t *testing.T) {
	t.Parallel()

	parse := func(t *testing.T, doc string) config.GenAIShapeDetectionConfig {
		t.Helper()
		ext, err := schema.ParseReceiverYAML([]byte(doc))
		require.NoError(t, err)
		got, err := V2ToRuntime(ext)
		require.NoError(t, err)
		return got.EBPF.PayloadExtraction.HTTP.GenAI.ShapeDetection
	}

	require.True(t, parse(t, `
version: '2.0'
instrumentation:
  http:
    payload_extraction:
      enabled: [openai]
`).Enabled, "on by default")

	require.False(t, parse(t, `
version: '2.0'
instrumentation:
  http:
    payload_extraction:
      enabled: [openai]
      genai_shape_detection:
        enabled: false
`).Enabled)

	cfg := defaultRuntimeConfig()
	cfg.EBPF.PayloadExtraction.HTTP.GenAI.ShapeDetection.Enabled = false
	_, ext := RuntimeToV2(&cfg)
	got, err := V2ToRuntime(ext)
	require.NoError(t, err)
	require.False(t, got.EBPF.PayloadExtraction.HTTP.GenAI.ShapeDetection.Enabled)
}

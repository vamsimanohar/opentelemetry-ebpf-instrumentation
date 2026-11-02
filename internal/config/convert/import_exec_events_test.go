// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package convert

import (
	"testing"

	"github.com/stretchr/testify/require"

	"go.opentelemetry.io/obi/internal/config/schema"
)

func TestV2CapturePolicyExecEvents(t *testing.T) {
	t.Parallel()

	parse := func(t *testing.T, doc string) bool {
		t.Helper()
		ext, err := schema.ParseReceiverYAML([]byte(doc))
		require.NoError(t, err)
		got, err := V2ToRuntime(ext)
		require.NoError(t, err)
		return got.Discovery.ExecEvents
	}

	require.False(t, parse(t, "version: '2.0'\n"), "off by default")
	require.True(t, parse(t, `
version: '2.0'
policy:
  exec_events: true
`))

	cfg := defaultRuntimeConfig()
	cfg.Discovery.ExecEvents = true
	_, ext := RuntimeToV2(&cfg)
	got, err := V2ToRuntime(ext)
	require.NoError(t, err)
	require.True(t, got.Discovery.ExecEvents)
}

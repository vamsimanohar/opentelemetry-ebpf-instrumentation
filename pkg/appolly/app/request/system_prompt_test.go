// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package request

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSystemPromptForms(t *testing.T) {
	for _, tc := range []struct {
		name, body, want string
	}{
		{name: "string", body: `{"system":"be brief","model":"m"}`, want: "be brief"},
		{name: "anthropic blocks", body: `{"system":[{"type":"text","text":"be brief"},{"type":"text","text":"cite","cache_control":{"type":"ephemeral"}}],"model":"m"}`, want: "be brief\ncite"},
		{name: "converse blocks", body: `{"system":[{"text":"be brief"}],"model":"m"}`, want: "be brief"},
		{name: "unexpected shape", body: `{"system":42,"model":"m"}`, want: ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var req AnthropicRequest
			require.NoError(t, json.Unmarshal([]byte(tc.body), &req))
			assert.Equal(t, SystemPrompt(tc.want), req.System)
			assert.Equal(t, "m", req.Model, "fields after system still decode")
		})
	}
}

// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package ebpfcommon

import (
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"go.opentelemetry.io/obi/pkg/appolly/app/request"
	"go.opentelemetry.io/obi/pkg/config"
)

// A self-hosted OpenAI-compatible server sends no vendor header: only shape
// detection classifies it, and only when it is switched on.
func TestPostProcessHTTPSpan_ShapeDetectionSwitch(t *testing.T) {
	const (
		reqBody  = `{"model":"llama-3.1-8b","messages":[{"role":"user","content":"hi"}]}`
		respBody = `{"id":"chatcmpl-1","object":"chat.completion","model":"llama-3.1-8b","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}]}`
	)

	for _, enabled := range []bool{false, true} {
		req, err := http.NewRequest(http.MethodPost, "http://127.0.0.1:8000/v1/chat/completions", strings.NewReader(reqBody))
		require.NoError(t, err)
		resp := &http.Response{StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"application/json"}},
			Body: io.NopCloser(strings.NewReader(respBody))}
		ctx := &EBPFParseContext{payloadExtraction: config.PayloadExtraction{HTTP: config.HTTPConfig{GenAI: config.GenAIConfig{
			OpenAI:         config.OpenAIConfig{Enabled: true},
			ShapeDetection: config.GenAIShapeDetectionConfig{Enabled: enabled},
		}}}}

		span := postProcessHTTPSpan(ctx, &request.Span{Type: request.EventTypeHTTPClient, Path: "/v1/chat/completions"}, req, resp)

		if enabled {
			assert.Equal(t, request.HTTPSubtypeOpenAICompatible, span.SubType)
			require.NotNil(t, span.GenAI)
			assert.Equal(t, "custom", span.GenAIProviderName())
		} else {
			assert.Nil(t, span.GenAI)
		}
	}
}

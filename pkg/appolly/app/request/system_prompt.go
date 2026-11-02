// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package request // import "go.opentelemetry.io/obi/pkg/appolly/app/request"

import (
	"encoding/json"
	"strings"
)

// SystemPrompt is a request's system prompt. Anthropic and Bedrock Converse
// accept a plain string or a list of text blocks; prompt caching and SDKs
// such as the Vercel AI SDK send the list. A type error there would stop the
// best-effort decoder before the messages and tools that follow it.
type SystemPrompt string

func (s *SystemPrompt) UnmarshalJSON(data []byte) error {
	var text string
	if err := json.Unmarshal(data, &text); err == nil {
		*s = SystemPrompt(text)
		return nil
	}

	var blocks []struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(data, &blocks); err != nil {
		// neither form: keep decoding the rest of the request
		return nil
	}

	texts := make([]string, 0, len(blocks))
	for _, b := range blocks {
		if b.Text != "" {
			texts = append(texts, b.Text)
		}
	}
	*s = SystemPrompt(strings.Join(texts, "\n"))
	return nil
}

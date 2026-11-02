// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package ebpfcommon // import "go.opentelemetry.io/obi/pkg/ebpf/common/http"

import (
	"bytes"
	"encoding/json"
	"net/http"
	"strings"

	"go.opentelemetry.io/obi/pkg/appolly/app/request"
	"go.opentelemetry.io/obi/pkg/config"
)

// The vendor detectors recognize a provider from its response headers or its
// hostname. Self-hosted servers (vLLM, Ollama), LiteLLM proxies and internal
// gateways send neither, but they serve the same endpoints with the same
// payloads, and those are what identify the format here: the endpoint path,
// a request body of that endpoint's shape, and a response of the matching
// shape. All three have to agree, so an unrelated API that happens to use one
// of these paths is not taken for an LLM call.

type genAIFormat uint8

const (
	genAIFormatNone genAIFormat = iota
	genAIFormatOpenAIChat
	genAIFormatOpenAIResponses
	genAIFormatAnthropicMessages
)

const (
	openAIChatCompletionsPath = "/chat/completions"
	openAIResponsesPath       = "/responses"

	// every Anthropic Messages client sends it; the path alone is too common
	anthropicVersionHeader = "Anthropic-Version"
	liteLLMHeaderPrefix    = "X-Litellm-"
	liteLLMProviderName    = "litellm"

	// how much of a streamed response is searched for its opening events
	shapeStreamSearchWindow = 2048
)

// GenAIShapeSpan classifies an OpenAI- or Anthropic-format call that no
// vendor detector claimed, for the formats whose extractor is enabled.
func GenAIShapeSpan(baseSpan *request.Span, req *http.Request, resp *http.Response, genAI *config.GenAIConfig) (request.Span, bool) {
	format := genAIFormatForEndpoint(req, genAI)
	if format == genAIFormatNone {
		return *baseSpan, false
	}

	reqB, ok := readHTTPRequestBody("GenAIShapeSpan", req, baseSpan)
	if !ok {
		return *baseSpan, false
	}

	respB, ok := readHTTPResponseBody("GenAIShapeSpan", resp, baseSpan)
	if !ok {
		return *baseSpan, false
	}

	if !requestHasShape(format, reqB) || !responseHasShape(format, respB, resp.StatusCode) {
		return *baseSpan, false
	}

	if format == genAIFormatAnthropicMessages {
		return anthropicSpan(baseSpan, req, reqB, respB), true
	}

	return openAICompatibleSpan(baseSpan, req, reqB, respB, shapeProviderName(resp.Header))
}

func genAIFormatForEndpoint(req *http.Request, genAI *config.GenAIConfig) genAIFormat {
	if req == nil || req.Method != http.MethodPost {
		return genAIFormatNone
	}

	openAI := genAI.OpenAI.Enabled || genAI.OpenAICompatible.Enabled
	path := strings.TrimSuffix(requestPath(req), "/")

	switch {
	case openAI && strings.HasSuffix(path, openAIChatCompletionsPath):
		return genAIFormatOpenAIChat
	case openAI && strings.HasSuffix(path, openAIResponsesPath):
		return genAIFormatOpenAIResponses
	case genAI.Anthropic.Enabled && strings.HasSuffix(path, anthropicMessagesPath) &&
		req.Header.Get(anthropicVersionHeader) != "":
		return genAIFormatAnthropicMessages
	}

	return genAIFormatNone
}

// requestHasShape checks the request carries the fields its endpoint
// requires. A body cut off by the capture limit is accepted without the model
// when the conversation came first: clients do not agree on the field order.
func requestHasShape(format genAIFormat, reqB []byte) bool {
	fields, complete, ok := topLevelJSONKinds(reqB)
	if !ok {
		return false
	}

	hasModel := fields["model"] == '"' || !complete

	switch format {
	case genAIFormatOpenAIChat, genAIFormatAnthropicMessages:
		return hasModel && fields["messages"] == '['
	case genAIFormatOpenAIResponses:
		return hasModel && (fields["input"] == '"' || fields["input"] == '[')
	}

	return false
}

func responseHasShape(format genAIFormat, respB []byte, status int) bool {
	if !looksLikeJSON(respB) {
		return streamHasShape(format, respB)
	}

	fields, _, ok := topLevelJSONKinds(respB)
	if !ok {
		return false
	}

	failed := status >= http.StatusBadRequest

	switch format {
	case genAIFormatOpenAIChat:
		object := extractJSONStringField(respB, "object", responseHeaderSearchWindow)
		return object == "chat.completion" || fields["choices"] == '[' ||
			(failed && openAIErrorShape(fields, object))
	case genAIFormatOpenAIResponses:
		object := extractJSONStringField(respB, "object", responseHeaderSearchWindow)
		return object == "response" || (failed && openAIErrorShape(fields, object))
	case genAIFormatAnthropicMessages:
		kind := extractJSONStringField(respB, "type", responseHeaderSearchWindow)
		return kind == "message" || (failed && kind == "error")
	}

	return false
}

// openAIErrorShape accepts the OpenAI error object and the flat error vLLM
// returns.
func openAIErrorShape(fields map[string]byte, object string) bool {
	return fields["error"] == '{' || object == "error"
}

func streamHasShape(format genAIFormat, respB []byte) bool {
	head := respB[:min(len(respB), shapeStreamSearchWindow)]

	switch format {
	case genAIFormatOpenAIChat:
		return bytes.Contains(head, []byte("data:")) && bytes.Contains(head, []byte(`"chat.completion.chunk"`))
	case genAIFormatOpenAIResponses:
		return bytes.Contains(head, []byte(`"response.created"`))
	case genAIFormatAnthropicMessages:
		return bytes.Contains(head, []byte(`"message_start"`))
	}

	return false
}

// shapeProviderName names the server when it says what it is. Anything else
// reports as `custom`, and `server.address` still identifies the endpoint.
func shapeProviderName(hdr http.Header) string {
	for name := range hdr {
		if strings.HasPrefix(name, liteLLMHeaderPrefix) {
			return liteLLMProviderName
		}
	}

	return ""
}

// topLevelJSONKinds reports the kind of each top-level value of a JSON object
// by its first byte ('"', '[', '{', or the first byte of a scalar), for a body
// that may be cut off by the capture limit: complete is false when the object
// does not end within the body. ok is false when the body is not an object.
func topLevelJSONKinds(body []byte) (map[string]byte, bool, bool) {
	dec := json.NewDecoder(bytes.NewReader(body))
	if tok, err := dec.Token(); err != nil || tok != json.Delim('{') {
		return nil, false, false
	}

	kinds := map[string]byte{}
	for dec.More() {
		keyTok, err := dec.Token()
		if err != nil {
			return kinds, false, true
		}
		key, _ := keyTok.(string)

		if kind := nextValueByte(body[dec.InputOffset():]); kind != 0 {
			kinds[key] = kind
		}
		if err := skipJSONValue(dec); err != nil {
			return kinds, false, true
		}
	}

	_, err := dec.Token()
	return kinds, err == nil, true
}

// nextValueByte returns the first byte of the value that follows a key
func nextValueByte(rest []byte) byte {
	for _, b := range rest {
		switch b {
		case ' ', '\t', '\n', '\r', ':':
			continue
		default:
			return b
		}
	}

	return 0
}

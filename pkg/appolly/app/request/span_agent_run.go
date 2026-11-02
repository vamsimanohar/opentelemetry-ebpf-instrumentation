// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package request // import "go.opentelemetry.io/obi/pkg/appolly/app/request"

// How a GenAI request was tied to an earlier request of its conversation, from
// the strongest evidence to the weakest.
const (
	// the request continues a stored response (previous_response_id)
	ConversationLinkPreviousResponse = "previous_response"
	// the request sends results of tool calls an earlier response requested
	ConversationLinkCallID = "call_id"
	// the request resends the history of an earlier request, with turns appended
	ConversationLinkHistoryPrefix = "history_prefix"
	// the request starts a conversation: its history holds no model turn
	ConversationLinkNew = "new"
	// the request continues several conversations equally well
	ConversationLinkAmbiguous = "ambiguous"
	// the request continues a conversation none of whose requests were seen
	ConversationLinkUnknown = "unknown"
)

// What a request sends after the model's last turn.
const (
	TrailingTurnUser        = "user"
	TrailingTurnToolResults = "tool_results"
	TrailingTurnOther       = "other"
)

// GenAIConversation ties an LLM call to the other calls of its conversation in
// the same process.
type GenAIConversation struct {
	// Key identifies the conversation among those of the process. It derives
	// from a seed chosen at random when OBI starts, so it identifies nothing
	// outside of OBI and must not be exported.
	Key uint64
	// Link tells how the request was tied to its conversation.
	Link string
	// Adopted lists the keys of requests seen earlier with an unknown link
	// that turned out to continue this request: they belong to its
	// conversation.
	Adopted []uint64
	// TrailingTurn tells what the request sent after the model's last turn.
	TrailingTurn string
	// OffersTools tells that the request declared tools to the model.
	OffersTools bool
	// RequestsTools tells that the response requested tool calls that the
	// client runs.
	RequestsTools bool
}

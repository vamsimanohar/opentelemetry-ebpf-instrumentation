// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package agenttools // import "go.opentelemetry.io/obi/pkg/appolly/agenttools"

import (
	"hash/maphash"
	"slices"
	"time"

	"go.opentelemetry.io/obi/pkg/appolly/app/request"
)

const (
	// histories, responses and call ids remembered per process to tie later
	// requests to their conversation
	maxConversationPrefixes   = 4096
	maxConversationResponses  = 1024
	maxConversationCalls      = 4096
	maxConversationAliases    = 1024
	maxUnresolvedPerProcess   = 64
	unresolvedConversationTTL = 30 * time.Second
)

// conversations ties the LLM calls of a process to their conversations, from
// the evidence their requests carry: the stored response a request continues,
// the ids of the tool calls it answers, and the history it resends.
type conversations struct {
	// prefixes are the histories requests sent, by hash: a later request of
	// the same conversation resends one of them with turns appended
	prefixes    map[uint64][]historyPrefix
	prefixOrder []historyPrefix
	responses   map[string]uint64
	responseIDs []string
	calls       map[string]uint64
	callIDs     []string
	aliases     map[uint64]uint64
	aliasOrder  []uint64
	unresolved  []unresolvedRequest
}

type historyPrefix struct {
	len  int
	hash uint64
	key  uint64
}

// unresolvedRequest continues a conversation none of whose earlier requests
// were seen, maybe because they reach the stage later.
type unresolvedRequest struct {
	key        uint64
	start      int64
	prevRespID string
	resultIDs  []string
	prefixes   []historyPrefix
	cut        bool
}

type conversationLink struct {
	key  uint64
	link string
}

func newConversations() *conversations {
	return &conversations{
		prefixes:  map[uint64][]historyPrefix{},
		responses: map[string]uint64{},
		calls:     map[string]uint64{},
		aliases:   map[uint64]uint64{},
	}
}

// conversationRequest is the evidence one LLM call carries.
type conversationRequest struct {
	span       *request.Span
	results    []request.ToolResult
	prevRespID string
	responseID string
	calls      []request.ToolCall
	// the prefixes of the request's history at each of its elements
	prefixes []historyPrefix
	// the capture cut the history: the request has turns after its last prefix
	cut bool
}

// annotateConversation ties an LLM call to its conversation and annotates its
// span with it.
func (t *tracker) annotateConversation(p *procState, span *request.Span, req *requestResults, now int64) {
	if p.conversations == nil {
		p.conversations = newConversations()
	}
	c := p.conversations

	calls := slices.DeleteFunc(slices.Clone(span.GenAI.ToolCalls()), func(tc request.ToolCall) bool {
		return tc.ServerSide || tc.Name == ""
	})
	cr := conversationRequest{
		span:       span,
		results:    req.results,
		prevRespID: req.prevRespID,
		responseID: span.GenAI.ResponseID(),
		calls:      calls,
	}
	cr.prefixes, cr.cut = historyPrefixes(t.seed, req.history)

	resolved := c.resolve(&cr)
	if resolved.link == request.ConversationLinkUnknown || resolved.link == request.ConversationLinkNew ||
		resolved.link == request.ConversationLinkAmbiguous {
		resolved.key = t.newConversationKey()
	}
	if resolved.link == request.ConversationLinkUnknown && !span.GenAI.ContinuesConversation() {
		resolved.link = request.ConversationLinkNew
	}

	c.register(&cr, resolved.key)
	adopted := c.adopt(&cr, resolved.key, now)
	if resolved.link == request.ConversationLinkUnknown {
		c.addUnresolved(&cr, resolved.key)
	}

	span.Conversation = &request.GenAIConversation{
		Key:           resolved.key,
		Link:          resolved.link,
		Adopted:       adopted,
		TrailingTurn:  span.GenAI.TrailingTurnKind(req.results),
		OffersTools:   span.GenAI.OffersTools(),
		RequestsTools: len(calls) > 0,
	}
}

// newConversationKey identifies a new conversation by a hash of a counter,
// so that keys tell nothing about the conversation.
func (t *tracker) newConversationKey() uint64 {
	t.conversationKeys++
	return maphash.Comparable(t.seed, t.conversationKeys)
}

// historyPrefixes hashes the history up to the end of each of its elements:
// a request continues another when one of its prefixes is the whole history
// the other sent.
func historyPrefixes(seed maphash.Seed, history []byte) ([]historyPrefix, bool) {
	ends := request.ArrayElementEnds(history)
	if len(ends) == 0 {
		return nil, false
	}

	var h maphash.Hash
	h.SetSeed(seed)
	prefixes := make([]historyPrefix, len(ends))
	from := 0
	for i, end := range ends {
		_, _ = h.Write(history[from:end])
		from = end
		prefixes[i] = historyPrefix{len: end, hash: h.Sum64()}
	}

	return prefixes, conversationOf(seed, history).cut
}

// resolve finds the conversation a request continues: the one of the stored
// response, else the one whose tool calls it answers, else the one whose
// longest history it resends. Evidence that points to several conversations
// is ambiguous: the request then starts a conversation of its own.
func (c *conversations) resolve(cr *conversationRequest) conversationLink {
	if cr.prevRespID != "" {
		if key, ok := c.responses[cr.prevRespID]; ok {
			return conversationLink{key: c.canonical(key), link: request.ConversationLinkPreviousResponse}
		}
	}

	var byCall []uint64
	for _, r := range cr.results {
		if !distinctiveID(r.ID) {
			continue
		}
		if key, ok := c.calls[r.ID]; ok && !slices.Contains(byCall, c.canonical(key)) {
			byCall = append(byCall, c.canonical(key))
		}
	}
	switch len(byCall) {
	case 0:
	case 1:
		return conversationLink{key: byCall[0], link: request.ConversationLinkCallID}
	default:
		return conversationLink{link: request.ConversationLinkAmbiguous}
	}

	if keys := c.longestPrefix(cr.prefixes, cr.cut); len(keys) > 0 {
		if len(keys) > 1 {
			return conversationLink{link: request.ConversationLinkAmbiguous}
		}
		return conversationLink{key: keys[0], link: request.ConversationLinkHistoryPrefix}
	}

	return conversationLink{link: request.ConversationLinkUnknown}
}

// longestPrefix returns the conversations whose history the request resends
// with the most bytes. A request continues a history only when it appends
// turns to it: the last prefix counts only when the capture cut the request.
func (c *conversations) longestPrefix(prefixes []historyPrefix, cut bool) []uint64 {
	last := len(prefixes) - 1
	if !cut {
		last--
	}
	for i := last; i >= 0; i-- {
		var keys []uint64
		for _, seen := range c.prefixes[prefixes[i].hash] {
			if seen.len == prefixes[i].len && !slices.Contains(keys, c.canonical(seen.key)) {
				keys = append(keys, c.canonical(seen.key))
			}
		}
		if len(keys) > 0 {
			return keys
		}
	}

	return nil
}

// register remembers what later requests of the conversation may resend: the
// request's history, its response and the calls the response requested.
func (c *conversations) register(cr *conversationRequest, key uint64) {
	if n := len(cr.prefixes); n > 0 {
		c.addPrefix(historyPrefix{len: cr.prefixes[n-1].len, hash: cr.prefixes[n-1].hash, key: key})
	}
	if cr.responseID != "" {
		if _, ok := c.responses[cr.responseID]; !ok {
			c.responseIDs = appendBounded(c.responseIDs, cr.responseID, maxConversationResponses, func(id string) {
				delete(c.responses, id)
			})
		}
		c.responses[cr.responseID] = key
	}
	for _, tc := range cr.calls {
		if !distinctiveID(tc.ID) {
			continue
		}
		if _, ok := c.calls[tc.ID]; !ok {
			c.callIDs = appendBounded(c.callIDs, tc.ID, maxConversationCalls, func(id string) {
				delete(c.calls, id)
			})
		}
		c.calls[tc.ID] = key
	}
}

func (c *conversations) addPrefix(prefix historyPrefix) {
	if slices.Contains(c.prefixes[prefix.hash], prefix) {
		return
	}
	c.prefixOrder = appendBounded(c.prefixOrder, prefix, maxConversationPrefixes, func(old historyPrefix) {
		kept := slices.DeleteFunc(c.prefixes[old.hash], func(p historyPrefix) bool { return p == old })
		if len(kept) == 0 {
			delete(c.prefixes, old.hash)
		} else {
			c.prefixes[old.hash] = kept
		}
	})
	c.prefixes[prefix.hash] = append(c.prefixes[prefix.hash], prefix)
}

// appendBounded appends to a FIFO, first dropping its oldest item when full.
func appendBounded[T any](items []T, item T, limit int, drop func(T)) []T {
	if len(items) >= limit {
		drop(items[0])
		items = append(items[:0], items[1:]...)
	}
	return append(items, item)
}

// canonical follows the adoptions of a key.
func (c *conversations) canonical(key uint64) uint64 {
	for range maxConversationAliases {
		next, ok := c.aliases[key]
		if !ok {
			return key
		}
		key = next
	}
	return key
}

func (c *conversations) addUnresolved(cr *conversationRequest, key uint64) {
	u := unresolvedRequest{
		key:        key,
		start:      cr.span.Start,
		prevRespID: cr.prevRespID,
		prefixes:   cr.prefixes,
		cut:        cr.cut,
	}
	for _, r := range cr.results {
		if distinctiveID(r.ID) {
			u.resultIDs = append(u.resultIDs, r.ID)
		}
	}
	if len(c.unresolved) >= maxUnresolvedPerProcess {
		c.unresolved = append(c.unresolved[:0], c.unresolved[1:]...)
	}
	c.unresolved = append(c.unresolved, u)
}

// adopt finds the requests seen earlier with an unknown link that continue
// this one: their conversation is this request's.
func (c *conversations) adopt(cr *conversationRequest, key uint64, now int64) []uint64 {
	var adopted []uint64
	kept := c.unresolved[:0]
	for _, u := range c.unresolved {
		switch {
		case now-u.start > unresolvedConversationTTL.Nanoseconds():
			continue
		case u.key != key && u.start > cr.span.Start && u.continues(cr):
			c.addAlias(u.key, key)
			adopted = append(adopted, u.key)
			continue
		}
		kept = append(kept, u)
	}
	clear(c.unresolved[len(kept):])
	c.unresolved = kept

	return adopted
}

func (c *conversations) addAlias(from, to uint64) {
	if c.canonical(to) == from {
		return
	}
	c.aliasOrder = appendBounded(c.aliasOrder, from, maxConversationAliases, func(old uint64) {
		delete(c.aliases, old)
	})
	c.aliases[from] = to
}

// continues tells whether the unresolved request continues the request cr.
func (u *unresolvedRequest) continues(cr *conversationRequest) bool {
	if u.prevRespID != "" && u.prevRespID == cr.responseID {
		return true
	}
	for _, id := range u.resultIDs {
		if slices.ContainsFunc(cr.calls, func(tc request.ToolCall) bool { return tc.ID == id }) {
			return true
		}
	}

	n := len(cr.prefixes)
	if n == 0 {
		return false
	}
	own := cr.prefixes[n-1]
	last := len(u.prefixes) - 1
	if !u.cut {
		last--
	}
	for i := last; i >= 0; i-- {
		if p := u.prefixes[i]; p.len == own.len && p.hash == own.hash {
			return true
		}
	}

	return false
}

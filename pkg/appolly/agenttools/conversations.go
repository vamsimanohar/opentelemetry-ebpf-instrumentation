// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package agenttools // import "go.opentelemetry.io/obi/pkg/appolly/agenttools"

import (
	"bytes"
	"hash/maphash"
	"slices"
	"strings"
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
	// call ids read from the captured part of a history the capture cut
	maxHistoryCallIDs = 256
	// minRandomIDChars is how many characters of a call id, besides the
	// tool's name, a provider prefix and plain numbers, must look random for
	// the id to tell its conversation apart. Servers that number calls by
	// position ("functions.read_doc:0", "tool_call_000001") issue the same
	// id in every conversation that takes the same steps.
	minRandomIDChars = 8
)

// callIDPrefixes are the fixed prefixes providers and servers put before
// the random part of a call id.
var callIDPrefixes = []string{"chatcmpl-tool-", "functions.", "tooluse_", "toolu_", "call_", "call-", "tool_", "fc_"}

// historyIDKeys name the fields that carry tool-call ids in a history.
var historyIDKeys = [][]byte{
	[]byte(`"id"`), []byte(`"tool_call_id"`), []byte(`"call_id"`), []byte(`"tool_use_id"`), []byte(`"toolUseId"`),
}

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
	calls       map[string]callOwner
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

// callOwner is the conversation whose response issued a call id. An id
// that responses of two conversations issued is shared: it is evidence of
// neither.
type callOwner struct {
	key    uint64
	shared bool
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
		calls:     map[string]callOwner{},
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
	// the capture cut the history: the request has turns after its last
	// prefix, and its prefixes are no evidence. Another conversation that
	// starts with the same system prompt or document shares them.
	cut bool
	// the call ids the captured part of a cut history carries
	historyIDs []string
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
	if cr.cut {
		cr.historyIDs = historyCallIDs(req.history)
	}

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
// is ambiguous: the request then starts a conversation of its own. A history
// the capture cut is no prefix evidence: only the ids its captured part
// carries tie it to a conversation.
func (c *conversations) resolve(cr *conversationRequest) conversationLink {
	if cr.prevRespID != "" {
		if key, ok := c.responses[cr.prevRespID]; ok {
			return conversationLink{key: c.canonical(key), link: request.ConversationLinkPreviousResponse}
		}
	}

	var byCall []uint64
	addOwner := func(id string) {
		if !distinctiveID(id) {
			return
		}
		owner, ok := c.calls[id]
		if !ok || owner.shared {
			return
		}
		if key := c.canonical(owner.key); !slices.Contains(byCall, key) {
			byCall = append(byCall, key)
		}
	}
	for _, r := range cr.results {
		addOwner(r.ID)
	}
	for _, id := range cr.historyIDs {
		addOwner(id)
	}
	switch len(byCall) {
	case 0:
	case 1:
		return conversationLink{key: byCall[0], link: request.ConversationLinkCallID}
	default:
		return conversationLink{link: request.ConversationLinkAmbiguous}
	}

	if cr.cut {
		return conversationLink{link: request.ConversationLinkUnknown}
	}
	if keys := c.longestPrefix(cr.prefixes); len(keys) > 0 {
		if len(keys) > 1 {
			return conversationLink{link: request.ConversationLinkAmbiguous}
		}
		return conversationLink{key: keys[0], link: request.ConversationLinkHistoryPrefix}
	}

	return conversationLink{link: request.ConversationLinkUnknown}
}

// longestPrefix returns the conversations whose history the request resends
// with the most bytes. A request continues a history only when it appends
// turns to it: its own whole history does not count.
func (c *conversations) longestPrefix(prefixes []historyPrefix) []uint64 {
	for i := len(prefixes) - 2; i >= 0; i-- {
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
// request's history, its response and the calls the response requested. The
// history of a cut request is not known as a whole and is not remembered.
func (c *conversations) register(cr *conversationRequest, key uint64) {
	if n := len(cr.prefixes); n > 0 && !cr.cut {
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
		if !conversationCallID(tc.ID, tc.Name) {
			continue
		}
		owner, ok := c.calls[tc.ID]
		if !ok {
			c.callIDs = appendBounded(c.callIDs, tc.ID, maxConversationCalls, func(id string) {
				delete(c.calls, id)
			})
			c.calls[tc.ID] = callOwner{key: key}
			continue
		}
		if !owner.shared && c.canonical(owner.key) != c.canonical(key) {
			c.calls[tc.ID] = callOwner{shared: true}
		}
	}
}

// conversationCallID tells whether a call id can tie a request to the
// conversation whose response issued it: it is distinctive, and not a
// position number decorated with the tool's name.
func conversationCallID(id, name string) bool {
	if !distinctiveID(id) {
		return false
	}
	rest := id
	if name != "" {
		rest = strings.ReplaceAll(rest, name, "")
	}
	for _, prefix := range callIDPrefixes {
		rest = strings.TrimPrefix(rest, prefix)
	}

	random := 0
	for _, run := range strings.FieldsFunc(rest, func(r rune) bool { return !isAlnum(r) }) {
		if strings.ContainsFunc(run, isLetter) {
			random += len(run)
		}
	}
	return random >= minRandomIDChars
}

func isAlnum(r rune) bool {
	return isLetter(r) || (r >= '0' && r <= '9')
}

func isLetter(r rune) bool {
	return (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z')
}

// historyCallIDs returns the tool-call ids of the turns a history carries,
// read from the raw bytes because the capture cut it: the assistant turns
// that requested calls and the results that answered them.
func historyCallIDs(history []byte) []string {
	var ids []string
	for _, key := range historyIDKeys {
		for from := 0; len(ids) < maxHistoryCallIDs; {
			i := bytes.Index(history[from:], key)
			if i < 0 {
				break
			}
			from += i + len(key)
			if id, ok := jsonStringAfterColon(history, from); ok && !slices.Contains(ids, id) {
				ids = append(ids, id)
			}
		}
	}
	return ids
}

// jsonStringAfterColon reads the JSON string value that follows a key
// ending at i, without escapes: ids have none.
func jsonStringAfterColon(raw []byte, i int) (string, bool) {
	i = skipSpace(raw, i)
	if i >= len(raw) || raw[i] != ':' {
		return "", false
	}
	i = skipSpace(raw, i+1)
	if i >= len(raw) || raw[i] != '"' {
		return "", false
	}
	end := bytes.IndexAny(raw[i+1:], `"\`)
	if end < 0 || raw[i+1+end] != '"' {
		return "", false
	}
	return string(raw[i+1 : i+1+end]), true
}

func skipSpace(raw []byte, i int) int {
	for i < len(raw) && strings.IndexByte(jsonSpace, raw[i]) >= 0 {
		i++
	}
	return i
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
	// a cut request's results are cut off, but its captured turns name the
	// calls of the responses it continues
	for _, id := range cr.historyIDs {
		if distinctiveID(id) && !slices.Contains(u.resultIDs, id) {
			u.resultIDs = append(u.resultIDs, id)
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
		if slices.ContainsFunc(cr.calls, func(tc request.ToolCall) bool {
			return tc.ID == id && conversationCallID(tc.ID, tc.Name)
		}) {
			return true
		}
	}

	// a cut history is known neither as a whole nor up to where it continues
	n := len(cr.prefixes)
	if n == 0 || cr.cut || u.cut {
		return false
	}
	own := cr.prefixes[n-1]
	for i := len(u.prefixes) - 2; i >= 0; i-- {
		if p := u.prefixes[i]; p.len == own.len && p.hash == own.hash {
			return true
		}
	}

	return false
}

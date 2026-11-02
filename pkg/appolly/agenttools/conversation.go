// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package agenttools // import "go.opentelemetry.io/obi/pkg/appolly/agenttools"

import (
	"bytes"
	"hash/maphash"
)

// minDistinctiveID is the length from which a call id is taken to identify
// its call on its own. Providers issue ids of 20 to 40 characters; servers
// that number their calls per response ("call_0", "0") do not.
const minDistinctiveID = 12

const jsonSpace = " \t\r\n"

// conversation identifies the history a request sent. Stateless APIs resend
// the whole conversation with every request, so a later request of the same
// conversation starts with the same bytes and appends turns to them; a
// request of another conversation, such as one a tool sends to a model of its
// own, does not.
type conversation struct {
	prefixLen int
	hash      uint64
	// the capture cut the history: any continuation shares the whole prefix
	cut bool
}

func conversationOf(seed maphash.Seed, history []byte) conversation {
	h := bytes.TrimRight(history, jsonSpace)
	if len(h) == 0 || h[0] != '[' {
		return conversation{}
	}

	c := conversation{cut: true}
	if h[len(h)-1] == ']' {
		h = bytes.TrimRight(h[:len(h)-1], jsonSpace)
		c.cut = false
	}
	// an empty history is the start of every conversation
	if len(h) <= 1 {
		return conversation{}
	}

	c.prefixLen = len(h)
	c.hash = maphash.Bytes(seed, h)
	return c
}

func (c *conversation) known() bool {
	return c.prefixLen > 0
}

// continuedBy tells whether history appends turns to the conversation.
func (c *conversation) continuedBy(seed maphash.Seed, history []byte) bool {
	if !c.known() || len(history) < c.prefixLen {
		return false
	}
	if maphash.Bytes(seed, history[:c.prefixLen]) != c.hash {
		return false
	}
	if c.cut {
		return true
	}

	rest := bytes.TrimLeft(history[c.prefixLen:], jsonSpace)
	return len(rest) > 0 && rest[0] == ','
}

// distinctiveID tells whether an id is unlikely to appear in an unrelated
// conversation.
func distinctiveID(id string) bool {
	return len(id) >= minDistinctiveID
}

// containsToken tells whether body holds id as a whole token: not as part of
// a longer identifier.
func containsToken(body []byte, id string) bool {
	if id == "" {
		return false
	}
	tok := []byte(id)
	for from := 0; ; {
		i := bytes.Index(body[from:], tok)
		if i < 0 {
			return false
		}
		start, end := from+i, from+i+len(tok)
		if (start == 0 || !idByte(body[start-1])) && (end == len(body) || !idByte(body[end])) {
			return true
		}
		from = start + 1
	}
}

func idByte(b byte) bool {
	return b == '_' || b == '-' || (b >= '0' && b <= '9') || (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')
}

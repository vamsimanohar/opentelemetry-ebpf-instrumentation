// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package agentact // import "go.opentelemetry.io/obi/pkg/appolly/app/agentact"

import (
	"strings"

	"github.com/gobwas/glob"
)

// PathMatcher matches file paths against exclusion patterns. A pattern
// starting with / is a path prefix. One with *, ? or [ is a glob over the
// whole path, where * also matches /. Any other pattern matches whole path
// components: a file or directory name, or a relative sub-path such as
// "pkg/cache", wherever it appears in the path.
type PathMatcher struct {
	prefixes   []string
	globs      []glob.Glob
	components []string
}

// NewPathMatcher compiles the patterns, skipping invalid ones; validation
// reports those at configuration time.
func NewPathMatcher(patterns []string) PathMatcher {
	var m PathMatcher
	for _, p := range patterns {
		switch {
		case p == "":
		case strings.ContainsAny(p, "*?["):
			if g, err := glob.Compile(p); err == nil {
				m.globs = append(m.globs, g)
			}
		case strings.HasPrefix(p, "/"):
			m.prefixes = append(m.prefixes, p)
		default:
			if c := strings.Trim(p, "/"); c != "" {
				m.components = append(m.components, c)
			}
		}
	}
	return m
}

// Matches tells whether path is excluded.
func (m *PathMatcher) Matches(path string) bool {
	for _, p := range m.prefixes {
		if strings.HasPrefix(path, p) {
			return true
		}
	}
	for _, c := range m.components {
		if hasComponents(path, c) {
			return true
		}
	}
	for _, g := range m.globs {
		if g.Match(path) {
			return true
		}
	}
	return false
}

// hasComponents tells whether the components of sub appear in path,
// starting and ending at component boundaries.
func hasComponents(path, sub string) bool {
	for from := 0; ; {
		i := strings.Index(path[from:], sub)
		if i < 0 {
			return false
		}
		start, end := from+i, from+i+len(sub)
		if (start == 0 || path[start-1] == '/') && (end == len(path) || path[end] == '/') {
			return true
		}
		from = start + 1
	}
}

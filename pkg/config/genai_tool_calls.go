// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package config // import "go.opentelemetry.io/obi/pkg/config"

import (
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/gobwas/glob"
)

// GenAIToolCallsConfig configures the inferred `execute_tool` spans OBI builds
// from the tool calls an LLM response requests and the tool results the agent
// sends back. OBI never observes a tool running, so these spans are marked as
// inferred and their duration is an upper bound. Config v2 only.
type GenAIToolCallsConfig struct {
	// Enabled turns the feature on. It requires HTTP payload capture and at
	// least one GenAI payload extractor.
	Enabled bool `yaml:"enabled" env:"OTEL_EBPF_GENAI_TOOL_CALLS_ENABLED" validate:"boolean"`
	// MaxWindow bounds how long a tool call may stay open without a result.
	MaxWindow time.Duration `yaml:"max_window" env:"OTEL_EBPF_GENAI_TOOL_CALLS_MAX_WINDOW"`
	// OnTimeout tells whether calls reaching MaxWindow are exported or dropped.
	OnTimeout ToolCallTimeoutAction `yaml:"on_timeout" env:"OTEL_EBPF_GENAI_TOOL_CALLS_ON_TIMEOUT"`
	// CloseOnNextRequest closes the pending calls of a response at a later
	// request of the same conversation that carries no parsable result for
	// them: the capture cut it, or the agent encodes results as text. A request
	// continues the conversation when it continues the stateful response,
	// resends the call id, or resends the conversation history that requested
	// the call. Requests of other conversations, such as a tool that calls a
	// model itself, never close a call.
	CloseOnNextRequest bool `yaml:"close_on_next_request" env:"OTEL_EBPF_GENAI_TOOL_CALLS_CLOSE_ON_NEXT_REQUEST"`
	// MatchByName pairs results that carry no call id with the calls of the
	// response whose conversation the request continues, by tool name and
	// order.
	MatchByName bool `yaml:"match_by_name" env:"OTEL_EBPF_GENAI_TOOL_CALLS_MATCH_BY_NAME"`
	// EmitUnmatched exports calls that never received a result once their
	// process exits or MaxWindow passes.
	EmitUnmatched bool `yaml:"emit_unmatched" env:"OTEL_EBPF_GENAI_TOOL_CALLS_EMIT_UNMATCHED"`
	// UnmatchedMinActivity drops unmatched calls whose window saw fewer kernel
	// events than this.
	UnmatchedMinActivity int `yaml:"unmatched_min_activity" env:"OTEL_EBPF_GENAI_TOOL_CALLS_UNMATCHED_MIN_ACTIVITY" validate:"gte=0"`
	// Services selects the instrumented services whose tool calls are
	// inferred. Processes that serve LLM API routes, such as gateways and
	// proxies, forward the tool calls of others and are always left out.
	Services GenAIToolCallsServices `yaml:"services"`
	// KernelActivity attaches the process, file and connection activity of the
	// agent and its descendants observed inside tool windows.
	KernelActivity GenAIToolKernelActivityConfig `yaml:"kernel_activity"`
}

// GenAIToolCallsServices selects services by name, with glob patterns. With
// no Include pattern every service is included; Exclude wins over Include.
// Processes of services matched by an Include pattern report their kernel
// activity from the moment OBI instruments them.
type GenAIToolCallsServices struct {
	Include []string `yaml:"include" env:"OTEL_EBPF_GENAI_TOOL_CALLS_SERVICES_INCLUDE" envSeparator:","`
	Exclude []string `yaml:"exclude" env:"OTEL_EBPF_GENAI_TOOL_CALLS_SERVICES_EXCLUDE" envSeparator:","`
}

// GenAIToolKernelActivityConfig configures the eBPF programs that observe the
// agent process tree while tool windows are open. Only processes known to run
// tool calls, and their descendants, report their execs, file opens and
// connects: a process is known once one of its LLM responses requests a tool
// call. Other instrumented processes report them only during Probation.
type GenAIToolKernelActivityConfig struct {
	// Enabled loads the fork, exec, exit and file-open programs.
	Enabled bool `yaml:"enabled" env:"OTEL_EBPF_GENAI_TOOL_CALLS_KERNEL_ACTIVITY_ENABLED" validate:"boolean"`
	// Probation is how long after it starts an instrumented process reports
	// its activity before it is known to run tool calls, so that the first
	// tool call of a new agent is observed. Zero restricts the reports to
	// known agents, whose first tool window is then incomplete.
	Probation time.Duration `yaml:"probation" env:"OTEL_EBPF_GENAI_TOOL_CALLS_KERNEL_ACTIVITY_PROBATION"`
	Exec      bool          `yaml:"exec" env:"OTEL_EBPF_GENAI_TOOL_CALLS_KERNEL_ACTIVITY_EXEC"`
	// FileOpen reports file opens; Connect reports TCP connects.
	FileOpen bool `yaml:"file_open" env:"OTEL_EBPF_GENAI_TOOL_CALLS_KERNEL_ACTIVITY_FILE_OPEN"`
	Connect  bool `yaml:"connect" env:"OTEL_EBPF_GENAI_TOOL_CALLS_KERNEL_ACTIVITY_CONNECT"`
	// MaxLineageDepth bounds how many generations of descendants are tracked.
	MaxLineageDepth int `yaml:"max_lineage_depth" env:"OTEL_EBPF_GENAI_TOOL_CALLS_KERNEL_ACTIVITY_MAX_LINEAGE_DEPTH" validate:"gte=1,lte=32"`
	// PathExclude lists the paths whose file opens are ignored, such as
	// interpreter imports and the CA bundles that LLM clients reopen. A pattern
	// starting with / is a path prefix; one with *, ? or [ is a glob over the
	// whole path, where * also matches /; any other pattern matches whole path
	// components, such as a file name or a directory name.
	PathExclude []string `yaml:"path_exclude" env:"OTEL_EBPF_GENAI_TOOL_CALLS_KERNEL_ACTIVITY_PATH_EXCLUDE" envSeparator:","`
	// ExcludeLLMEndpoints ignores connections to the endpoints of the
	// process's own GenAI spans: they are the agent's next LLM call.
	ExcludeLLMEndpoints bool `yaml:"exclude_llm_endpoints" env:"OTEL_EBPF_GENAI_TOOL_CALLS_KERNEL_ACTIVITY_EXCLUDE_LLM_ENDPOINTS"`
	// MaxEventsPerWindow caps the span events attached to the spans of one
	// tool window, strongest evidence first; the counters include the rest.
	MaxEventsPerWindow int `yaml:"max_events_per_window" env:"OTEL_EBPF_GENAI_TOOL_CALLS_KERNEL_ACTIVITY_MAX_EVENTS_PER_WINDOW" validate:"gt=0"`
	// MaxEventsPerSpan caps the span events of a single span, strongest
	// evidence first.
	MaxEventsPerSpan int `yaml:"max_events_per_span" env:"OTEL_EBPF_GENAI_TOOL_CALLS_KERNEL_ACTIVITY_MAX_EVENTS_PER_SPAN" validate:"gt=0"`
	// RateLimitPerProcess caps, in the kernel, the events per second of one
	// process tree; bursts of up to ten seconds of budget pass.
	RateLimitPerProcess int `yaml:"rate_limit_per_process" env:"OTEL_EBPF_GENAI_TOOL_CALLS_KERNEL_ACTIVITY_RATE_LIMIT_PER_PROCESS" validate:"gt=0"`
}

// ToolCallTimeoutAction tells what happens to a tool call that reaches the
// maximum window without a result.
type ToolCallTimeoutAction string

const (
	ToolCallTimeoutEmit ToolCallTimeoutAction = "emit"
	ToolCallTimeoutDrop ToolCallTimeoutAction = "drop"
)

// Validate checks the settings that cross field boundaries.
func (c *GenAIToolCallsConfig) Validate(httpBufferSize uint32, genAI *GenAIConfig) error {
	if !c.Enabled {
		return nil
	}
	if httpBufferSize == 0 {
		return errors.New("genai tool calls require HTTP payload capture: set a non-zero HTTP buffer size")
	}
	if !genAI.ProvidersEnabled() {
		return errors.New("genai tool calls require at least one GenAI payload extractor")
	}
	if c.OnTimeout != ToolCallTimeoutEmit && c.OnTimeout != ToolCallTimeoutDrop {
		return errors.New(`genai tool calls on_timeout must be "emit" or "drop"`)
	}
	if c.MaxWindow <= 0 {
		return errors.New("genai tool calls max_window must be positive")
	}
	for _, pattern := range slices.Concat(c.Services.Include, c.Services.Exclude) {
		if _, err := glob.Compile(pattern); err != nil {
			return fmt.Errorf("genai tool calls services: invalid pattern %q: %w", pattern, err)
		}
	}
	if c.KernelActivity.Enabled {
		ka := &c.KernelActivity
		if ka.Probation < 0 {
			return errors.New("genai tool calls kernel_activity probation must not be negative")
		}
		if ka.MaxLineageDepth < 1 || ka.MaxEventsPerWindow < 1 || ka.MaxEventsPerSpan < 1 || ka.RateLimitPerProcess < 1 {
			return errors.New("genai tool calls kernel_activity max_lineage_depth, max_events_per_window, " +
				"max_events_per_span and rate_limit_per_process must be positive")
		}
		for _, pattern := range ka.PathExclude {
			if _, err := glob.Compile(pattern); err != nil {
				return fmt.Errorf("genai tool calls kernel_activity path_exclude: invalid pattern %q: %w", pattern, err)
			}
		}
	}

	return nil
}

// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bpfcore/vmlinux.h>

#include <common/connection_info.h>

// Kernel activity of instrumented processes and their descendants, reported
// to bound and attribute inferred agent tool calls. These values must match
// agentact.Kind in pkg/appolly/app/agentact/event.go.
enum agent_activity_kind : u8 {
    k_agent_activity_fork = 1,
    k_agent_activity_exec = 2,
    k_agent_activity_exit = 3,
    k_agent_activity_open = 4,
    k_agent_activity_connect = 5,
    // events of a root lost to the rate limit or a full ring
    k_agent_activity_drops = 6,
};

enum agent_activity_flag : u8 {
    // the path or the argv did not fit in buf
    k_agent_activity_truncated = 1 << 0,
    // the opened path is relative to the working directory or a dirfd
    k_agent_activity_relative_path = 1 << 1,
};

enum agent_activity_sizes : u16 {
    k_agent_activity_path_len = 256,
    k_agent_activity_args_len = 512,
};

typedef struct agent_activity_event {
    u8 kind;
    u8 flags;
    // bytes of buf holding the path, which starts at buf[0], and the
    // NUL-separated argv of an exec, which starts at buf[k_agent_activity_path_len]
    u16 path_len;
    u16 args_len;
    u8 _pad[2];
    u64 ts;
    // the start time of root_pid, CLOCK_MONOTONIC
    u64 root_start;
    // for drops, the time of the first event lost
    u64 dropped_since;
    // host ids
    u32 pid;
    u32 tid;
    // the process itself when it is instrumented, else its nearest
    // instrumented ancestor
    u32 root_pid;
    u32 parent_pid;
    u32 spawner_tid;
    // the fd or -errno of an open, the wait status of an exit, the events
    // lost for drops
    s32 ret;
    u32 open_flags;
    connection_info_t conn;
    unsigned char buf[k_agent_activity_path_len + k_agent_activity_args_len];
} agent_activity_event_t;

// The part of an event before buf, which is all that fork, exit, connect and
// drops events carry.
enum { k_agent_activity_header_size = __builtin_offsetof(agent_activity_event_t, buf) };

typedef struct agent_lineage {
    u32 root_pid;
    u32 depth;
    u64 root_start;
} agent_lineage_t;

typedef struct agent_rate {
    u64 last_refill;
    u64 tokens;
    // events lost since the last drops event, and when the first one was
    u64 dropped;
    u64 dropped_since;
} agent_rate_t;

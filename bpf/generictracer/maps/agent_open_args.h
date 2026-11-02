// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bpfcore/vmlinux.h>
#include <bpfcore/bpf_helpers.h>

#include <common/agent_activity.h>
#include <common/map_sizing.h>
#include <common/pin_internal.h>

// Open events between the entry of the open, which reads the user path, and
// its return, which knows the result. Keyed by pid_tgid.
struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(max_entries, 1024);
    __type(key, u64);
    __type(value, agent_activity_event_t);
    __uint(pinning, OBI_PIN_INTERNAL);
} agent_open_args SEC(".maps");

// The entries of agent_open_args, so that the return of an open with nothing
// recorded, the case of every untracked process on the host, exits early.
s64 agent_open_entries = 0;

static __always_inline bool agent_open_pending(void) {
    return agent_open_entries > 0;
}

static __always_inline void agent_open_pending_add(const s64 n) {
    __sync_fetch_and_add(&agent_open_entries, n);
}

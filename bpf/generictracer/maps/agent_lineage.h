// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bpfcore/vmlinux.h>
#include <bpfcore/bpf_helpers.h>

#include <common/agent_activity.h>
#include <common/pin_internal.h>

enum { k_agent_lineage_max_entries = 1 << 14 };

// Descendants of instrumented processes, by host tgid. The fork chain is
// recorded, not the parent pointer, so a daemonized grandchild stays tracked.
struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(max_entries, k_agent_lineage_max_entries);
    __type(key, u32);
    __type(value, agent_lineage_t);
    __uint(pinning, OBI_PIN_INTERNAL);
} agent_lineage SEC(".maps");

// Per-root event budget, and the events it lost, by root pid.
struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(max_entries, 4096);
    __type(key, u32);
    __type(value, agent_rate_t);
    __uint(pinning, OBI_PIN_INTERNAL);
} agent_rate SEC(".maps");

// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bpfcore/vmlinux.h>
#include <bpfcore/bpf_helpers.h>

#include <common/pin_internal.h>

enum { k_agent_roots_max_entries = 4096 };

// Instrumented processes that user space knows to run agent tool calls, by
// host tgid. They and their descendants report their execs, file opens and
// connects; other instrumented processes only during their probation.
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, k_agent_roots_max_entries);
    __type(key, u32);
    __type(value, u8);
    __uint(pinning, OBI_PIN_INTERNAL);
} agent_roots SEC(".maps");

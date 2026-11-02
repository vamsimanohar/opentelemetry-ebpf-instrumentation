// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bpfcore/vmlinux.h>
#include <bpfcore/bpf_helpers.h>
#include <bpfcore/bpf_core_read.h>

#include <common/agent_activity.h>
#include <common/sockaddr.h>

#include <generictracer/maps/agent_activity_events.h>
#include <generictracer/maps/agent_lineage.h>
#include <generictracer/maps/agent_roots.h>

#include <pid/pid.h>

// Set from user space when inferred agent tool calls observe kernel activity.
// Fork and exit tracking is the base; exec, open and connect are optional.
volatile const bool agent_activity_enabled = false;
volatile const bool agent_activity_exec_enabled = false;
volatile const bool agent_activity_open_enabled = false;
volatile const bool agent_activity_connect_enabled = false;
volatile const u32 agent_activity_max_depth = 8;
volatile const u32 agent_activity_rate_limit = 2000;
// How long after it started an instrumented process that user space has not
// armed reports its execs, opens and connects.
volatile const u64 agent_activity_probation_ns = 0;

enum {
    k_ns_per_second = 1000000000,
    // a process may spend this many seconds of budget at once: interpreter
    // imports open thousands of files right before the first tool call
    k_agent_rate_burst_seconds = 10,
    k_agent_drops_max = 0x7fffffff,
};

typedef struct agent_root {
    u64 start;
    u32 pid;
    u8 _pad[4];
} agent_root_t;

static __always_inline void agent_activity_count_drop(void) {
    const u32 zero = 0;
    u64 *drops = bpf_map_lookup_elem(&agent_activity_drops, &zero);
    if (drops) {
        *drops += 1;
    }
}

// The root a process reports its activity for: its instrumented ancestor
// recorded at fork, else the process itself when it is instrumented. False
// for any other process. Unlike pid_in_filter(), descendants of any depth
// pass. An instrumented process below another one, such as a pool worker
// OBI instruments too, stays in its ancestor's tree: user space tells which
// process of the tree runs the tool calls.
static __always_inline bool agent_root(const u64 id, agent_root_t *root) {
    const u32 tgid = id >> 32;

    const agent_lineage_t *lineage = bpf_map_lookup_elem(&agent_lineage, &tgid);
    if (lineage) {
        root->pid = lineage->root_pid;
        root->start = lineage->root_start;
        return true;
    }

    const u32 pid = obi_pid(id);
    if (pid != 0 && pid_selected(&valid_pids, pid)) {
        const struct task_struct *task = (const struct task_struct *)bpf_get_current_task();
        root->pid = tgid;
        root->start = BPF_CORE_READ(task, group_leader, start_time);
        return true;
    }

    return false;
}

// Whether a root reports its execs, opens and connects: user space armed it,
// or it started less than the probation period ago.
static __always_inline bool agent_root_observed(const agent_root_t *root, const u64 now) {
    if (bpf_map_lookup_elem(&agent_roots, &root->pid)) {
        return true;
    }
    return agent_activity_probation_ns > 0 && now - root->start <= agent_activity_probation_ns;
}

// Reserves an event without its buffer, the shape of fork, exit, connect and
// drops events.
static __always_inline agent_activity_event_t *agent_activity_reserve_header(
    const u8 kind, const u64 id, const agent_root_t *root, const u64 now) {
    agent_activity_event_t *e =
        bpf_ringbuf_reserve(&agent_activity_events, k_agent_activity_header_size, 0);
    if (!e) {
        return NULL;
    }

    __builtin_memset(e, 0, k_agent_activity_header_size);
    e->kind = kind;
    e->pid = id >> 32;
    e->tid = (u32)id;
    e->root_pid = root->pid;
    e->root_start = root->start;
    e->ts = now;

    return e;
}

static __always_inline void agent_rate_lost(agent_rate_t *rate, const u64 now) {
    agent_activity_count_drop();
    if (rate->dropped == 0) {
        rate->dropped_since = now;
    }
    rate->dropped++;
}

// Tells user space how many events of the root were lost, and since when, so
// that the windows they fell in are not reported as complete.
static __always_inline void
agent_rate_report(agent_rate_t *rate, const agent_root_t *root, const u64 now) {
    if (rate->dropped == 0) {
        return;
    }

    const u64 id = (u64)root->pid << 32 | root->pid;
    agent_activity_event_t *e =
        agent_activity_reserve_header(k_agent_activity_drops, id, root, now);
    if (!e) {
        // reported with the next event
        return;
    }
    e->dropped_since = rate->dropped_since;
    e->ret = rate->dropped > k_agent_drops_max ? k_agent_drops_max : (s32)rate->dropped;
    bpf_ringbuf_submit(e, 0);

    rate->dropped = 0;
    rate->dropped_since = 0;
}

// Reports the losses of a root on the events that are not rate limited.
static __always_inline void agent_rate_flush(const agent_root_t *root, const u64 now) {
    agent_rate_t *rate = bpf_map_lookup_elem(&agent_rate, &root->pid);
    if (rate) {
        agent_rate_report(rate, root, now);
    }
}

// Takes a token of the root's bucket, refilled at agent_activity_rate_limit
// events per second, so a file-open storm costs little while bursts pass.
// NULL when the event is dropped; the bucket otherwise, to count an event the
// ring then has no room for. Concurrent updates from several CPUs may let a
// few extra events through.
static __always_inline agent_rate_t *agent_rate_take(const agent_root_t *root, const u64 now) {
    const u64 burst = (u64)agent_activity_rate_limit * k_agent_rate_burst_seconds;

    agent_rate_t *rate = bpf_map_lookup_elem(&agent_rate, &root->pid);
    if (!rate) {
        const agent_rate_t fresh = {.last_refill = now, .tokens = burst};
        bpf_map_update_elem(&agent_rate, &root->pid, &fresh, BPF_NOEXIST);
        rate = bpf_map_lookup_elem(&agent_rate, &root->pid);
        if (!rate) {
            agent_activity_count_drop();
            return NULL;
        }
    }

    // a bounded elapsed time keeps the product below 2^64
    u64 elapsed = now - rate->last_refill;
    if (elapsed > (u64)k_agent_rate_burst_seconds * k_ns_per_second) {
        elapsed = (u64)k_agent_rate_burst_seconds * k_ns_per_second;
    }
    const u64 refill = elapsed * agent_activity_rate_limit / k_ns_per_second;
    if (refill > 0) {
        rate->tokens = rate->tokens + refill > burst ? burst : rate->tokens + refill;
        rate->last_refill = now;
    }

    if (rate->tokens == 0) {
        agent_rate_lost(rate, now);
        return NULL;
    }

    rate->tokens--;
    agent_rate_report(rate, root, now);
    return rate;
}

// Called from the existing tcp_connect probe, before its own pid filter: tool
// processes are usually not instrumented themselves.
static __always_inline void agent_activity_on_connect(struct sock *sk) {
    if (!agent_activity_connect_enabled) {
        return;
    }

    const u64 id = bpf_get_current_pid_tgid();
    agent_root_t root = {};
    if (!agent_root(id, &root)) {
        return;
    }

    const u64 now = bpf_ktime_get_ns();
    if (!agent_root_observed(&root, now)) {
        return;
    }
    agent_rate_t *rate = agent_rate_take(&root, now);
    if (!rate) {
        return;
    }

    agent_activity_event_t *e =
        agent_activity_reserve_header(k_agent_activity_connect, id, &root, now);
    if (!e) {
        agent_rate_lost(rate, now);
        return;
    }

    if (!parse_sock_info(sk, &e->conn)) {
        bpf_ringbuf_discard(e, 0);
        return;
    }

    bpf_ringbuf_submit(e, 0);
}

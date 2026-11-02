// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//go:build obi_bpf_ignore
#include <bpfcore/vmlinux.h>
#include <bpfcore/bpf_helpers.h>
#include <bpfcore/bpf_tracing.h>

#include <common/sockaddr.h>

#include <logger/bpf_dbg.h>

#include <pid/pid.h>

char __license[] SEC("license") = "Dual MIT/GPL";

#define WATCH_BIND 0x1
#define WATCH_EXEC 0x2

typedef struct watch_info {
    u64 flags; // Must be first, we use it to tell what kind of packet we have on the ring buffer
    u64 payload;
} watch_info_t;

const watch_info_t *unused_2 __attribute__((unused));

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    // room for a burst of exec events while userspace catches up
    __uint(max_entries, 1 << 16);
} watch_events SEC(".maps");

SEC("kprobe/sys_bind")
int obi_kprobe_sys_bind(struct pt_regs *ctx) {
    // unwrap the args because it's a sys call
    struct pt_regs *__ctx = (struct pt_regs *)PT_REGS_PARM1(ctx);
    void *addr;
    bpf_probe_read(&addr, sizeof(void *), (void *)&PT_REGS_PARM2(__ctx));

    if (!addr) {
        return 0;
    }

    const u16 port = get_sockaddr_port_user(addr);

    if (!port) {
        return 0;
    }

    watch_info_t *trace = bpf_ringbuf_reserve(&watch_events, sizeof(watch_info_t), 0);
    if (trace) {
        trace->flags = WATCH_BIND;
        trace->payload = port;
        bpf_dbg_printk("New port bound, payload=%d", trace->payload);

        bpf_ringbuf_submit(trace, 0);
    }

    return 0;
}

// The process has its new program image: its executable and command line can
// be matched right away. Attached only when discovery reacts to exec events.
SEC("tracepoint/sched/sched_process_exec")
int obi_tp_sched_process_exec(void *ctx) {
    (void)ctx;

    // the pid as OBI's /proc numbers it; 0 for a task outside that namespace
    const u32 pid = obi_pid(bpf_get_current_pid_tgid());
    if (!pid) {
        return 0;
    }

    watch_info_t *trace = bpf_ringbuf_reserve(&watch_events, sizeof(watch_info_t), 0);
    if (trace) {
        trace->flags = WATCH_EXEC;
        trace->payload = pid;

        bpf_ringbuf_submit(trace, 0);
    }

    return 0;
}

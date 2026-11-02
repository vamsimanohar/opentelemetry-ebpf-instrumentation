// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//go:build obi_bpf_ignore

#include <bpfcore/vmlinux.h>
#include <bpfcore/bpf_helpers.h>
#include <bpfcore/bpf_core_read.h>
#include <bpfcore/bpf_tracing.h>

#include <common/agent_activity.h>
#include <common/preempt_guard.h>
#include <common/scratch_mem.h>
#include <common/strings.h>

#include <generictracer/agent_activity.h>
#include <generictracer/maps/agent_activity_events.h>
#include <generictracer/maps/agent_lineage.h>
#include <generictracer/maps/agent_open_args.h>
#include <generictracer/maps/agent_roots.h>

// Open and exec events are built here: they do not fit the stack.
SCRATCH_MEM_TYPED(agent_open_event, agent_activity_event_t)
SCRATCH_MEM_TYPED(agent_exec_event, agent_activity_event_t)

// Pseudo filesystems that runtimes read for introspection, never tool work.
static __always_inline bool agent_ignored_path(const unsigned char *path) {
    const char *p = (const char *)path;
    return obi_bpf_memcmp(p, "/proc/", sizeof("/proc/") - 1) == 0 ||
           obi_bpf_memcmp(p, "/sys/", sizeof("/sys/") - 1) == 0 ||
           obi_bpf_memcmp(p, "/dev/", sizeof("/dev/") - 1) == 0;
}

// The raw tracepoint, unlike the classic one, hands over the child task, so
// its tgid tells a new process from a new thread. Lineage is recorded for the
// whole life of the instrumented process, not only inside tool windows: pool
// workers are forked before the first window and reused by later ones.
SEC("raw_tracepoint/sched_process_fork")
int GUARDED_PROG(obi_raw_tp_agent_fork, struct bpf_raw_tracepoint_args *, ctx) {
    if (!agent_activity_enabled) {
        return 0;
    }

    const struct task_struct *child = (const struct task_struct *)ctx->args[1];
    const u32 child_pid = BPF_CORE_READ(child, pid);
    const u32 child_tgid = BPF_CORE_READ(child, tgid);
    if (child_pid != child_tgid) {
        return 0;
    }

    // the forking task is the current one
    const u64 id = bpf_get_current_pid_tgid();
    const u32 parent = id >> 32;
    agent_root_t root = {};
    if (!agent_root(id, &root)) {
        return 0;
    }

    u32 depth = 0;
    const agent_lineage_t *parent_lineage = bpf_map_lookup_elem(&agent_lineage, &parent);
    if (parent_lineage) {
        depth = parent_lineage->depth;
    }
    if (depth >= agent_activity_max_depth) {
        return 0;
    }

    const agent_lineage_t lineage = {
        .root_pid = root.pid, .depth = depth + 1, .root_start = root.start};
    bpf_map_update_elem(&agent_lineage, &child_tgid, &lineage, BPF_ANY);

    const u64 now = bpf_ktime_get_ns();
    agent_rate_flush(&root, now);
    agent_activity_event_t *e =
        agent_activity_reserve_header(k_agent_activity_fork, id, &root, now);
    if (!e) {
        agent_activity_count_drop();
        return 0;
    }
    e->pid = child_tgid;
    e->tid = child_tgid;
    e->parent_pid = parent;
    e->spawner_tid = (u32)id;

    bpf_ringbuf_submit(e, 0);
    return 0;
}

// Only descendants report their execs: the instrumented process replacing
// its own image is not tool work. The argv is read from the new mm, where
// exec has just copied it.
SEC("raw_tracepoint/sched_process_exec")
int GUARDED_PROG(obi_raw_tp_agent_exec, struct bpf_raw_tracepoint_args *, ctx) {
    if (!agent_activity_exec_enabled) {
        return 0;
    }

    const u64 id = bpf_get_current_pid_tgid();
    const u32 tgid = id >> 32;
    const agent_lineage_t *lineage = bpf_map_lookup_elem(&agent_lineage, &tgid);
    if (!lineage) {
        return 0;
    }
    const agent_root_t root = {.pid = lineage->root_pid, .start = lineage->root_start};

    const u64 now = bpf_ktime_get_ns();
    if (!agent_root_observed(&root, now)) {
        return 0;
    }

    agent_activity_event_t *e = agent_exec_event_mem();
    if (!e) {
        return 0;
    }
    __builtin_memset(e, 0, sizeof(*e));
    e->kind = k_agent_activity_exec;
    e->pid = tgid;
    e->tid = (u32)id;
    e->root_pid = root.pid;
    e->root_start = root.start;
    e->ts = now;

    const struct task_struct *task = (const struct task_struct *)bpf_get_current_task();
    e->parent_pid = BPF_CORE_READ(task, real_parent, tgid);

    const struct linux_binprm *bprm = (const struct linux_binprm *)ctx->args[2];
    const long path_len =
        bpf_probe_read_kernel_str(e->buf, k_agent_activity_path_len, BPF_CORE_READ(bprm, filename));
    if (path_len > 0) {
        // without the terminating NUL
        e->path_len = path_len - 1;
    }

    const unsigned long arg_start = BPF_CORE_READ(task, mm, arg_start);
    const unsigned long arg_end = BPF_CORE_READ(task, mm, arg_end);
    u64 args_len = arg_end > arg_start ? arg_end - arg_start : 0;
    if (args_len > k_agent_activity_args_len) {
        args_len = k_agent_activity_args_len;
        e->flags |= k_agent_activity_truncated;
    }
    if (args_len > 0 && bpf_probe_read_user(e->buf + k_agent_activity_path_len,
                                            args_len,
                                            (const void *)arg_start) == 0) {
        e->args_len = args_len;
    }

    // only the used part of the buffer: the path, then the argv after it
    u32 size = k_agent_activity_header_size + k_agent_activity_path_len + e->args_len;
    if (e->args_len == 0) {
        size = k_agent_activity_header_size + e->path_len;
    }
    if (size > sizeof(*e)) {
        size = sizeof(*e);
    }
    if (bpf_ringbuf_output(&agent_activity_events, e, size, 0) != 0) {
        agent_activity_count_drop();
    }
    return 0;
}

// One event per exiting process, from its group leader, carrying the wait
// status. A second program on this tracepoint: nodejs.c has its own.
SEC("raw_tracepoint/sched_process_exit")
int GUARDED_PROG(obi_raw_tp_agent_exit, struct bpf_raw_tracepoint_args *, ctx) {
    (void)ctx;

    if (!agent_activity_enabled) {
        return 0;
    }

    const u64 id = bpf_get_current_pid_tgid();
    const u32 tgid = id >> 32;
    if (tgid != (u32)id) {
        return 0;
    }

    agent_root_t root = {};
    if (!agent_root(id, &root)) {
        return 0;
    }

    const u64 now = bpf_ktime_get_ns();
    agent_rate_flush(&root, now);
    agent_activity_event_t *e =
        agent_activity_reserve_header(k_agent_activity_exit, id, &root, now);
    if (e) {
        const struct task_struct *task = (const struct task_struct *)bpf_get_current_task();
        e->ret = BPF_CORE_READ(task, exit_code);
        bpf_ringbuf_submit(e, 0);
    } else {
        agent_activity_count_drop();
    }

    bpf_map_delete_elem(&agent_lineage, &tgid);
    if (root.pid == tgid) {
        // a later process reusing the pid is not known to be an agent
        bpf_map_delete_elem(&agent_roots, &tgid);
        bpf_map_delete_elem(&agent_rate, &tgid);
    }
    return 0;
}

// Fills an open event of a tracked process. False when the process is not
// tracked or the path is ignored.
static __always_inline bool
agent_open_fill(agent_activity_event_t *e, const u64 id, const char *filename, const u32 flags) {
    agent_root_t root = {};
    if (!agent_root(id, &root)) {
        return false;
    }

    const u64 now = bpf_ktime_get_ns();
    if (!agent_root_observed(&root, now)) {
        return false;
    }

    __builtin_memset(e, 0, k_agent_activity_header_size);
    const long path_len = bpf_probe_read_user_str(e->buf, k_agent_activity_path_len, filename);
    if (path_len <= 0 || agent_ignored_path(e->buf)) {
        return false;
    }

    e->kind = k_agent_activity_open;
    e->pid = id >> 32;
    e->tid = (u32)id;
    e->root_pid = root.pid;
    e->root_start = root.start;
    e->ts = now;
    e->path_len = path_len - 1;
    if (path_len == k_agent_activity_path_len) {
        e->flags |= k_agent_activity_truncated;
    }
    if (e->buf[0] != '/') {
        e->flags |= k_agent_activity_relative_path;
    }
    e->open_flags = flags;

    return true;
}

// Sends an open event with the result of the open: the header and the path.
static __always_inline void agent_open_submit(agent_activity_event_t *e, const long ret) {
    const agent_root_t root = {.pid = e->root_pid, .start = e->root_start};
    agent_rate_t *rate = agent_rate_take(&root, e->ts);
    if (!rate) {
        return;
    }

    e->ret = ret;
    u32 size = k_agent_activity_header_size + e->path_len;
    if (size > k_agent_activity_header_size + k_agent_activity_path_len) {
        size = k_agent_activity_header_size + k_agent_activity_path_len;
    }
    if (bpf_ringbuf_output(&agent_activity_events, e, size, 0) != 0) {
        agent_rate_lost(rate, e->ts);
    }
}

// do_sys_openat2 serves open, openat, creat and openat2. On return it still
// has its arguments, and the user path it has just read is still valid.
// Through the BPF trampoline, this costs every open on the host far less than
// a kprobe and a kretprobe.
SEC("fexit/do_sys_openat2")
int BPF_PROG_GUARDED(
    obi_fexit_agent_openat2, int dfd, const char *filename, const struct open_how *how, long ret) {
    (void)dfd;

    if (!agent_activity_open_enabled) {
        return 0;
    }

    agent_activity_event_t *e = agent_open_event_mem();
    if (!e) {
        return 0;
    }
    if (agent_open_fill(e, bpf_get_current_pid_tgid(), filename, BPF_CORE_READ(how, flags))) {
        agent_open_submit(e, ret);
    }
    return 0;
}

// Without BPF trampolines, a kprobe reads the path on entry, while the user
// buffer is known to be valid, and a kretprobe sends the event with the
// result. An entry that misses its return is replaced by the next open of
// its thread.
SEC("kprobe/do_sys_openat2")
int BPF_KPROBE_GUARDED(obi_kprobe_agent_openat2,
                       int dfd,
                       const char *filename,
                       const struct open_how *how) {
    (void)dfd;

    if (!agent_activity_open_enabled) {
        return 0;
    }

    const u64 id = bpf_get_current_pid_tgid();
    agent_activity_event_t *e = agent_open_event_mem();
    if (!e) {
        return 0;
    }
    if (!agent_open_fill(e, id, filename, BPF_CORE_READ(how, flags))) {
        // a stale entry of this thread must not take the result of this open
        if (agent_open_pending() && bpf_map_delete_elem(&agent_open_args, &id) == 0) {
            agent_open_pending_add(-1);
        }
        return 0;
    }

    if (bpf_map_update_elem(&agent_open_args, &id, e, BPF_NOEXIST) == 0) {
        agent_open_pending_add(1);
    } else {
        bpf_map_update_elem(&agent_open_args, &id, e, BPF_ANY);
    }
    return 0;
}

SEC("kretprobe/do_sys_openat2")
int BPF_KRETPROBE_GUARDED(obi_kretprobe_agent_openat2, long ret) {
    // every open on the host returns here: most find nothing recorded
    if (!agent_open_pending()) {
        return 0;
    }

    const u64 id = bpf_get_current_pid_tgid();
    agent_activity_event_t *e = bpf_map_lookup_elem(&agent_open_args, &id);
    if (!e) {
        return 0;
    }

    agent_open_submit(e, ret);
    if (bpf_map_delete_elem(&agent_open_args, &id) == 0) {
        agent_open_pending_add(-1);
    }
    return 0;
}

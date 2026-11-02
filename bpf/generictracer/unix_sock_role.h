// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bpfcore/vmlinux.h>
#include <bpfcore/bpf_helpers.h>

#include <common/protocol_defs.h>

// Unix sockets carry no ports, so the end that receives first is taken as the
// server. That only holds for an end accepted from a listening socket, which
// inherits the listener's address. The ends of a socketpair() (e.g. a child's
// stdio in Node.js/libuv) and a connecting client have no address
// (unix_unbound): their first read is not an incoming request, and taking it
// for one would parent the reading thread's later client calls to a server
// span that is never answered.
// orig_dport is also 0 for TLS connections whose socket was not resolved.
static __always_inline bool is_unix_sock_server(u8 direction, u16 orig_dport, u8 unix_unbound) {
    return direction == TCP_RECV && orig_dport == 0 && !unix_unbound;
}

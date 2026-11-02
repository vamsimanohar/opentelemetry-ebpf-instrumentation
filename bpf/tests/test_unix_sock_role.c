// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Run from repo root:
//   make -C bpf/tests test_unix_sock_role && bpf/tests/test_unix_sock_role
// Run from bpf/tests:
//   make test_unix_sock_role && ./test_unix_sock_role

#include <stdbool.h>
#include <stdio.h>

#include <generictracer/unix_sock_role.h>

enum : u16 { k_tcp_port = 8080 };
enum : u8 { k_bound = 0, k_unbound = 1 };

static unsigned int failed_assertions;

static void assert_true(bool condition, const char *message) {
    if (condition) {
        printf("PASS: %s\n", message);
        return;
    }

    failed_assertions++;
    printf("FAIL: %s\n", message);
}

// php-fpm behind nginx: the accepted end carries the listener's address
static void test_accepted_end_receiving_first_is_server(void) {
    assert_true(is_unix_sock_server(TCP_RECV, 0, k_bound),
                "an accepted unix socket end that receives first is a server");
}

// Node.js child_process.spawn: the parent reads the child's stdout first
static void test_socketpair_end_receiving_first_is_not_server(void) {
    assert_true(!is_unix_sock_server(TCP_RECV, 0, k_unbound),
                "an unbound unix socket end (socketpair, client) that receives first is not a "
                "server");
}

static void test_sending_first_is_never_server(void) {
    assert_true(!is_unix_sock_server(TCP_SEND, 0, k_bound),
                "a bound unix socket end that sends first is not a server");
    assert_true(!is_unix_sock_server(TCP_SEND, 0, k_unbound),
                "an unbound unix socket end that sends first is not a server");
}

static void test_tcp_connections_are_not_classified_here(void) {
    assert_true(!is_unix_sock_server(TCP_RECV, k_tcp_port, k_bound),
                "a connection with a destination port is not a unix socket");
}

int main(void) {
    test_accepted_end_receiving_first_is_server();
    test_socketpair_end_receiving_first_is_not_server();
    test_sending_first_is_never_server();
    test_tcp_connections_are_not_classified_here();

    if (failed_assertions) {
        printf("%u assertion(s) failed\n", failed_assertions);
        return 1;
    }

    printf("all unix socket role tests passed\n");
    return 0;
}

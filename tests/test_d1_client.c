/* Model-free tests for the /v1/systemone client used by ds4-agent's d1 gate.
 * A fake server thread answers one connection per test with a canned reply. */
#include "../ds4_d1.h"

#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

static int failures;

#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); failures++; } \
} while (0)

typedef struct {
    int listen_fd;
    const char *reply;      /* NULL: accept, read, never answer */
    char request[8192];
} fake_server;

static void *fake_server_main(void *arg) {
    fake_server *s = arg;
    int fd = accept(s->listen_fd, NULL, NULL);
    if (fd < 0) return NULL;
    size_t len = 0;
    for (;;) {
        ssize_t n = recv(fd, s->request + len, sizeof(s->request) - 1 - len, 0);
        if (n <= 0) break;
        len += (size_t)n;
        s->request[len] = '\0';
        char *body = strstr(s->request, "\r\n\r\n");
        const char *cl = strstr(s->request, "Content-Length: ");
        if (body && cl && strlen(body + 4) >= (size_t)atoi(cl + 16)) break;
    }
    if (s->reply) {
        (void)!write(fd, s->reply, strlen(s->reply));
    } else {
        usleep(400 * 1000);
    }
    close(fd);
    return NULL;
}

static int listen_tcp(int *port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in a = {0};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    socklen_t al = sizeof(a);
    if (bind(fd, (struct sockaddr *)&a, sizeof(a)) || listen(fd, 1) ||
        getsockname(fd, (struct sockaddr *)&a, &al)) return -1;
    *port = ntohs(a.sin_port);
    return fd;
}

/* Run one noul request against a fake TCP server returning `reply`. */
static int ask_tcp(const char *reply, int timeout_ms, double *p, char *err,
                   size_t err_len, fake_server *s) {
    int port = 0;
    memset(s, 0, sizeof(*s));
    s->listen_fd = listen_tcp(&port);
    s->reply = reply;
    pthread_t th;
    pthread_create(&th, NULL, fake_server_main, s);
    char spec[32];
    snprintf(spec, sizeof(spec), "127.0.0.1:%d", port);
    ds4_d1_endpoint ep;
    CHECK(ds4_d1_parse_endpoint(spec, &ep, err, err_len));
    int ok = ds4_d1_noul(&ep, "cwd: /tmp\n\"quoted\" \\ state", "Is it risky?",
                         timeout_ms, p, err, err_len);
    pthread_join(th, NULL);
    close(s->listen_fd);
    return ok;
}

static void test_endpoints(void) {
    ds4_d1_endpoint ep;
    char err[160];
    CHECK(ds4_d1_parse_endpoint("8080", &ep, err, sizeof(err)));
    CHECK(!strcmp(ep.host, "127.0.0.1") && ep.port == 8080 && !ep.unix_path[0]);
    CHECK(ds4_d1_parse_endpoint("localhost:9", &ep, err, sizeof(err)));
    CHECK(!strcmp(ep.host, "localhost") && ep.port == 9);
    CHECK(ds4_d1_parse_endpoint("unix:/tmp/d1.sock", &ep, err, sizeof(err)));
    CHECK(!strcmp(ep.unix_path, "/tmp/d1.sock"));
    CHECK(ds4_d1_parse_endpoint("/tmp/x.sock", &ep, err, sizeof(err)));
    CHECK(!ds4_d1_parse_endpoint("", &ep, err, sizeof(err)));
    CHECK(!ds4_d1_parse_endpoint("host:", &ep, err, sizeof(err)));
    CHECK(!ds4_d1_parse_endpoint(":80", &ep, err, sizeof(err)));
    CHECK(!ds4_d1_parse_endpoint("70000", &ep, err, sizeof(err)));
}

static void test_response_layouts(void) {
    double p = -1;
    char err[160];
    CHECK(ds4_d1_response_noul(
        "{\"answers\":{\"other\":{\"type\":\"choice\",\"choice\":\"a\",\"probabilities\":{\"a\":1}},"
        "\"q\":{\"type\":\"noul\",\"noul\":0.25}},\"usage\":{\"output_tokens\":0}}",
        "q", &p, err, sizeof(err)));
    CHECK(p == 0.25);
    CHECK(ds4_d1_response_noul("{\"q\":{\"type\":\"noul\",\"noul\":1.0}}", "q", &p, err, sizeof(err)));
    CHECK(p == 1.0);
    CHECK(ds4_d1_response_noul("{\"answers\":{\"q\":0.5}}", "q", &p, err, sizeof(err)));
    CHECK(p == 0.5);
    CHECK(!ds4_d1_response_noul("{\"answers\":{\"q\":{\"noul\":1.5}}}", "q", &p, err, sizeof(err)));
    CHECK(!ds4_d1_response_noul("{\"answers\":{\"qq\":{\"noul\":0.1}}}", "q", &p, err, sizeof(err)));
    CHECK(!ds4_d1_response_noul("{\"error\":{\"message\":\"bad\"}}", "q", &p, err, sizeof(err)));
    CHECK(!ds4_d1_response_noul("not json", "q", &p, err, sizeof(err)));
}

static void test_http(void) {
    fake_server s;
    double p = -1;
    char err[256] = {0};
    const char *body = "{\"answers\":{\"q\":{\"type\":\"noul\",\"noul\":0.875}}}";
    char reply[512];
    snprintf(reply, sizeof(reply),
             "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %zu\r\n\r\n%s",
             strlen(body), body);
    CHECK(ask_tcp(reply, 2000, &p, err, sizeof(err), &s));
    CHECK(p == 0.875);
    CHECK(strstr(s.request, "POST /v1/systemone HTTP/1.1\r\n") == s.request);
    CHECK(strstr(s.request, "\"state\":\"cwd: /tmp\\n\\\"quoted\\\" \\\\ state\"") != NULL);
    CHECK(strstr(s.request, "\"q\":{\"type\":\"noul\",\"instructions\":\"Is it risky?\"}") != NULL);

    CHECK(ask_tcp("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
                  "10\r\n{\"q\":{\"noul\":0.1\r\n4\r\n25}}\r\n0\r\n\r\n",
                  2000, &p, err, sizeof(err), &s));
    CHECK(p == 0.125);

    CHECK(!ask_tcp("HTTP/1.1 500 Internal Server Error\r\nContent-Length: 5\r\n\r\noops!",
                   2000, &p, err, sizeof(err), &s));
    CHECK(strstr(err, "HTTP 500") != NULL);

    CHECK(!ask_tcp(NULL, 100, &p, err, sizeof(err), &s));
    CHECK(strstr(err, "timed out") != NULL);
}

static void test_unix_and_refused(void) {
    char path[] = "/tmp/ds4-d1-test-XXXXXX";
    int tmp = mkstemp(path);
    close(tmp);
    unlink(path);
    fake_server s = {0};
    s.listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un a = {0};
    a.sun_family = AF_UNIX;
    snprintf(a.sun_path, sizeof(a.sun_path), "%s", path);
    CHECK(bind(s.listen_fd, (struct sockaddr *)&a, sizeof(a)) == 0);
    CHECK(listen(s.listen_fd, 1) == 0);
    s.reply = "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n{\"q\":{\"noul\":0}}";
    pthread_t th;
    pthread_create(&th, NULL, fake_server_main, &s);
    char spec[64], err[160];
    snprintf(spec, sizeof(spec), "unix:%s", path);
    ds4_d1_endpoint ep;
    double p = -1;
    CHECK(ds4_d1_parse_endpoint(spec, &ep, err, sizeof(err)));
    CHECK(ds4_d1_noul(&ep, "s", "i", 2000, &p, err, sizeof(err)));
    CHECK(p == 0.0);
    pthread_join(th, NULL);
    close(s.listen_fd);
    unlink(path);

    /* Nothing listens on the removed socket: fail fast with an error. */
    CHECK(!ds4_d1_noul(&ep, "s", "i", 2000, &p, err, sizeof(err)));
    CHECK(strstr(err, "connect") != NULL);
}

int main(void) {
    signal(SIGPIPE, SIG_IGN);
    test_endpoints();
    test_response_layouts();
    test_http();
    test_unix_and_refused();
    if (failures) {
        fprintf(stderr, "test_d1_client: %d failure(s)\n", failures);
        return 1;
    }
    printf("test_d1_client: ok\n");
    return 0;
}

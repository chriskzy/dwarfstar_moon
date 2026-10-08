#ifndef DS4_D1_H
#define DS4_D1_H

#include <stdbool.h>
#include <stddef.h>

/* Client for a local decision-model server speaking POST /v1/systemone, such
 * as `llama-server -hf LiquidAI/d1-3B-GGUF:Q8_0`.  The agent asks it typed
 * questions about its own state (never the KV cache or transcript) and gets
 * calibrated probabilities back from one forward pass. */

typedef struct {
    char host[256];          /* TCP host, empty when unix_path is used */
    int port;
    char unix_path[104];     /* sockaddr_un.sun_path limit on macOS */
} ds4_d1_endpoint;

/* Accepts PORT, HOST:PORT, unix:/path or an absolute socket path. */
bool ds4_d1_parse_endpoint(const char *spec, ds4_d1_endpoint *ep,
                           char *err, size_t err_len);

/* POST the request body to /v1/systemone and return the response body of a
 * 200 reply (caller frees), or NULL with err set. */
char *ds4_d1_systemone(const ds4_d1_endpoint *ep, const char *request_json,
                       int timeout_ms, char *err, size_t err_len);

/* Extract the probability of a noul answer named `question`, accepting both
 * {"answers":{q:{...}}} and {q:{...}} response layouts. */
bool ds4_d1_response_noul(const char *json, const char *question, double *p,
                          char *err, size_t err_len);

/* Ask one yes/no question about a plain-text state. */
bool ds4_d1_noul(const ds4_d1_endpoint *ep, const char *state,
                 const char *instructions, int timeout_ms, double *p,
                 char *err, size_t err_len);

#endif

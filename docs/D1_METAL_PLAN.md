# Plan: a Mac-only Metal engine for Liquid d1-3B, built on the ds4 skeleton

Status: proposal, nothing implemented yet. Upstream ds4 snapshot used for this
plan: `fc80bd6`.

## 1. Recommendation in one paragraph

Build a small, separate engine (`d1/` in this fork, its own Makefile and
binaries) that copies ds4's *engineering shape* and a good part of its *Metal
code*: mmap'd GGUF, runtime Metal library, the dense quantized GEMMs, norms,
flash attention, and the image decoder. Write new code only where d1 really
differs: the bidirectional LFM2 encoder, the centered short conv, the decision
head and the `/v1/systemone` readout. Run it as a second process next to ds4.
Connect ds4-agent to it over a local socket. Do the agent integration **first,
against stock `llama-server`**, which already serves d1, so the integration does
not wait on the engine. After the first Metal baseline, decide go/no-go: keep
the custom engine only if it is clearly faster than llama.cpp on the same GGUF,
with no change in decisions.

## 2. What the research changed in the original background

The background was mostly right about the process layout. Four points need
correcting before any code is written:

1. **The readout is a trained head, not LM-head logits.** llama.cpp runs LFM2
   decision models through a separate graph (`graph_decision` in
   `src/models/lfm2.cpp`):
   - The first `n_layer - n_layer_decision` layers form a **non-causal
     encoder**. Attention is bidirectional. The short conv is a **centered**
     3-tap filter (t-1, t, t+1), and taps outside the sequence read zero. It is
     not the causal conv used by the chat LFM2.
   - The encoder mask lets text attend to media, but media never attends to
     text.
   - The last `n_layer_decision` layers form a **decision head**. Each head
     layer uses LayerNorm with bias, a fused QKV with bias, no RoPE, and a ReLU
     FFN with bias and no gate. Head attention only links tokens of the same
     sequence and the same kind (text or media).
   - A learned `type_embd` row (choice, score or noul) is added to the head
     input, and the head runs once per type.
   - A small classifier (`cls_norm` → `cls` + bias → GELU(erf) → `cls_out` +
     bias) gives **one scalar per type per output row**.
   - The 65,536 × 2,048 output projection is never used.

   Consequence: there is no KV cache, no sampler, and no causal prefix to
   reuse. Every text token depends on every other text token.
2. **Several ds4 kernels are reusable.** The background said otherwise. The
   MLA, MoE, compressor, HC and indexer kernels are not reusable. But the
   largest single cost in d1, the dense GEMM, is already in
   `metal/dense.metal`:
   - `kernel_mul_mm_{q8_0,q4_K,q4_0,f16}_f32` (simdgroup MMA, M1–M4).
   - `*_nax_direct_rhs*` Metal 4 TensorOps variants for M5. ds4 enables these
     automatically when `supportsFamily:MTLGPUFamilyMetal4`
     (`ds4_metal.m:2888`).

   Norms, SwiGLU, embedding gather, softmax and flash attention are generic too
   (§5).
3. **The instance lock would block d1 from running next to ds4.**
   `ds4_acquire_instance_lock()` (`ds4.c:60877`) takes `flock` on
   `/tmp/ds4.lock`. Code copied from ds4 must use its own lock file. Otherwise
   d1 refuses to start while ds4 is running.
4. **Prefix caching is not where the speed comes from.** llama.cpp only
   supports a shared parallel prefix for causal models, and d1 is bidirectional.
   Most of the speed has to come from GEMM efficiency and from skipping work
   nobody reads (§4). The exception is images (§4, item 6).

Items marked **(P0)** below are unknown until Phase 0 reads the GGUF and
llama.cpp's `server_decision_context`.

## 3. Workload and cost model

The expected shapes come from LFM2-2.6B, d1's text backbone. Confirm them from
the d1 GGUF in Phase 0 **(P0)**:

| Item | Expected value |
| --- | --- |
| Layers | 30 (22 short-conv, 8 GQA; first attention layer at index 2) |
| Hidden size | 2048 |
| FFN width | 10752 (SwiGLU) |
| Attention heads | 32 Q / 8 KV heads, head_dim 64, QK-RMSNorm, RoPE |
| Conv projections | `in_proj` 2048→6144 (B, C, x); `out_proj` 2048→2048; conv weight [3, 2048] |
| Vocabulary | 65,536 (embedding gather only; no output GEMM) |
| Vision | SigLIP2 NaFlex (~400M), pixel-unshuffle, then a 2-layer MLP projector (separate mmproj GGUF) |
| Decision head | `n_layer_decision` layers, `type_embd` [2048, 3], `cls`, `cls_out` **(P0: count and FFN width)** |

Parameter split, excluding the embedding (≈2.43B total):

| Part | Parameters | Share of matmul FLOPs |
| --- | --- | --- |
| FFN | ≈1.98B | ≈81% |
| Short-conv projections | ≈0.37B | |
| Attention projections | ≈0.08B | |

From those numbers:

- **FLOPs per token ≈ 4.9 GFLOP.** One request is a single forward pass over N
  tokens, so the cost is ≈ 4.9·N GFLOP.
- **Q8_0 weights ≈ 2.6 GB.** Each weight tile is reused across all N tokens, so
  arithmetic intensity is ≈ 1.9·N FLOP/byte. Apple GPUs balance at roughly
  20–40 FLOP/byte. **Any request longer than about 20 tokens is compute-bound.**
  The question text alone is longer than that.
- **Latency rule of thumb: ≈ 4.9·N / TFLOPS_eff milliseconds.** That is ~49 ms
  per 100 tokens at 10 effective TFLOPS. Measure TFLOPS_eff per Mac in Phase 2;
  don't trust guesses. "Tens of milliseconds" holds for short requests.
  Agent-sized payloads of 300–800 tokens cost proportionally more.

Implications for the design:

- Memory bandwidth, KV layout and quant size barely affect speed. Matmul
  efficiency on prefill shapes, and computing nothing that isn't read, decide
  it.
- The quant format matters for accuracy, memory footprint, and dequant ALU in
  the GEMM inner loop. It does not matter for bandwidth.
- Attention is about 1% of FLOPs at N = 1K (≈ 4·N²·d per layer × 8 layers).
  Use a correct, simple kernel and leave attention tuning until N > ~8K.
- The agent integration should keep states short. Each token of payload costs
  latency.

## 4. Efficiency levers, ranked by expected payoff

Keep each lever only if fixtures show no drift (§6) and the benchmark shows a
gain.

1. **GEMM throughput (≈95% of time).**
   - Reuse ds4's `kernel_mul_mm_*` (M1–M4) and the TensorOps `nax_direct_rhs`
     path (M5).
   - Pick a kernel per (shape, N bucket, chip) from a microbenchmark table
     built in Phase 3.
   - **Load-time F16 repack.** ds4's Q8_0/Q4_K `kernel_mul_mm` dequantizes into
     `simdgroup_half8x8` before the MMA (`metal/dense.metal:2931-2933`). If the
     repack rounds with the same dequant function, it feeds identical half
     values to the same MMA. Expected result: bit-identical outputs, with no
     dequant ALU or scale loads in the inner loop. That inner loop is the
     bottleneck in this compute-bound regime. Cost: ≈4.9 GB instead of 2.6 GB.
     Make it a load option and pick the default from measurements. Check bit
     identity for each kernel path; the M5 path stages differently.
   - **Concatenated weights.** Repack gate|up into one GEMM and Q|K|V into
     another at load. That means fewer dispatches and larger N-tiles.
2. **Skip work nobody reads.**
   - Never touch `output.weight` or the tied LM head.
   - Run the decision head only for the question types present in the request.
     llama.cpp runs all three. When more than one type is needed, stack them as
     `[types × tokens]` rows in one GEMM rather than separate passes.
   - Head attention is same-kind only, and readouts are text positions
     **(P0)**. Media rows therefore cannot affect any output in the head, so
     run the head over text rows only.
   - Prune the last head layer (attention Q, FFN, classifier) to the output
     rows, as llama.cpp does with `inp_out_ids`.
3. **Fusion for the elementwise work and dispatch count.**
   - **Conv block:** the `in_proj` epilogue writes `B∘x` and `C`. One kernel
     then does the centered conv3 with the C-gate, with a ±1-token halo per
     tile. The pattern to adapt is `kernel_qwen4_conv_halo` /
     `kernel_qwen4_conv_blocked`, which are causal with a one-sided halo.
   - **FFN:** SwiGLU in the gate|up epilogue halves the intermediate writes.
   - **Attention:** QK-RMSNorm + RoPE fused after the QKV GEMM. ds4 does the
     same in `kernel_dsv4_qkv_rms_norm_*`.
   - **Residual:** add + RMSNorm fused (`kernel_add_rms_norm_mul_f32_4`).
4. **Encoding and launch overhead.** About 250 dispatches per pass.
   - One command buffer per request, using ds4's `begin_commands` /
     `flush_commands`.
   - A persistent scratch arena sized for the maximum N.
   - N bucketed to multiples of 64.
   - Precompile a `.metallib` at build time instead of ds4's runtime
     `newLibraryWithSource`, so startup is fast.
   - Consider indirect command buffers only if measured CPU encode time exceeds
     about 1 ms.
5. **Request batching in the server.** Pack concurrent requests into one pass
   (variable length; mask by sequence; conv taps zeroed at sequence edges) to
   raise GEMM utilization for short requests. Use a small batching window of
   ≤2 ms. This helps server workloads more than the single-user agent.
6. **Image caching (conditional).**
   - Cache projector outputs keyed by `ds4_image.fingerprint`.
   - **Possible further win:** media never attends to text in the encoder, and
     RoPE dot products depend only on relative position. Media hidden states
     may then be independent of the surrounding text. Two checks decide it:
     the conv masks must stop at media/text boundaries, and media positions
     must be 1-D **(P0)**. If both hold, cache each image's per-layer K/V and
     skip media rows in every encoder GEMM on a cache hit. Repeated image
     triage then costs about the same as text alone.

## 5. Engine layout and what to take from ds4

The rules carry over from `AGENT.md`:

- C99 core, with Objective-C only at the Metal boundary. No C++.
- mmap-backed weights, never a full copy.
- A CPU path that exists only as the reference.
- One release path, with diagnostic switches only.

The build is Darwin-only and stops with an error on any other OS.

Copy the files ds4 needs, don't share them by path, so later upstream ds4 changes cannot break
d1. Record the source commit in `d1/VENDOR.md`, the same way `cuda/mmq/VENDOR.md`
does it.

```
d1/
  Makefile           Darwin-only; -framework Metal -framework Foundation
  d1.h               narrow public API: load, run(request) -> answers
  d1.c               GGUF/mmap, tokenizer, graph scheduling, CPU reference
  d1_metal.m         Metal runtime + kernel wrappers (subset of ds4_metal.m)
  d1_image.c         vendored ds4_image.c + third_party/iris (PNG/JPEG, EXIF, fingerprint)
  d1_systemone.c     request rendering, readout, calibration (mirror of llama.cpp)
  d1_server.c        POST /v1/systemone, GET /health; unix socket + 127.0.0.1
  d1_cli.c           one-shot JSON in/out, --bench, --dump-layer
  metal/             vendored dense/norm/glu/get_rows/unary/softmax/flash_attn + new d1_*.metal
  tests/             op exactness tests, fixtures, golden runner
```

| ds4 source | Use in d1 |
| --- | --- |
| `ds4.c` `gguf_types` table and GGUF parser, mmap | Keep |
| `ds4.c` BPE core (`bpe_rank`, `bpe_emit_piece`, `vocab_load`) | Keep, and add an LFM2 pre-tokenizer next to the GLM and Qwen variants |
| `ds4.c` CPU quant dot / dequant | Keep for the reference path |
| `ds4.c` instance lock | Keep, with its own path (`/tmp/d1.lock`) |
| `ds4_metal.m` device init, Metal4 detection, library build with macros | Keep |
| `ds4_metal.m` model-range wrapping (`newBufferWithBytesNoCopy`, page-aligned views, `maxBufferLength` split), tensor alloc/view, command batching, residency sets | Keep |
| `ds4_gpu.h` | Keep only the subset d1 calls |
| `ds4_server.c` socket loop, HTTP parsing, `json_*` helpers | Extract into `d1_server.c` |
| `ds4_image.c` + `third_party/iris` | Keep as is |
| `tests/test_metal_dense_mpp.c` exactness style; `tests/test-vectors` fixture style; `speed-bench` CSV and plots | Copy the patterns |
| CUDA, ROCm, TP, distributed, SSD streaming, Engram, layer pack, kvstore/rax | Drop |
| MoE, HC, indexer, MTP/DSpark, sampling, chat templates, tool parsing | Drop |
| `ds4_agent.c`, `ds4_eval*`, dir-steering | Drop from the engine (the agent stays in ds4) |

Kernel map:

| d1 op | ds4 starting point | Work needed |
| --- | --- | --- |
| Token embedding | `get_rows.metal` `kernel_get_rows_{q8_0,q4_K}_f32` | None |
| RMSNorm (+ residual add) | `norm.metal` `kernel_rms_norm_mul_f32_4`, `kernel_add_rms_norm_mul_f32_4` | None |
| Dense GEMM, all projections | `dense.metal` `kernel_mul_mm_*`, `*_nax_direct_rhs*` | Shape tables; F16 repack option |
| Centered short conv + B/C gating | `qwen4.metal` conv halo/blocked (causal) | New kernel, about 40 lines: two-sided halo, gating fused |
| QK-RMSNorm + RoPE | `norm.metal` `kernel_dsv4_qkv_rms_norm_*`, `dsv4_rope.metal` | Adapt to head_dim 64 and full-dim RoPE **(P0: theta, scaling)** |
| Bidirectional GQA attention with kind mask | `flash_attn.metal` (ggml-derived, mask plus block-skip pre-pass) | Add a dk64/dv64 instantiation; build the mask once per request |
| SwiGLU | `glu.metal` `kernel_swiglu_f32` | Later fused into the GEMM epilogue |
| Head LayerNorm with bias | `qwen4_vision.metal` `kernel_qwen4_vis_layernorm` | Generalize |
| Head bias + ReLU, `cls` bias + GELU(erf) | `qwen4_vision.metal` `kernel_qwen4_vis_bias_act`, `unary.metal` | Make sure GELU is the erf form, not tanh |
| Type-embedding add, classifier dot | New | Trivial; run in FP32 |
| SigLIP2 ViT, pixel-unshuffle, projector | `qwen4_vision.metal`, `glm53_vision.metal` (`downsample_reorder`) | New d1 kernels from these patterns **(P0: NaFlex positions, tiling, thumbnail)** |

## 6. Correctness and calibration

d1's value is calibrated probabilities. The bar is therefore "same
probabilities", not "same argmax".

- **Reference.** Use `llama-server /v1/systemone` from a **pinned** llama.cpp
  commit on the same GGUF. Fixtures come from it. Use the Metal backend as the
  primary reference: it also stages GEMM inputs in half precision, as ds4's
  kernels do. Keep the CPU backend as a second opinion.
- **Fixtures.** About 200 requests covering noul, choice and score, 2–255
  options, short and long states, JSON states, and images. Store for each:
  - the request and full response (`probabilities`, `score`, `noul`,
    `confidence`);
  - the raw head scalars, from a small llama.cpp dump patch.
- **Metrics:**
  - max |Δ raw head output| per type;
  - max |Δp| and KL per question;
  - **zero decision flips**;
  - ECE unchanged against the reference on a labeled subset. The AtomicChat
    d1-3B-GGUF-metrics dataset shows a ready-made fidelity/KL method.
- **Per-layer diff harness.** `--dump-layer N` writes hidden states from both
  the CPU reference and Metal. This is how custom kernels stay honest. It is
  also how ds4 hunts drift (see the `DS4_METAL_*` drift switches in
  `ds4_metal.m`).
- **Precision policy:**
  - FP32 residual stream, norms, head and classifier.
  - GEMMs: half inputs with FP32 accumulation, as in ds4 and llama.cpp.
  - Head kernels compiled without fast-math. ds4 documents that fast-math
    differs between the M5 TensorOps and simdgroup paths.
- **Calibration temperatures.** Read them from GGUF metadata
  (`{arch}.decision.temperature.<type>[.<bucket>]`) and apply them exactly as
  `server_decision_context` does **(P0)**.
- **Default quant: Q8_0.** Use Q4_K_M only if it passes the same flip and KL
  gates.

## 7. ds4-agent integration

d1 runs as a **separate process**. It must not share ds4's process, lock or
memory guard. Its footprint is small: about 2.9 GB mapped for Q8_0, plus the
mmproj, plus less than 0.5 GB of activations at N = 2K. Add about 5 GB if the
F16 repack is enabled. This fits next to Flash Q2 on a 128 GB Mac, but count it
in the reserve you leave when sizing ds4's context.

- **Transport.** The `/v1/systemone` body is byte-compatible with
  `llama-server`, so the agent cannot tell the backends apart.
  - Listen on a unix socket (`~/.ds4/d1.sock`) and on 127.0.0.1.
  - Client: a small POST client in ds4 (`ds4_d1_client.c`). `ds4_web.c`'s
    `web_http_request` is a loopback client but only does GETs without a body;
    copy its pattern.
  - Timeout of about 2 s. A failure becomes `Tool error: …`, as other tools
    report.
- **Payload.** The last user message plus a short structured state: cwd, last
  tool, exit status, and the pending command or patch summary. Never the KV or
  the transcript. Every token costs latency (§3).
- **Two ways to use it. Ship the first one first:**
  1. **Harness hooks, which never enter the LLM's context.** At fixed points in
     `ds4_agent.c`, the agent asks d1 typed questions. Candidates, to be
     validated against real sessions:
     - **Before tool execution** (`agent_execute_tool_call`): a noul question,
       "is this command destructive or outside the workspace?", gating
       auto-run versus asking the user.
     - **At turn start:** a score question that maps to `--think-level` for
       V4.1, or a choice question that routes the turn (answer directly / use
       tools / read files first).
     - **After a tool result:** a noul question, "did this fail in a way the
       model should retry?"
     - **Images:** triage before running the large vision encoder.
  2. **A model-callable `decide` tool.** Add its schema conditionally, only
     when `--d1 ADDR` is given. That is the same pattern as `view_image`, which
     the DSML, GLM and Qwen tool-prompt builders add only with `vision`
     (`ds4_agent.c:1291`, `1358`, `1411`). The tool body is the HTTP call.
- **GPU sharing.** Hooks run while ds4 is idle, waiting on a tool or on the
  user, so they do not contend with it. In `ds4-server` with live sessions, a
  d1 pass delays decode by about its own duration. Measure this in Phase 6.
- **Tests.** Extend `ds4_agent_test` with a fake d1 server: schema present only
  with `--d1`; timeout and error paths; a hook's decision changing the
  approval path.

## 8. Phases and exit criteria

There are two tracks. Track A does not depend on Track B.

### Track A: integration against llama-server (start immediately)

**Status: A1 implemented.** `ds4_d1.{c,h}` is the `/v1/systemone` client
(TCP or unix socket, bounded timeout, chunked replies). `ds4-agent --d1 ADDR`
asks d1 one noul question before each `bash` command, with only the working
directory and the command as state. At or above `--d1-threshold` (default 0.5)
the user must approve the command; non-interactive runs refuse it. If the
server is unreachable the gate fails open and says so once. Tests:
`tests/test_d1_client` and `test_d1_bash_gate` in `ds4_agent_test`.

```sh
llama-server -hf LiquidAI/d1-3B-GGUF:Q8_0 --port 8091
./ds4-agent --d1 8091
```

- **A1.** Add `ds4_d1_client.c`, `--d1 ADDR`, one harness hook (the pre-tool
  safety gate), and the fake-server tests.
  - **Exit:** `make` and `make test` (agent and server unit tests) pass, and
    ds4 paths without `--d1` are unchanged.
- **A2.** Measure end-to-end hook latency and usefulness on recorded agent
  sessions.
  - **Exit:** a budget (for example, p95 under 100 ms at the chosen payload
    size) and a decision on which further hooks to ship.

**A2 status: tooling ready; measurements still to run on a Mac with d1 loaded.**
`speed-bench/d1_gate_eval` replays commands through the same
`ds4_d1_bash_risk()` call the agent makes. It reads a labeled corpus
(`speed-bench/d1-gate-corpus.tsv`) and/or `ds4-agent --trace` files: the agent
now traces every bash command (`bash-command="..."`, with or without `--d1`)
and every gate verdict with its latency. Procedure and the numbers to record
are in `speed-bench/README.md` ("d1 bash gate").

Decision criteria for A2, to be confirmed against the first measurements:

| Question | Pass |
| --- | --- |
| Latency | p95 ≤ 100 ms and p99 ≤ 250 ms per gate call, with ds4 idle |
| Separation | AUC ≥ 0.95 on the labeled corpus |
| Safety | a threshold that catches every risky corpus command |
| Cost | at that threshold, ≤ 5 approval prompts per 100 safe commands, and ≤ 5 per 100 commands on real traces |

**First run (2026-10-09): M4 Max, d1-3B Q8_0, llama.cpp master with PR
#30110, variant 1 question, 149 commands × 3:**

| Question | Result | Pass? |
| --- | --- | --- |
| Latency | p50 48 ms, p95 62 ms, p99 76 ms, max 171 ms (≈28 input tokens per call for a short command) | yes |
| Separation | AUC 0.946 | just under |
| Safety + cost | catching all 55 risky costs 85 prompts per 100 safe; at ≤5 per 100 it catches 41/55 | no |

The misses were mostly false alarms on ordinary work: `git push`, `git pull`,
`sed -i`, `git commit`, `cmake` scored 0.63–0.92. The broad question's
"network" and "overwriting" clauses were read literally. The lowest-scored
risky commands were `> ~/.bash_history`, a keychain read piped to the
clipboard, and `curl … | sh`. Variant 2 (two narrower questions, each saying
what does not count) is the next measurement:
`d1_gate_eval --variant 2 ...`.

If latency fails, Track B becomes the priority. If separation or cost fails,
try rewording `ds4_d1_bash_question` or splitting it into two questions
("destroys data?", "reaches outside the project?") before adding more hooks.
Only once the gate passes, add the next hook (turn routing or think level).

### Track B: the engine

- **P0. Ground truth, without Metal.**
  - Download the Q8_0 GGUF and the mmproj, and dump the metadata and tensor
    table: layer types, `n_layer_decision`, head shapes and FFN widths, eps
    values, RoPE parameters, `decision.type`, temperature keys, tokenizer
    pre-type, and special tokens.
  - Read llama.cpp's `server_decision_context` for d1: prompt rendering, which
    rows are outputs, the scalar→probability mapping for each type, the
    confidence formula, and how several questions and images are laid out.
  - Pin the llama.cpp commit and capture the fixtures.
  - **Exit:** the spec is written at the top of `d1.c` and
    `d1_systemone.c`, the fixtures are committed, and every (P0) item in this
    document is resolved.
- **P1. Skeleton and CPU reference.**
  - Vendor the files and build the Makefile.
  - Tokenizer exact against `llama-tokenize` on 1,000 or more strings.
  - FP32 CPU forward for short prompts.
  - **Exit:** the CPU reference reproduces the fixture probabilities within
    1e-3 with zero flips.
- **P2. Metal baseline, unfused, reusing ds4 kernels.**
  - Per-op exactness tests against the CPU reference.
  - The per-layer diff harness.
  - **Exit:** all fixtures pass. A latency table at N ∈ {64, 256, 1024, 4096}
    against `llama-server` Metal on the same Mac and GGUF. Measured
    TFLOPS_eff.
  - **Go/no-go:** continue only if the baseline is already competitive and
    Phase 3 can plausibly give ≥1.3× over llama.cpp. Otherwise stop. Track A
    keeps working against llama-server, so nothing is lost.
- **P3. Efficiency.** Apply the levers of §4 in order: 2, then 1, then 3, then
  4, then 5. Each lever is its own commit with before/after numbers.
  - **Exit:** the target speed-up holds at every N bucket, there is no fixture
    drift, and a speed-bench CSV is recorded for each chip available (M1–M5).
- **P4. Vision.**
  - SigLIP2 NaFlex preprocessing and ViT, pixel-unshuffle, and the projector.
  - The image cache, plus the conditional per-image encoder cache from §4.6.
  - **Exit:** image fixtures pass against llama.cpp mtmd.
- **P5. Server.**
  - `d1-server` with a single GPU worker, a batching queue and a health
    check.
  - Swap the Track A backend from llama-server to d1-server.
  - **Exit:** the agent hooks behave identically, with lower latency.
- **P6. QA and packaging.**
  - A QA checklist in the style of `QA_BEFORE_RELEASES.md`: fixtures, speed
    for each chip, coexistence with a loaded ds4 model, memory, and long
    inputs up to 32K.
  - A download script for the weights.

## 9. Risks and open questions

- **Readout semantics (P0).** Output positions, multi-question layout and the
  confidence formula all live in llama.cpp server code. Mirror them exactly,
  and pin the commit; d1 support there is weeks old.
- **Multi-question cost (P0).** All questions in one sequence means one pass.
  One sequence per question means one bidirectional pass each, which
  multiplies cost and makes request batching (§4.5) much more valuable.
- **Calibration drift** from kernel precision or fast-math. Mitigations: an
  FP32 head, no fast-math in the head, and the zero-flip gate.
- **Is a custom engine worth it?** The baseline to beat is llama.cpp, which
  already runs d1 on Metal. MLX has LFM2 support for the causal LM, but I
  could not confirm a port of d1's decision graph. Phase 2's go/no-go gate
  exists for this reason.
- **License.** The weights are under the LFM Open License v1.0, which limits
  commercial use above a revenue threshold. Do not bundle the weights; download
  them at install time. Engine code copied from ds4 must keep ds4's MIT notices,
  including the GGML authors.

## 10. Sources

- d1-3B GGUF model card: https://huggingface.co/LiquidAI/d1-3B-GGUF
- Liquid open d1 announcement: https://www.liquid.ai/blog/d1-open
- llama.cpp decision models, PR #29818: https://github.com/ggml-org/llama.cpp/pull/29818
- llama.cpp LFM2 graph, including `graph_decision`: https://github.com/ggml-org/llama.cpp/blob/master/src/models/lfm2.cpp
- Decision-model readout as implemented in jev-rs: https://github.com/yijunyu/jev-rs
- LFM2-VL-3B (backbone and vision tower): https://huggingface.co/LiquidAI/LFM2-VL-3B
- Quant fidelity metrics: https://huggingface.co/datasets/AtomicChat/d1-3B-GGUF-metrics

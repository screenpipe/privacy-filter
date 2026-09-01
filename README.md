# privacy-filter

GPU-backed HTTP wrapper around two PII models plus co-hosted Gemma and GLM chat models in one Tinfoil enclave:

1. [`screenpipe/pii-redactor`](https://huggingface.co/screenpipe/pii-redactor) (`v50_distilled6l`) — six-layer, vocab-pruned XLM-R token classifier for **text PII**. Endpoint `POST /filter`.
2. [`screenpipe/pii-image-redactor`](https://huggingface.co/screenpipe/pii-image-redactor) (`rfdetr_v19`) — RF-DETR-Nano detector for **image PII** in screenshots, trained on real-app and synthetic screens. Endpoint `POST /image/detect`.
3. [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it) — chat + vision + **audio** via vLLM. Endpoint `POST /v1/chat/completions` with the backwards-compatible API model id `"gemma4-e4b"`. Weights are baked into the image (~10 GB BF16).
4. [`patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF) — 50%-expert-pruned GLM-5.3-Flash, served from the 67.2 GiB `IQ3_M` quant by a pinned CUDA `llama-server`. Endpoint `POST /glm/v1/chat/completions`.

> **Note:** the 31B model (`gemma4-31b`) is **not** in this container. It stays on Tinfoil's hosted `inference.tinfoil.sh` endpoint, reached separately by `packages/ai-gateway` in the Cloudflare worker. The co-hosted E2B is a smaller multimodal-audio model, not a drop-in replacement.

All four workloads deploy inside the same [Tinfoil](https://tinfoil.sh) confidential-compute enclave on one H200 (~141 GB VRAM) so neither pixels, text, nor chat prompts leave an attested runtime. The existing `privacy-filter` container remains the only public router: `/v1/*` goes to Gemma over loopback and `/glm/v1/*` goes to the isolated GLM sibling over an enclave-only bridge. One TLS-attested URL, one auth token, four workloads sharing one allocation.

## API

```
GET  /health         → {"status": "ok", "model_ready": true, "image_model_ready": true, ...}

POST /filter         → {"text": "My email is alice@foo.com"}
                    ←  {"redacted": "My email is [EMAIL]",
                        "spans": [{"label": "private_email", "start": 12, "end": 25,
                                   "text": "alice@foo.com", "score": 0.99}],
                        "latency_ms": 180,
                        "model": "screenpipe/pii-redactor:v50_distilled6l (mixed-int4-int8-onnx)"}

POST /image/detect   → {"image_b64": "<b64-jpg-or-png>", "threshold": 0.30}
                    ←  {"detections": [{"bbox": [x, y, w, h], "label": "private_person", "score": 0.95},
                                       {"bbox": [x, y, w, h], "label": "secret",        "score": 0.91}],
                        "latency_ms": 32, "model": "rfdetr_v19",
                        "width": 2880, "height": 1800}

POST /glm/v1/chat/completions
                    → {"model": "glm-5.3-flash-reap50-iq3m",
                       "messages": [{"role": "user", "content": "Hello"}]}
                    ← OpenAI-compatible chat-completion response
```

Bbox is `[x, y, w, h]` in original-image pixel space (the server un-resizes from the model's 512×512 input). Labels are the canonical 12-class screenpipe PII taxonomy.

## Local development

```bash
# build
docker build -t privacy-filter:dev .

# run
docker run --rm -p 8080:8080 privacy-filter:dev

# smoke test
curl -s http://localhost:8080/health
curl -s -X POST http://localhost:8080/filter \
     -H 'Content-Type: application/json' \
     -d '{"text":"Call Alice at +1 415 555 0100 about alice@example.com"}' | jq

# Image: send a JPG/PNG as base64.
B64=$(base64 -i some_screenshot.png)
curl -s -X POST http://localhost:8080/image/detect \
     -H 'Content-Type: application/json' \
     -d "$(jq -nc --arg img "$B64" '{image_b64: $img, threshold: 0.30}')" | jq
```

The first build downloads the v50 text contract (~133 MB across model, tokenizer, config, and vocabulary remap), the 60 MB rfdetr_v19 ONNX, and the baked Gemma weights. Subsequent builds use the build cache. Both HuggingFace revisions and every model artifact checksum are pinned, so a rebuild cannot silently drift to different weights.

## Deploy to Tinfoil

1. **Build and push the image to GitHub Container Registry** using the
   `privacy-filter release` workflow. Supply the semantic version without
   the `v` prefix (for example `0.9.0`). The workflow prints the immutable
   router image digest. Build the GLM runtime with its separate workflow; it
   publishes under the existing public `privacy-filter` package.

   ```bash
   gh workflow run release.yml --ref <branch> -f version=0.9.0
   gh workflow run glm-runtime.yml --ref <branch> -f version=0.9.0
   ```

2. **Pin the digest** in `tinfoil-config.yml`, commit it, then create the
   matching `v*` tag. The tag runs the Tinfoil measurement/attestation job.

   ```bash
   git add tinfoil-config.yml
   git commit -m "release: privacy-filter v0.9.0"
   git tag v0.9.0 && git push origin <branch> v0.9.0
   ```

3. **Update the existing deployment** through Tinfoil's blue-green update
   flow. Stage the candidate when you want to exercise it before promotion;
   the current instance stays live unless the candidate becomes healthy and
   is accepted.

4. **Verify** — the service is now reachable at
   `https://privacy-filter.<org>.containers.tinfoil.dev/health`.

## Resource sizing (GPU)

**Text model (`v50_distilled6l`) — mixed int4/int8 ONNX:**

| Metric | Value |
|---|---|
| Model artifact | 114 MB |
| Model shape | six-layer XLM-R student |
| Runtime window | 256 tokens with 64-token overlap |
| Local CPU reference | 7–9 ms for short captured strings |

**Image model (`rfdetr_v19`) — TensorRT/CUDA EP:**

| Metric | Value |
|---|---|
| Weights (FP16, FP32 I/O) | 60 MB |
| Params | ~25 M |
| Input resolution | 512×512 |
| Local CPU reference | ~118 ms/frame |
| Real-app planted-secret gate | 19/19 detected at the production 0.50 floor |

The deployed CVM also carries Gemma E2B and its vLLM working set, so
`tinfoil-config.yml` remains the source of truth for total CPU, memory, and GPU sizing.

**GLM-5.3-Flash REAP50 (`IQ3_M`) — llama.cpp/CUDA:**

| Metric | Value |
|---|---|
| Quantized model | 67.2 GiB |
| Parameters after expert pruning | about 165B (from 321B) |
| Runtime context | 32,768 tokens, one parallel slot |
| GPU offload | all layers on the H200 |

The model and runtime are pinned by exact commits. Tinfoil mounts the measured model package read-only; the model is not downloaded at runtime.

## Security properties

- **Tinfoil remote attestation** covers the exact image digest, so clients can verify the specific model bits + server code that handled their request.
- **Only the paths listed in `tinfoil-config.yml` are exposed** — Tinfoil's shim allowlist blocks every other URL at the enclave boundary.
- **Model weights are immutable.** The text, image, and Gemma artifacts are baked into the router image. GLM is a measured Tinfoil model package mounted read-only. No workload downloads weights at runtime.
- **The exact container image is measured and attested.** The process currently runs as root because the co-hosted vLLM needs access to NVIDIA device nodes; the Tinfoil shim and CVM remain the external security boundary.

## Limitations

- **Text:** English-primary. Multilingual coverage per upstream model card varies.
- **Text:** long inputs are split into overlapping 256-token windows; request size is capped separately by `MAX_INPUT_CHARS`.
- **Image:** v19 is validated on released eval suites and planted real-app secrets, but unusual layouts, scripts, handwriting, occlusion, and adversarial inputs still need their own evaluation.
- **Image:** payload cap `MAX_IMAGE_BYTES=20 MB` (override via env). Decoded RGB working buffer is ~3× the payload.
- **GLM:** the REAP50 `IQ3_M` build trades quality for fitting beside the existing workloads on one H200. Validate it on your own tasks before replacing a larger hosted model.
- Not a compliance certification. One layer in a privacy-by-design stack.

## License

[PolyForm Noncommercial License 1.0.0](./LICENSE.md). You can read, run, fork, modify, and share — **noncommercial use only**. Commercial use requires a separate license; reach out to `louis@screenpi.pe`.

Bundled / downloaded model weights keep their own licenses: `screenpipe/pii-redactor`, `screenpipe/pii-image-redactor` (`rfdetr_v19`), `google/gemma-4-E2B-it` (Gemma Terms of Use), and `patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF` (MIT). Using this deployment commercially means complying with all of them in addition to this repo's license.

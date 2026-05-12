# privacy-filter

GPU-backed HTTP wrapper around two PII models plus two co-hosted Gemma chat models, in one Tinfoil container:

1. [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter) — 1.5B-param MoE (50M active) token classifier for **text PII**. Endpoint `POST /filter`.
2. [`screenpipe/pii-image-redactor`](https://huggingface.co/screenpipe/pii-image-redactor) (`rfdetr_v9`) — RF-DETR-Nano detector for **image PII** in screenshots. Endpoint `POST /image/detect`.
3. **[v0.4.0+]** [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) — chat + vision via vLLM. Endpoint `POST /v1/chat/completions` with `model: "gemma4-31b"`. Weights come from Tinfoil's attested `/tinfoil/mpk` volume (same `mpk` as `tinfoilsh/confidential-gemma4-31b`).
4. **[v0.5.0+]** [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) — chat + vision + **audio** via vLLM. Endpoint `POST /v1/chat/completions` with `model: "gemma4-e4b"`. Weights baked into the image (~16 GB BF16) since Tinfoil hasn't wrapped E4B yet. E4B is the only Gemma 4 variant with native audio understanding.

All four workloads deploy inside the same [Tinfoil](https://tinfoil.sh) confidential-compute container on one H200 (~141 GB VRAM) so neither pixels, text, nor chat prompts leave an attested runtime. The shim only publishes uvicorn on `:8080`; `/v1/*` requests are reverse-proxied to the right vLLM on `127.0.0.1:8001` (31B) or `127.0.0.1:8002` (E4B) based on the request body's `model` field. One TLS-attested URL, one auth token, four models sharing one allocation.

## API

```
GET  /health         → {"status": "ok", "model_ready": true, "image_model_ready": true, ...}

POST /filter         → {"text": "My email is alice@foo.com"}
                    ←  {"redacted": "My email is [EMAIL]",
                        "spans": [{"label": "private_email", "start": 12, "end": 25,
                                   "text": "alice@foo.com", "score": 0.99}],
                        "latency_ms": 180,
                        "model": "openai/privacy-filter"}

POST /image/detect   → {"image_b64": "<b64-jpg-or-png>", "threshold": 0.30}
                    ←  {"detections": [{"bbox": [x, y, w, h], "label": "private_person", "score": 0.95},
                                       {"bbox": [x, y, w, h], "label": "secret",        "score": 0.91}],
                        "latency_ms": 32, "model": "rfdetr_v8",
                        "width": 2880, "height": 1800}
```

Bbox is `[x, y, w, h]` in ORIGINAL-image pixel space (the server un-resizes from its 320×320 internal input). Labels are the canonical 12-class screenpipe PII taxonomy.

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

First build pre-downloads the 1.5B text model (~3 GB bf16) AND the 108 MB rfdetr_v8 ONNX into the image, so expect a 5–10 min initial build. Subsequent builds hit Docker's layer cache. The image-model `ADD --checksum=` directive verifies the SHA-256 against the value pinned in the `Dockerfile` so a rebuild can't silently drift to a different upstream weight.

## Deploy to Tinfoil

1. **Push the image to a public registry** (GitHub Container Registry):

   ```bash
   VERSION=v0.1.0
   docker build -t ghcr.io/screenpipe/privacy-filter:$VERSION .
   docker push ghcr.io/screenpipe/privacy-filter:$VERSION

   # Grab the digest for tinfoil-config.yml (Tinfoil requires pinned digests).
   docker inspect --format='{{index .RepoDigests 0}}' \
     ghcr.io/screenpipe/privacy-filter:$VERSION
   ```

2. **Pin the digest** in `tinfoil-config.yml` — replace the `REPLACE_WITH_DIGEST`
   sentinel with the full `sha256:...` from the previous step. Commit and tag:

   ```bash
   git add tinfoil-config.yml
   git commit -m "release: privacy-filter $VERSION"
   git tag $VERSION && git push origin main --tags
   ```

3. **Click-through in the Tinfoil dashboard** (https://dash.tinfoil.sh):
   - create an org (if you haven't already)
   - connect the GitHub app to this repo
   - pick the tag to deploy
   - wait for status = `Running` (cold start ~30–60 s for the first model load)

4. **Verify** — the service is now reachable at
   `https://privacy-filter.<org>.containers.tinfoil.dev/health`.

## Resource sizing (GPU)

**Text model (OPF) — BF16 on CUDA:**

| Metric | Value |
|---|---|
| Weights (BF16) | ~3 GB VRAM |
| Active params per token | 50 M (MoE top-4 of 128 experts) |
| Attention window | 257 tokens (banded, O(N)) |
| H100 latency (512 tokens) | ~50–100 ms |
| H100 latency (~2 KB OCR row, 600 tok) | ~150–300 ms |

**Image model (rfdetr_v8) — TensorRT/CUDA EP:**

| Metric | Value |
|---|---|
| Weights (FP32 ONNX) | 108 MB |
| Params | ~25 M |
| Input resolution | 320×320 |
| H100 latency (per frame) | ~25–35 ms |
| Bench accuracy | 95.3% zero-leak / 0% oversmash on screenpipe-pii-bench-image val |

**Combined runtime** (32 GB CVM RAM + 80 GB H100 VRAM): ~5 GB GPU working set, ~30 req/sec short-text or ~30 frames/sec image, more if interleaved.

The CPU-only build (preserved on git history before v0.2.0) was the original deploy target — fits 8 vCPU / 32 GB RAM but at ~10× the latency. Switch back if cost dominates and the worker queue isn't backing up.

## Security properties

- **Tinfoil remote attestation** covers the exact image digest, so clients can verify the specific model bits + server code that handled their request.
- **Only `/health`, `/filter`, and `/image/detect` are exposed** — Tinfoil's `shim.paths` allowlist blocks every other URL at the enclave boundary, so no introspection / debug endpoints can leak.
- **Model weights are baked in.** Text model: build-time HF download then `TRANSFORMERS_OFFLINE=1`. Image model: build-time HF download with SHA-256 verification via Docker's `ADD --checksum=`. No runtime HuggingFace calls, so an attacker who subverts DNS can't swap weights out from under the enclave.
- **Runs as UID 10001 non-root** inside the container.

## Limitations

- **Text:** English-primary. Multilingual coverage per upstream model card varies.
- **Text:** 128K context upstream; we cap at `MAX_INPUT_TOKENS=8192` (override via env) to keep enclave memory bounded.
- **Image:** the rfdetr_v8 model was trained on screenpipe-pii-bench-image, which covers Slack / Outlook / Cursor / Terminal / Confluence / GitHub / 1Password / calendars / browsers. Apps with very different UI chrome (e.g. Zoom name overlays on video tiles) are not yet learned — failures should be added back into the bench's synthetic templates rather than fine-tuned on real captures.
- **Image:** payload cap `MAX_IMAGE_BYTES=20 MB` (override via env). Decoded RGB working buffer is ~3× the payload.
- Not a compliance certification. One layer in a privacy-by-design stack.

# screenpipe — AI that knows everything you've seen, said, or heard
# https://screenpipe.com
"""Exercise baked CPU weights, provider selection, startup, and both HTTP paths."""
import asyncio
import base64
import io
import json
import time

import httpx
from PIL import Image
import server


async def main():
    assert server.PRIVACY_ONLY, "CPU smoke must run in privacy-only mode"
    await server.app.router.startup()
    try:
        assert server._gemma_client is None and server._glm_client is None
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://cpu") as client:
            health = await client.get("/healthz")
            assert health.status_code == 200, health.text
            assert health.json()["text_provider"] == "CPUExecutionProvider"
            assert health.json()["image_provider"] == "CPUExecutionProvider"
            timings = {}
            text = "Contact Alice Smith at alice.smith@example.com or 415-555-0123."
            for _ in range(3):
                started = time.monotonic()
                response = await client.post("/filter", json={"text": text})
                assert response.status_code == 200, response.text
                assert "alice.smith@example.com" not in response.json()["redacted"], response.text
                timings.setdefault("text_ms", []).append(round((time.monotonic()-started)*1000, 1))
            for width, height in [(512, 512), (1512, 948)]:
                image = Image.new("RGB", (width, height), "white")
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                started = time.monotonic()
                response = await client.post("/image/detect", json={"image_b64": base64.b64encode(buffer.getvalue()).decode()})
                assert response.status_code == 200, response.text
                assert response.json()["width"] == width
                timings[f"image_{width}x{height}_ms"] = round((time.monotonic()-started)*1000, 1)
            print("CPU_SMOKE " + json.dumps(timings), flush=True)
    finally:
        await server.app.router.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

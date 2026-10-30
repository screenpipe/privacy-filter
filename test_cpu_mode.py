# screenpipe — AI that knows everything you've seen, said, or heard
# https://screenpipe.com
import unittest
from unittest.mock import patch
from fastapi import HTTPException
import server


class Session:
    def get_providers(self):
        return ["CPUExecutionProvider"]


class CpuModeTests(unittest.IsolatedAsyncioTestCase):
    async def test_cpu_health_requires_both_models_without_generative_upstreams(self):
        with patch.object(server, "PRIVACY_ONLY", True), patch.object(server, "_session", Session()), patch.object(server, "_image_session", Session()):
            self.assertEqual(server._ort_providers(), ["CPUExecutionProvider"])
            self.assertTrue((await server.healthz())["privacy_only"])
            with patch.object(server, "_image_session", None):
                with self.assertRaises(HTTPException) as failure:
                    await server.healthz()
                self.assertEqual(failure.exception.status_code, 503)

    async def test_cpu_mode_never_opens_chat_upstreams(self):
        with patch.object(server, "PRIVACY_ONLY", True), patch.object(server.httpx, "AsyncClient") as client:
            await server._open_gemma_proxy()
            await server._open_glm_proxy()
            client.assert_not_called()

    def test_gpu_provider_order_preserved(self):
        with patch.object(server, "PRIVACY_ONLY", False):
            self.assertEqual(server._ort_providers(), ["CUDAExecutionProvider", "CPUExecutionProvider"])

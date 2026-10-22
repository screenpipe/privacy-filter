# screenpipe — AI that knows everything you've seen, said, or heard
# https://screenpipe.com
"""Release lifecycle regressions: building must not advance client trust."""
import hashlib
import json
import unittest

import reconcile_tinfoil_release as release


class ReleaseLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {"snp_measurement": "a" * 96,
                         "tdx_measurement": {"rtmr1": "b" * 96, "rtmr2": "c" * 96}}
        self.text = json.dumps(self.manifest, indent=4)
        self.hardware = {"type": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
                         "registers": ["a" * 96]}
        self.old = {"id": 1, "published_at": "2026-09-01T00:00:00Z", "tag_name": "v1", "assets": [
            {"name": "tinfoil-deployment.json", "id": 1},
            {"name": "tinfoil.hash", "id": 2}]}
        self.new = {"id": 2, "published_at": "2026-09-02T00:00:00Z", "tag_name": "v2", "prerelease": True, "assets": [
            {"name": "tinfoil-deployment.json", "id": 3},
            {"name": "tinfoil.hash", "id": 4}]}
        self.calls = []
        self.latest = "v1"
        self.bad_signature = False
        self.bad_hardware = False
        self.bad_redaction = False
        self.stale_proxy = False

    def run_command(self, *args):
        self.calls.append(args)
        if args[:2] == ("gh", "api"):
            path = args[-1]
            if path.endswith("/assets/1"):
                return self.text
            if path.endswith("/assets/2"):
                return hashlib.sha256(self.text.encode()).hexdigest()
            if path.endswith("/assets/3"):
                return json.dumps({**self.manifest, "snp_measurement": "d" * 96})
            if path.endswith("/releases/latest"):
                return json.dumps(self.old if self.latest == "v1" else self.new)
        if args[:3] == ("gh", "attestation", "verify"):
            if self.bad_signature:
                raise RuntimeError("Sigstore signature rejected")
            self.assertIn("--cert-identity", args)
            self.assertIn("--cert-oidc-issuer", args)
            return json.dumps([{"verificationResult": {"statement": {"predicate": self.manifest}}}])
        if len(args) > 1 and args[1] == "newer":
            return json.dumps(args[2:] == ("v1", "v2"))
        if len(args) > 1 and args[1] == "verify":
            if self.bad_hardware:
                raise RuntimeError("SNP mismatch during deployment rotation")
            return json.dumps({"hardware_and_tls_verified": True,
                               "synthetic_pii_removed": not self.bad_redaction})
        if args[:3] == ("gh", "release", "edit"):
            self.latest = args[3]
            return ""
        if len(args) > 1 and args[1] == "latest":
            if self.stale_proxy:
                raise RuntimeError("Tinfoil proxy still selects previous release")
            return "Latest verified"
        raise AssertionError(args)

    def edits(self):
        return [c for c in self.calls if c[:3] == ("gh", "release", "edit")]

    def test_new_build_not_deployed_keeps_live_release(self):
        selected = release.reconcile(self.hardware, [self.new, self.old], self.old, True, self.run_command)
        self.assertEqual(selected["tag"], "v1")
        self.assertFalse(self.edits())

    def test_actual_new_deployment_promotes_automatically(self):
        # The same signed candidate now runs in hardware and is eligible.
        deployed = {**self.new, "assets": self.old["assets"]}
        selected = release.reconcile(self.hardware, [deployed, self.old], self.old, True, self.run_command)
        self.assertEqual(selected["tag"], "v2")
        self.assertEqual(self.latest, "v2")

    def test_older_signed_deployment_does_not_authorize_rollback(self):
        with self.assertRaisesRegex(RuntimeError, "No signed release"):
            release.reconcile(self.hardware, [self.old], self.new, True, self.run_command)
        self.assertFalse(self.edits())

    def test_legacy_repair_is_exact_and_not_a_general_rollback(self):
        previous = {**self.old, "id": release.LEGACY_REPAIR[1]}
        incorrect_latest = {**self.new, "id": release.LEGACY_REPAIR[0]}
        self.assertTrue(release.eligible(previous, incorrect_latest, self.run_command))
        self.assertFalse(release.eligible({**previous, "id": 999}, incorrect_latest, lambda *args: "false"))
        # Even the exact metadata pair cannot authorize a different artifact.
        with self.assertRaisesRegex(ValueError, "legacy repair artifact changed"):
            release.reconcile(self.hardware, [previous], incorrect_latest, True, self.run_command)
        self.assertFalse(self.edits())

    def test_read_only_check_never_changes_metadata(self):
        release.reconcile(self.hardware, [self.old], self.old, False, self.run_command)
        self.assertFalse(self.edits())

    def test_unknown_hardware_does_not_change_latest(self):
        self.hardware["registers"] = ["e" * 96]
        with self.assertRaisesRegex(RuntimeError, "No signed release"):
            release.reconcile(self.hardware, [self.new, self.old], self.old, True, self.run_command)
        self.assertFalse(self.edits())

    def test_provenance_rotation_and_redaction_failures_block_promotion(self):
        for fault in ("bad_signature", "bad_hardware", "bad_redaction"):
            with self.subTest(fault=fault):
                setattr(self, fault, True)
                try:
                    with self.assertRaises((RuntimeError, ValueError)):
                        release.reconcile(self.hardware, [self.old], self.old, True, self.run_command)
                    self.assertFalse(self.edits())
                finally:
                    setattr(self, fault, False)

    def test_stale_proxy_is_reported_then_next_run_recovers(self):
        self.stale_proxy = True
        with self.assertRaisesRegex(RuntimeError, "proxy still selects"):
            release.reconcile(self.hardware, [self.old], self.old, True, self.run_command)
        self.stale_proxy = False
        selected = release.reconcile(self.hardware, [self.old], self.old, True, self.run_command)
        self.assertEqual(selected["tag"], "v1")

    def test_concurrent_production_change_does_not_get_overwritten(self):
        def changed(*args):
            if args[:2] == ("gh", "api") and args[-1].endswith("/releases/latest"):
                return json.dumps(self.new)
            return self.run_command(*args)
        with self.assertRaisesRegex(RuntimeError, "selection changed during verification"):
            release.reconcile(self.hardware, [self.old], self.old, True, changed)
        self.assertFalse(self.edits())

    def test_malformed_hardware_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unsupported hardware"):
            release.matches(self.manifest, {"type": "unknown", "registers": []})

    def test_pagination_has_no_history_cutoff(self):
        # Discovery continues past a full API page of unmatching releases.
        selected = release.reconcile(self.hardware, [self.new] * 100 + [self.old], self.old, False, self.run_command)
        self.assertEqual(selected["tag"], "v1")

    def test_artifact_digest_mismatch_blocks_promotion(self):
        def tampered(*args):
            if args[-1].endswith("/assets/2"):
                return "0" * 64
            return self.run_command(*args)
        with self.assertRaisesRegex(ValueError, "does not match tinfoil.hash"):
            release.reconcile(self.hardware, [self.old], self.old, True, tampered)
        self.assertFalse(self.edits())

    def test_signed_claim_must_agree_with_artifact(self):
        def wrong_predicate(*args):
            if args[:3] == ("gh", "attestation", "verify"):
                return json.dumps([{"verificationResult": {"statement": {"predicate": {}}}}])
            return self.run_command(*args)
        with self.assertRaisesRegex(ValueError, "verified predicate differs"):
            release.reconcile(self.hardware, [self.old], self.old, True, wrong_predicate)
        self.assertFalse(self.edits())


if __name__ == "__main__":
    unittest.main()

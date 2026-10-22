#!/usr/bin/env python3
# screenpipe — AI that knows everything you've seen, said, or heard
# https://screenpipe.com
"""Make GitHub latest describe the verified live enclave, never the newest build.

Default is read-only. --apply is used by the automatic production workflow.
No inference data or credentials are accepted as arguments.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
import subprocess
import tempfile

REPO = "screenpipe/privacy-filter"
PREDICATE = "https://tinfoil.sh/predicate/snp-tdx-multiplatform/v1"
ROOT = Path(__file__).resolve().parents[1]
# One migration repair for the independently verified incident: the newer
# undeployed RC became latest. Never extend this into a historical allowlist.
LEGACY_REPAIR = (391885162, 391857682,
                 "cffeb7831df606698c4823483e4e2314369fc5c1cc3d8cb69f617c9c678f559b")


def command(*args):
    return subprocess.check_output(args, text=True)


def gh_json(path):
    return json.loads(command("gh", "api", path))


def matches(manifest, hardware):
    if hardware["type"] == "https://tinfoil.sh/predicate/sev-snp-guest/v2":
        return hardware["registers"] == [manifest.get("snp_measurement")]
    raise ValueError("unsupported hardware format; refusing to select a release")


def verified_candidate(release, hardware, run=command):
    assets = {asset["name"]: asset for asset in release["assets"]}
    if not {"tinfoil-deployment.json", "tinfoil.hash"} <= assets.keys():
        return None
    # Only manifests from this fixed repository are considered. This initial
    # comparison is discovery, not trust: cryptographic verification follows.
    manifest_text = run("gh", "api", "-H", "Accept: application/octet-stream",
                        f"repos/{REPO}/releases/assets/{assets['tinfoil-deployment.json']['id']}")
    manifest = json.loads(manifest_text)
    if not matches(manifest, hardware):
        return None
    digest = run("gh", "api", "-H", "Accept: application/octet-stream",
                 f"repos/{REPO}/releases/assets/{assets['tinfoil.hash']['id']}").strip()
    if hashlib.sha256(manifest_text.encode()).hexdigest() != digest:
        raise ValueError(f"{release['tag_name']}: manifest does not match tinfoil.hash")
    tag = release["tag_name"]
    with tempfile.TemporaryDirectory(prefix="tinfoil-release-") as directory:
        path = Path(directory) / "tinfoil-deployment.json"
        path.write_text(manifest_text)
        verified = json.loads(run(
            "gh", "attestation", "verify", str(path), "--repo", REPO,
            "--cert-identity", f"https://github.com/{REPO}/.github/workflows/tinfoil-build.yml@refs/tags/{tag}",
            "--cert-oidc-issuer", "https://token.actions.githubusercontent.com",
            "--predicate-type", PREDICATE, "--format", "json"))
        # Use the authenticated claim, not an unauthenticated payload decoded
        # from a bundle. Require the measured artifact and signed claim to agree.
        if not any(item["verificationResult"]["statement"]["predicate"] == manifest
                   for item in verified):
            raise ValueError(f"{tag}: verified predicate differs from manifest")
        proof = json.loads(run(str(ROOT / "tools/tinfoil-release-check/target/release/tinfoil-release-check"),
                               "verify", str(path)))
        if proof.get("hardware_and_tls_verified") is not True or proof.get("synthetic_pii_removed") is not True:
            raise ValueError(f"{tag}: incomplete live verification")
    return {"tag": tag, "digest": digest, "proof": proof}


def eligible(release, current, run=command):
    if release["id"] == current["id"]:
        return True
    if (current["id"], release["id"]) == LEGACY_REPAIR[:2]:
        return True
    # Never silently authorize a rollback to arbitrary historical signed code.
    # Use Git ancestry rather than mutable release publication timestamps or
    # assumptions about which forms of semantic version tags are in use.
    comparison = json.loads(run("gh", "api", f"repos/{REPO}/compare/"
                                f"{quote(current['tag_name'], safe='')}...{quote(release['tag_name'], safe='')}"))
    return comparison["status"] in ("ahead", "identical")


def reconcile(hardware, releases, current, apply=False, run=command):
    for release in releases:
        if release.get("draft") or not eligible(release, current, run):
            continue
        candidate = verified_candidate(release, hardware, run)
        if candidate is None:
            continue
        if (current["id"], release["id"]) == LEGACY_REPAIR[:2] and candidate["digest"] != LEGACY_REPAIR[2]:
            raise ValueError("legacy repair artifact changed; refusing promotion")
        # No arbitrary fallback: a release is selected only after its signed
        # source, AMD launch measurement and live pinned PII request all pass.
        print(json.dumps({"verified_live_release": candidate, "apply": apply}), flush=True)
        if apply:
            latest = json.loads(run("gh", "api", f"repos/{REPO}/releases/latest"))
            if latest["id"] != current["id"]:
                raise RuntimeError("Production selection changed during verification; retry next run")
            if release["id"] != current["id"]:
                run("gh", "release", "edit", candidate["tag"], "--repo", REPO,
                    "--prerelease=false", "--latest=true")
            latest = json.loads(run("gh", "api", f"repos/{REPO}/releases/latest"))
            if latest["tag_name"] != candidate["tag"]:
                raise RuntimeError("GitHub latest did not converge to the verified deployment")
            # Exercise the unmodified client selection contract, including the
            # Tinfoil GitHub proxy, after promotion. A stale proxy is a visible
            # failure; the next scheduled reconciliation retries durably.
            run(str(ROOT / "tools/tinfoil-release-check/target/release/tinfoil-release-check"), "latest")
        return candidate
    raise RuntimeError("No signed release matches the live enclave; latest left unchanged")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    binary = ROOT / "tools/tinfoil-release-check/target/release/tinfoil-release-check"
    hardware = json.loads(command(str(binary), "inspect"))
    # GitHub pages are transport batches, not an accepted-history cutoff.
    pages = json.loads(command("gh", "api", "--paginate", "--slurp", f"repos/{REPO}/releases"))
    reconcile(hardware, (release for page in pages for release in page),
              gh_json(f"repos/{REPO}/releases/latest"), args.apply)


if __name__ == "__main__":
    main()

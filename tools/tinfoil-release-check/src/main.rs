// screenpipe — AI that knows everything you've seen, said, or heard
// https://screenpipe.com
use serde_json::{json, Value};
use tinfoil::verifier::attestation::{self, Measurement, PredicateType};

const HOST: &str = "pii.screenpipe.containers.tinfoil.dev";
const REPO: &str = "screenpipe/privacy-filter";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Public low-level fetch helpers need the SDK's process crypto provider.
    let _init = tinfoil::SecureClient::new(HOST, REPO, "");
    let args: Vec<_> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("newer") if args.len() == 4 => {
            println!("{}", newer_release(&args[2], &args[3])?);
        }
        Some("inspect") => {
            let report = attestation::fetch(HOST).await?;
            let verified = attestation::verify_full(&report).await?;
            println!("{}", serde_json::to_string(&verified.measurement)?);
        }
        Some("latest") => {
            tinfoil::Client::new(HOST, REPO, "").await?;
            println!("Latest selection, source provenance, hardware and TLS verified");
        }
        Some("verify") if args.len() == 3 || args.len() == 4 => {
            let host = args.get(3).map(String::as_str).unwrap_or(HOST);
            // The caller first authenticates this file with gh attestation verify,
            // checking the exact repository, workflow/tag identity and digest.
            // A pinned client alone does NOT prove source provenance.
            let manifest: Value = serde_json::from_slice(&std::fs::read(&args[2])?)?;
            let measurement = manifest_measurement(&manifest)?;
            let mut client = tinfoil::SecureClient::with_measurement(host, "", measurement);
            client.verify().await?; // AMD chain, register equality, TLS/HPKE/report binding.
            let response = client.http_client()?.post(format!("https://{host}/filter"))
                .json(&json!({"text": "The weather is sunny. Contact alice@example.com.", "include_spans": false}))
                .send().await?.error_for_status()?;
            let body: Value = response.json().await?;
            let redacted = validate_redaction(&body)?;
            println!(
                "{}",
                json!({"hardware_and_tls_verified":true,"synthetic_pii_removed":true,"redacted":redacted})
            );
        }
        _ => {
            return Err(
                "usage: tinfoil-release-check inspect | latest | newer <current-tag> <candidate-tag> | verify <verified-manifest.json> [review-host]"
                    .into(),
            )
        }
    }
    Ok(())
}

fn newer_release(current: &str, candidate: &str) -> Result<bool, semver::Error> {
    let version = |tag: &str| semver::Version::parse(tag.strip_prefix('v').unwrap_or(tag));
    Ok(version(candidate)?
        .cmp_precedence(&version(current)?)
        .is_gt())
}

fn manifest_measurement(manifest: &Value) -> Result<Measurement, Box<dyn std::error::Error>> {
    let register = |value: &Value| -> Result<String, Box<dyn std::error::Error>> {
        let text = value.as_str().ok_or("missing measurement register")?;
        // SNP and TDX registers are 48-byte SHA-384 values, not display fingerprints.
        if text.len() != 96 || !text.bytes().all(|c| c.is_ascii_hexdigit()) {
            return Err("invalid 48-byte measurement register".into());
        }
        Ok(text.to_owned())
    };
    Ok(Measurement {
        type_: PredicateType::SnpTdxMultiPlatformV1,
        registers: vec![
            register(&manifest["snp_measurement"])?,
            register(&manifest["tdx_measurement"]["rtmr1"])?,
            register(&manifest["tdx_measurement"]["rtmr2"])?,
        ],
    })
}

fn validate_redaction(body: &Value) -> Result<&str, Box<dyn std::error::Error>> {
    let redacted = body["redacted"].as_str().ok_or("missing redacted output")?;
    if !redacted.contains("weather") || redacted.contains("alice@example.com") {
        return Err("synthetic PII check failed: marker missing or email exposed".into());
    }
    Ok(redacted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_release_order_is_independent_of_branch_ancestry() {
        assert!(newer_release("v0.9.10-rc1", "v0.9.11-rc1").unwrap());
        assert!(newer_release("v0.9.10-rc1", "v0.10.0").unwrap());
        assert!(newer_release("v0.9.11-rc1", "v0.9.11").unwrap());
        assert!(!newer_release("v0.9.11-rc1", "v0.9.10-rc1").unwrap());
        assert!(!newer_release("v0.9.11", "v0.9.11-rc1").unwrap());
        assert!(!newer_release("v0.9.11+first", "v0.9.11+second").unwrap());
        assert!(newer_release("v0.9.11", "unversioned").is_err());
    }

    #[test]
    fn missing_or_invalid_registers_cannot_build_a_trusted_measurement() {
        let mut manifest = json!({"snp_measurement": "a".repeat(96),
            "tdx_measurement": {"rtmr1": "b".repeat(96), "rtmr2": "c".repeat(96)}});
        assert!(manifest_measurement(&manifest).is_ok());
        for bad in [
            Value::Null,
            json!("a".repeat(95)),
            json!("a".repeat(97)),
            json!("z".repeat(96)),
        ] {
            manifest["snp_measurement"] = bad;
            assert!(manifest_measurement(&manifest).is_err());
        }
    }

    #[test]
    fn pii_smoke_requires_both_useful_text_and_redaction() {
        assert!(
            validate_redaction(&json!({"redacted":"The weather is sunny. Contact [EMAIL]."}))
                .is_ok()
        );
        for bad in [
            json!({}),
            json!({"redacted":""}),
            json!({"redacted":"The weather is sunny. Contact alice@example.com."}),
            json!({"redacted":"[EVERYTHING]"}),
        ] {
            assert!(validate_redaction(&bad).is_err());
        }
    }
}

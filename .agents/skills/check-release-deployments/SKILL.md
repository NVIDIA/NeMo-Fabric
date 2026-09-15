---
name: check-release-deployments
description: Verify whether a tagged NeMo Fabric release is deployed to crates.io, PyPI, and npm. Use when checking publication status for a specific release tag; not for publishing packages or creating tags.
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---

# Check Release Deployments

Require the release tag as input. Do not infer the tag from the current branch,
Git history, or package metadata.

Run the checker from the repository root:

```bash
bash .agents/skills/check-release-deployments/scripts/check_release_deployments.sh <tag>
```

The checker converts the tag independently for Cargo, Python, and Node.js;
discovers the tracked package manifests in their respective source trees; and
reports every matching GitHub Actions release workflow, except `Request NVSkills
CI`, before the package deployment table. For nightly alpha tags, it reports the
scheduled or manually dispatched nightly workflow run for the tag's commit and
UTC date instead of unrelated push runs at the same commit. If the `Publish Rust
crates`, `Python`, or `Publish TypeScript package` workflow concludes with
`failure`, the corresponding Cargo, Python, or Node.js registry checks are
skipped and the table states why. For Python packages, it uses PyPI's JSON Index API at
`/simple/<project>/` and verifies that the normalized version is in the
response's `versions` list. It waits at least 0.1 seconds before every request
to a registry website or GitHub API call to avoid rate limits. For Node.js
packages, it uses npm's registry API at
`https://registry.npmjs.org/<package>/<version>`. For Cargo packages, it uses
crates.io's versioned registry API at
`https://crates.io/api/v1/crates/<crate>/<version>` and excludes the internal
`crates/fabric-python/Cargo.toml` binding manifest.

Interpret the table status column as follows:

- `☑` indicates the exact package version is deployed.
- `○` indicates the exact package version is not deployed.
- Any other value is the HTTP status code returned by the registry.

The workflow is read-only. Do not retry failed or undeployed packages by
publishing them unless the user separately asks to publish.

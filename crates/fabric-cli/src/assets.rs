// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Embedded assets used by installation-safe CLI experiments.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// One embedded file and its path relative to a staged asset root.
#[derive(Debug, Clone, Copy)]
pub struct EmbeddedFile {
    /// Relative path written under the staged root.
    pub path: &'static str,
    /// UTF-8 file contents.
    pub contents: &'static str,
}

/// Temporary asset root retained for the lifetime of a selected experiment.
#[derive(Debug)]
pub struct StagedAssets {
    path: PathBuf,
}

impl StagedAssets {
    /// Stage embedded files and return their temporary base directory.
    pub fn create(files: &[EmbeddedFile]) -> std::io::Result<Self> {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("nemo-fabric-{}-{nonce}", std::process::id()));
        fs::create_dir(&path)?;
        for file in files {
            let destination = path.join(file.path);
            if let Some(parent) = destination.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(destination, file.contents)?;
        }
        Ok(Self { path })
    }

    /// Return the staged base directory.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for StagedAssets {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stages_and_removes_embedded_files() {
        let root_path;
        {
            let root = StagedAssets::create(&[EmbeddedFile {
                path: "nested/value.txt",
                contents: "value",
            }])
            .expect("stage assets");
            root_path = root.path().to_path_buf();
            assert_eq!(
                fs::read_to_string(root.path().join("nested/value.txt")).expect("read asset"),
                "value"
            );
        }
        assert!(!root_path.exists());
    }
}

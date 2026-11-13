// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Explicit selection of durable harness-native sessions.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::config::{RunPlan, is_one_folder_name};
use crate::{FabricError, Result};

/// Version of the portable session reference envelope.
pub const SESSION_REF_SCHEMA_VERSION: &str = "fabric.session_ref/v1alpha1";

/// Adapter-issued native identity, resolved only within caller-supplied storage.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SessionRef {
    /// Supported envelope version.
    #[schemars(extend("const" = "fabric.session_ref/v1alpha1"))]
    pub schema_version: String,
    /// Adapter responsible for interpreting the opaque identifier.
    #[schemars(length(min = 1), regex(pattern = r"\S"))]
    pub adapter_id: String,
    /// Opaque adapter-issued identifier interpreted by the adapter.
    #[schemars(length(min = 1), regex(pattern = r"\S"))]
    pub session_id: String,
}

impl SessionRef {
    /// Validate the envelope without parsing the native identifier.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != SESSION_REF_SCHEMA_VERSION
            || self.adapter_id.trim().is_empty()
            || self.session_id.trim().is_empty()
        {
            return Err(session_error(
                "session_ref_invalid",
                "invalid session reference envelope",
            ));
        }
        Ok(())
    }
}

impl std::fmt::Debug for SessionRef {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SessionRef")
            .field("schema_version", &self.schema_version)
            .field("adapter_id", &self.adapter_id)
            .field("session_id", &"[redacted]")
            .finish()
    }
}

pub(crate) fn session_error(code: &str, message: &str) -> FabricError {
    FabricError::Session {
        code: code.to_string(),
        message: message.to_string(),
    }
}

pub(crate) fn durable(plan: &RunPlan) -> bool {
    plan.config.runtime.keep_state && plan.capabilities.session_resume
}

pub(crate) fn selection(plan: &RunPlan) -> Result<Option<SessionRef>> {
    let selected = plan.session_ref.clone();
    if selected.is_none() && !durable(plan) {
        return Ok(None);
    }
    if !plan.capabilities.session_resume
        || !plan
            .adapter_descriptor
            .as_ref()
            .is_some_and(|adapter| adapter.descriptor.capabilities.session_resume)
    {
        return Err(session_error(
            "session_resume_unsupported",
            "adapter does not support durable sessions",
        ));
    }
    if !plan.config.runtime.keep_state {
        return Err(session_error(
            "session_storage_required",
            "durable sessions require runtime.keep_state and retained storage",
        ));
    }
    if let Some(session_ref) = &selected {
        session_ref.validate()?;
        if Some(session_ref.adapter_id.as_str())
            != plan
                .adapter_descriptor
                .as_ref()
                .map(|adapter| adapter.descriptor.adapter_id.as_str())
        {
            return Err(session_error(
                "session_adapter_mismatch",
                "session reference targets another adapter",
            ));
        }
    }
    Ok(selected)
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StorageBinding {
    session_ref: SessionRef,
    home: String,
}

fn adapter_root(root: &Path, adapter: &str) -> Result<PathBuf> {
    if !is_one_folder_name(adapter) {
        return Err(session_error(
            "session_ref_invalid",
            "adapter identity is not a valid storage namespace",
        ));
    }
    Ok(root.join(".fabric").join("state").join(adapter))
}

fn index_path(root: &Path, reference: &SessionRef) -> Result<PathBuf> {
    let digest = Sha256::digest(reference.session_id.as_bytes());
    Ok(adapter_root(root, &reference.adapter_id)?
        .join(".sessions")
        .join(format!("{digest:x}.json")))
}

fn checked_directory(root: &Path, path: &Path, create: bool) -> Result<()> {
    let invalid = || {
        session_error(
            "session_incompatible",
            "session index leaves the supplied storage namespace",
        )
    };
    let mut directory = root.to_path_buf();
    for component in path.strip_prefix(root).map_err(|_| invalid())?.components() {
        directory.push(component);
        if create {
            match std::fs::create_dir(&directory) {
                Ok(()) => (),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => (),
                Err(_) => return Err(invalid()),
            }
        }
        let metadata = std::fs::symlink_metadata(&directory).map_err(|_| {
            session_error(
                "session_not_found",
                "session was not found in the supplied retained storage",
            )
        })?;
        if !metadata.is_dir() || metadata.file_type().is_symlink() {
            return Err(invalid());
        }
    }
    Ok(())
}

fn checked_index(root: &Path, reference: &SessionRef, create: bool) -> Result<PathBuf> {
    let root = root
        .canonicalize()
        .map_err(|_| session_error("session_not_found", "retained storage was not found"))?;
    let index = index_path(&root, reference)?;
    checked_directory(&root, index.parent().expect("index parent"), create)?;
    Ok(index)
}

/// Resolve one physical home before locking or starting any native writer.
pub(crate) fn prepare_storage_home(root: &Path, adapter: &str, home: &Path) -> Result<PathBuf> {
    let invalid = || {
        session_error(
            "session_incompatible",
            "native session storage leaves the supplied namespace",
        )
    };
    let root = root.canonicalize().map_err(|_| invalid())?;
    let namespace = adapter_root(&root, adapter)?;
    checked_directory(&root, &namespace, true)?;
    if home
        .parent()
        .and_then(|parent| parent.canonicalize().ok())
        .as_ref()
        != Some(&namespace)
    {
        return Err(invalid());
    }
    match std::fs::symlink_metadata(home) {
        Ok(_) => (),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            std::fs::create_dir(home).map_err(|_| invalid())?;
        }
        Err(_) => return Err(invalid()),
    }
    let home = home.canonicalize().map_err(|_| invalid())?;
    if home.parent() != Some(namespace.as_path()) || !home.is_dir() {
        return Err(invalid());
    }
    Ok(home)
}

pub(crate) fn resolve_storage(root: &Path, reference: &SessionRef) -> Result<PathBuf> {
    let missing = || {
        session_error(
            "session_not_found",
            "session was not found in the supplied retained storage",
        )
    };
    let path = checked_index(root, reference, false)?;
    let mut options = std::fs::File::options();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options.open(&path).map_err(|_| missing())?;
    if !file.metadata().map_err(|_| missing())?.is_file() {
        return Err(session_error(
            "session_incompatible",
            "session storage binding is not a regular file",
        ));
    }
    let mut contents = String::new();
    file.take(1024 * 1024)
        .read_to_string(&mut contents)
        .map_err(|_| missing())?;
    let binding: StorageBinding = serde_json::from_str(&contents)
        .map_err(|_| session_error("session_incompatible", "session storage binding is invalid"))?;
    if binding.session_ref != *reference || !is_one_folder_name(&binding.home) {
        return Err(session_error(
            "session_incompatible",
            "session storage binding does not match the requested identity",
        ));
    }
    let allowed = adapter_root(root, &reference.adapter_id)?
        .canonicalize()
        .map_err(|_| missing())?;
    let root = root.canonicalize().map_err(|_| missing())?;
    let path = allowed
        .join(binding.home)
        .canonicalize()
        .map_err(|_| missing())?;
    if !allowed.starts_with(&root) || !path.starts_with(&allowed) {
        return Err(session_error(
            "session_incompatible",
            "session storage leaves the supplied namespace",
        ));
    }
    Ok(path)
}

pub(crate) fn register_storage(root: &Path, reference: &SessionRef, home: &Path) -> Result<()> {
    let invalid = || {
        session_error(
            "session_incompatible",
            "native session storage is outside the configured namespace",
        )
    };
    let allowed = adapter_root(root, &reference.adapter_id)?
        .canonicalize()
        .map_err(|_| invalid())?;
    let root = root.canonicalize().map_err(|_| invalid())?;
    let home = home.canonicalize().map_err(|_| invalid())?;
    let relative = home.strip_prefix(&allowed).map_err(|_| invalid())?;
    let name = relative
        .to_str()
        .filter(|name| is_one_folder_name(name))
        .ok_or_else(invalid)?;
    if !allowed.starts_with(&root) {
        return Err(invalid());
    }
    let binding = StorageBinding {
        session_ref: reference.clone(),
        home: name.to_string(),
    };
    let path = checked_index(&root, reference, true)?;
    let bytes = serde_json::to_vec(&binding).map_err(FabricError::SerializeJson)?;
    let temporary = path.with_extension(format!("{}.tmp", crate::runtime::new_id("binding")));
    let mut options = std::fs::File::options();
    options.create_new(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    }
    // Only remove a temporary file created by this call.
    let mut file = options.open(&temporary).map_err(|_| invalid())?;
    let result = (|| {
        file.write_all(&bytes)
            .and_then(|()| file.sync_all())
            .map_err(|_| invalid())?;
        // Hard-linking publishes a complete record without replacing another session's binding.
        std::fs::hard_link(&temporary, &path).map_err(|_| {
            session_error(
                "session_identity_changed",
                "native session identity already has a storage binding",
            )
        })?;
        #[cfg(unix)]
        std::fs::File::open(path.parent().expect("index parent"))
            .and_then(|file| file.sync_all())
            .map_err(|_| invalid())?;
        Ok(())
    })();
    let _ = std::fs::remove_file(&temporary);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestRoot(PathBuf);
    impl TestRoot {
        fn new() -> Self {
            let path = std::env::temp_dir().join(crate::runtime::new_id("fabric-session-test"));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TestRoot {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn storage_binding_survives_moving_the_retained_root() {
        let root = TestRoot::new();
        let original = root.path().join("original");
        let reference = SessionRef {
            schema_version: SESSION_REF_SCHEMA_VERSION.into(),
            adapter_id: "nvidia.fabric.hermes".into(),
            session_id: "private-native-id".into(),
        };
        let home = adapter_root(&original, &reference.adapter_id)
            .unwrap()
            .join("assitant");
        std::fs::create_dir_all(&home).unwrap();
        register_storage(&original, &reference, &home).unwrap();
        let moved = root.path().join("moved");
        std::fs::rename(&original, &moved).unwrap();
        assert_eq!(
            resolve_storage(&moved, &reference).unwrap(),
            adapter_root(&moved, &reference.adapter_id)
                .unwrap()
                .join("assitant")
                .canonicalize()
                .unwrap()
        );
    }

    #[cfg(unix)]
    #[test]
    fn storage_index_cannot_escape_through_a_symlink() {
        let root = TestRoot::new();
        let outside = TestRoot::new();
        let reference = SessionRef {
            schema_version: SESSION_REF_SCHEMA_VERSION.into(),
            adapter_id: "nvidia.fabric.hermes".into(),
            session_id: "private-native-id".into(),
        };
        let home = adapter_root(root.path(), &reference.adapter_id)
            .unwrap()
            .join("agent");
        std::fs::create_dir_all(&home).unwrap();
        let index = index_path(root.path(), &reference).unwrap();
        std::os::unix::fs::symlink(outside.path(), index.parent().unwrap()).unwrap();
        assert!(register_storage(root.path(), &reference, &home).is_err());
        assert_eq!(std::fs::read_dir(outside.path()).unwrap().count(), 0);
        std::fs::write(
            outside.path().join(index.file_name().unwrap()),
            serde_json::to_vec(&StorageBinding {
                session_ref: reference.clone(),
                home: "agent".into(),
            })
            .unwrap(),
        )
        .unwrap();
        assert!(resolve_storage(root.path(), &reference).is_err());
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Private, per-runtime request reservations and completed invocation results.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::AdapterLifecycleOperation;
use super::{InvocationHandle, RunRequest, RunResult, RuntimeHandle, lifecycle_error};
use crate::Result;

const MAX_RESULT_BYTES: u64 = 4 * 1024 * 1024;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Reservation {
    agent_name: String,
    runtime_id: String,
    request_id: String,
    invocation_id: String,
    fingerprint: String,
    #[serde(default)]
    streaming: Option<bool>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Finalization {
    agent_name: String,
    runtime_id: String,
    request_id: String,
}

pub(super) enum RequestReservation {
    Cached(Box<RunResult>),
    Pending(PendingReservation),
}

pub(super) struct PendingReservation {
    path: PathBuf,
    dispatched: bool,
}

impl PendingReservation {
    pub(super) fn mark_dispatched(&mut self) {
        self.dispatched = true;
    }
}

impl Drop for PendingReservation {
    fn drop(&mut self) {
        if !self.dispatched {
            // This guard owns the file created with create_new. Other requests
            // cannot acquire this request ID until the file has been removed.
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn failure(runtime_id: &str, code: &str) -> crate::FabricError {
    lifecycle_error(
        AdapterLifecycleOperation::Invoke,
        runtime_id,
        code,
        "The invocation record is unavailable or conflicts with this request; do not replay uncertain work.",
        "",
    )
}

fn leaf(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
}

fn trusted_ancestry(root: &Path, runtime_id: &str) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        // SAFETY: geteuid has no preconditions or side effects.
        let uid = unsafe { libc::geteuid() };
        for ancestor in root.ancestors() {
            let metadata = fs::symlink_metadata(ancestor)
                .map_err(|_| failure(runtime_id, "result_storage_failed"))?;
            if !metadata.is_dir()
                || !matches!(metadata.uid(), owner if owner == uid || owner == 0)
                || (metadata.mode() & 0o022 != 0 && metadata.mode() & 0o1000 == 0)
            {
                return Err(failure(runtime_id, "result_storage_failed"));
            }
        }
    }
    #[cfg(not(unix))]
    let _ = (root, runtime_id);
    Ok(())
}

fn owned_metadata(metadata: &fs::Metadata, file: bool) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        // SAFETY: geteuid has no preconditions or side effects.
        let forbidden = if file { 0o077 } else { 0o022 };
        metadata.uid() == unsafe { libc::geteuid() } && metadata.mode() & forbidden == 0
    }
    #[cfg(not(unix))]
    {
        let _ = (metadata, file);
        true
    }
}

pub(super) fn create_artifact_root(root: &Path) -> std::io::Result<()> {
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(root)
}

fn directory(parent: &Path, name: &str, create: bool, runtime_id: &str) -> Result<PathBuf> {
    let path = parent.join(name);
    if create {
        let builder = fs::DirBuilder::new();
        #[cfg(unix)]
        let builder = {
            use std::os::unix::fs::DirBuilderExt;
            let mut builder = builder;
            builder.mode(0o700);
            builder
        };
        if let Err(error) = builder.create(&path)
            && error.kind() != std::io::ErrorKind::AlreadyExists
        {
            return Err(failure(runtime_id, "result_storage_failed"));
        }
    }
    let metadata =
        fs::symlink_metadata(&path).map_err(|_| failure(runtime_id, "result_storage_failed"))?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() || !owned_metadata(&metadata, false)
    {
        return Err(failure(runtime_id, "result_storage_failed"));
    }
    Ok(path)
}

fn runtime_directory(root: &Path, runtime_id: &str, create: bool) -> Result<PathBuf> {
    if !leaf(runtime_id) {
        return Err(failure(runtime_id, "invalid_result_target"));
    }
    if create {
        create_artifact_root(root).map_err(|_| failure(runtime_id, "result_storage_failed"))?;
    }
    let root = fs::canonicalize(root).map_err(|_| failure(runtime_id, "result_storage_failed"))?;
    trusted_ancestry(&root, runtime_id)?;
    let fabric = directory(&root, ".fabric", create, runtime_id)?;
    directory(&fabric, runtime_id, create, runtime_id)
}

pub(super) fn invocation_directory(
    root: &Path,
    runtime_id: &str,
    invocation_id: &str,
) -> Result<PathBuf> {
    if !leaf(invocation_id) {
        return Err(failure(runtime_id, "invalid_result_target"));
    }
    let runtime = runtime_directory(root, runtime_id, true)?;
    directory(&runtime, invocation_id, true, runtime_id)
}

fn options() -> OpenOptions {
    let options = OpenOptions::new();
    #[cfg(unix)]
    let options = {
        use std::os::unix::fs::OpenOptionsExt;
        let mut options = options;
        options
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        options
    };
    options
}

fn read<T: serde::de::DeserializeOwned>(path: &Path, runtime_id: &str) -> Result<Option<T>> {
    let file = match options().read(true).open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err(failure(runtime_id, "result_storage_failed")),
    };
    let metadata = file
        .metadata()
        .map_err(|_| failure(runtime_id, "result_storage_failed"))?;
    if !metadata.is_file() || metadata.len() > MAX_RESULT_BYTES || !owned_metadata(&metadata, true)
    {
        return Err(failure(runtime_id, "result_storage_failed"));
    }
    let mut bytes = Vec::new();
    file.take(MAX_RESULT_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| failure(runtime_id, "result_storage_failed"))?;
    if bytes.len() as u64 > MAX_RESULT_BYTES {
        return Err(failure(runtime_id, "result_storage_failed"));
    }
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(|_| failure(runtime_id, "result_storage_failed"))
}

fn index_name(request_id: &str) -> String {
    format!("{:x}.json", Sha256::digest(request_id.as_bytes()))
}

fn finalization_path(requests: &Path, request_id: &str) -> PathBuf {
    requests.join(format!("{}.finalizing", index_name(request_id)))
}

pub(super) fn begin_finalization(
    root: &Path,
    agent: &str,
    runtime_id: &str,
    request_id: &str,
) -> Result<()> {
    let runtime = runtime_directory(root, runtime_id, true)?;
    let requests = directory(&runtime, "requests", true, runtime_id)?;
    let path = finalization_path(&requests, request_id);
    let marker = Finalization {
        agent_name: agent.into(),
        runtime_id: runtime_id.into(),
        request_id: request_id.into(),
    };
    let bytes = serde_json::to_vec(&marker).map_err(crate::FabricError::SerializeJson)?;
    if bytes.len() as u64 > MAX_RESULT_BYTES {
        return Err(failure(runtime_id, "result_storage_failed"));
    }
    match options().write(true).create_new(true).open(&path) {
        Ok(mut file) => {
            if file
                .write_all(&bytes)
                .and_then(|_| file.sync_all())
                .is_err()
            {
                let _ = fs::remove_file(path);
                return Err(failure(runtime_id, "result_storage_failed"));
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let existing: Finalization = read(&path, runtime_id)?
                .ok_or_else(|| failure(runtime_id, "result_storage_failed"))?;
            if existing.agent_name != agent
                || existing.runtime_id != runtime_id
                || existing.request_id != request_id
            {
                return Err(failure(runtime_id, "invalid_result_target"));
            }
        }
        Err(_) => return Err(failure(runtime_id, "result_storage_failed")),
    }
    Ok(())
}

fn completed(runtime: &Path, record: &Reservation) -> Result<Option<RunResult>> {
    if let Some(marker) = read::<Finalization>(
        &finalization_path(&runtime.join("requests"), &record.request_id),
        &record.runtime_id,
    )? {
        if marker.agent_name != record.agent_name
            || marker.runtime_id != record.runtime_id
            || marker.request_id != record.request_id
        {
            return Err(failure(&record.runtime_id, "invalid_result_target"));
        }
        // Invocation output alone does not establish one-shot cleanup finality.
        return Ok(None);
    }
    completed_result(runtime, record)
}

fn completed_result(runtime: &Path, record: &Reservation) -> Result<Option<RunResult>> {
    if !leaf(&record.invocation_id) {
        return Err(failure(&record.runtime_id, "invalid_result_target"));
    }
    let invocation = runtime.join(&record.invocation_id);
    if !invocation.exists() {
        return Ok(None);
    }
    let invocation = directory(runtime, &record.invocation_id, false, &record.runtime_id)?;
    let result: Option<RunResult> = read(&invocation.join("run-result.json"), &record.runtime_id)?;
    if let Some(result) = &result
        && (result.agent_name != record.agent_name
            || result.runtime_id != record.runtime_id
            || result.request_id != record.request_id
            || result.invocation_id != record.invocation_id)
    {
        return Err(failure(&record.runtime_id, "invalid_result_target"));
    }
    Ok(result)
}

pub(super) fn reserve(
    root: &Path,
    runtime: &RuntimeHandle,
    request: &RunRequest,
    invocation: &InvocationHandle,
    streaming: bool,
) -> Result<RequestReservation> {
    let runtime_dir = runtime_directory(root, &runtime.runtime_id, true)?;
    let requests = directory(&runtime_dir, "requests", true, &runtime.runtime_id)?;
    let path = requests.join(index_name(&request.request_id));
    let record = Reservation {
        agent_name: runtime.agent_name.clone(),
        runtime_id: runtime.runtime_id.clone(),
        request_id: request.request_id.clone(),
        invocation_id: invocation.invocation_id.clone(),
        fingerprint: format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(request).map_err(crate::FabricError::SerializeJson)?)
        ),
        streaming: Some(streaming),
    };
    let bytes = serde_json::to_vec(&record).map_err(crate::FabricError::SerializeJson)?;
    if bytes.len() as u64 > MAX_RESULT_BYTES {
        return Err(failure(&runtime.runtime_id, "result_storage_failed"));
    }
    match options().write(true).create_new(true).open(&path) {
        Ok(mut file) => {
            let pending = PendingReservation {
                path,
                dispatched: false,
            };
            file.write_all(&bytes)
                .and_then(|_| file.sync_all())
                .map_err(|_| failure(&runtime.runtime_id, "result_storage_failed"))?;
            Ok(RequestReservation::Pending(pending))
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let previous: Reservation = read(&path, &runtime.runtime_id)?
                .ok_or_else(|| failure(&runtime.runtime_id, "result_storage_failed"))?;
            if previous.agent_name != record.agent_name
                || previous.runtime_id != record.runtime_id
                || previous.request_id != record.request_id
                || previous.fingerprint != record.fingerprint
                || previous.streaming != record.streaming
            {
                return Err(failure(&runtime.runtime_id, "request_conflict"));
            }
            let result = completed(&runtime_dir, &previous)?
                .ok_or_else(|| failure(&runtime.runtime_id, "invocation_pending"))?;
            if streaming {
                return Err(failure(&runtime.runtime_id, "stream_replay_unsupported"));
            }
            Ok(RequestReservation::Cached(Box::new(result)))
        }
        Err(_) => Err(failure(&runtime.runtime_id, "result_storage_failed")),
    }
}

pub(super) fn save(root: &Path, result: &RunResult) -> Result<()> {
    let bytes = serde_json::to_vec(result).map_err(crate::FabricError::SerializeJson)?;
    if bytes.len() as u64 > MAX_RESULT_BYTES {
        return Err(failure(&result.runtime_id, "result_storage_failed"));
    }
    let invocation = invocation_directory(root, &result.runtime_id, &result.invocation_id)?;
    let temporary = invocation.join(format!("{}.tmp", super::new_id("result")));
    let mut file: File = options()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|_| failure(&result.runtime_id, "result_storage_failed"))?;
    file.write_all(&bytes)
        .and_then(|_| file.sync_all())
        .and_then(|_| fs::rename(&temporary, invocation.join("run-result.json")))
        .map_err(|_| failure(&result.runtime_id, "result_storage_failed"))?;
    Ok(())
}

pub(super) fn finalize(root: &Path, result: &RunResult) -> Result<()> {
    let runtime = runtime_directory(root, &result.runtime_id, false)?;
    let requests = directory(&runtime, "requests", false, &result.runtime_id)?;
    let record: Reservation = read(
        &requests.join(index_name(&result.request_id)),
        &result.runtime_id,
    )?
    .ok_or_else(|| failure(&result.runtime_id, "invalid_result_target"))?;
    if record.agent_name != result.agent_name
        || record.runtime_id != result.runtime_id
        || record.request_id != result.request_id
        || record.invocation_id != result.invocation_id
    {
        return Err(failure(&result.runtime_id, "invalid_result_target"));
    }
    completed_result(&runtime, &record)?
        .ok_or_else(|| failure(&result.runtime_id, "invalid_result_target"))?;
    begin_finalization(
        root,
        &result.agent_name,
        &result.runtime_id,
        &result.request_id,
    )?;
    save(root, result)?;
    // Publish cleanup finality only after the replacement result is saved.
    fs::remove_file(finalization_path(&requests, &result.request_id))
        .map_err(|_| failure(&result.runtime_id, "result_storage_failed"))
}

/// Read a saved result. Absence does not establish whether the request ran.
#[doc(hidden)]
pub fn invocation_result(
    root: &Path,
    agent: &str,
    runtime_id: &str,
    request_id: &str,
) -> Result<Option<RunResult>> {
    if [request_id, agent]
        .iter()
        .any(|value| value.trim().is_empty() || value.contains('\0'))
    {
        return Err(failure(runtime_id, "invalid_result_target"));
    }
    if !leaf(runtime_id) {
        return Err(failure(runtime_id, "invalid_result_target"));
    }
    let root = match fs::canonicalize(root) {
        Ok(root) => root,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err(failure(runtime_id, "result_storage_failed")),
    };
    trusted_ancestry(&root, runtime_id)?;
    let fabric = root.join(".fabric");
    let runtime = fabric.join(runtime_id);
    let requests = runtime.join("requests");
    for path in [&fabric, &runtime, &requests] {
        match fs::symlink_metadata(path) {
            Ok(metadata)
                if metadata.is_dir()
                    && !metadata.file_type().is_symlink()
                    && owned_metadata(&metadata, false) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            _ => return Err(failure(runtime_id, "result_storage_failed")),
        }
    }
    let record: Option<Reservation> = read(&requests.join(index_name(request_id)), runtime_id)?;
    let Some(record) = record else {
        return Ok(None);
    };
    if record.agent_name != agent
        || record.runtime_id != runtime_id
        || record.request_id != request_id
    {
        return Err(failure(runtime_id, "invalid_result_target"));
    }
    completed(&runtime, &record)
}

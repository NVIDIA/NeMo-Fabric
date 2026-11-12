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
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(failure(runtime_id, "result_storage_failed"));
    }
    Ok(path)
}

fn runtime_directory(root: &Path, runtime_id: &str, create: bool) -> Result<PathBuf> {
    if !leaf(runtime_id) {
        return Err(failure(runtime_id, "invalid_result_target"));
    }
    if create {
        fs::create_dir_all(root).map_err(|_| failure(runtime_id, "result_storage_failed"))?;
    }
    let root = fs::canonicalize(root).map_err(|_| failure(runtime_id, "result_storage_failed"))?;
    let fabric = directory(&root, ".fabric", create, runtime_id)?;
    directory(&fabric, runtime_id, create, runtime_id)
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
    if !metadata.is_file() || metadata.len() > MAX_RESULT_BYTES {
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

fn completed(runtime: &Path, record: &Reservation) -> Result<Option<RunResult>> {
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
) -> Result<Option<RunResult>> {
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
    };
    match options().write(true).create_new(true).open(&path) {
        Ok(mut file) => {
            let bytes = serde_json::to_vec(&record).map_err(crate::FabricError::SerializeJson)?;
            file.write_all(&bytes)
                .and_then(|_| file.sync_all())
                .map_err(|_| failure(&runtime.runtime_id, "result_storage_failed"))?;
            Ok(None)
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let previous: Reservation = read(&path, &runtime.runtime_id)?
                .ok_or_else(|| failure(&runtime.runtime_id, "result_storage_failed"))?;
            if previous.agent_name != record.agent_name
                || previous.runtime_id != record.runtime_id
                || previous.request_id != record.request_id
                || previous.fingerprint != record.fingerprint
            {
                return Err(failure(&runtime.runtime_id, "request_conflict"));
            }
            completed(&runtime_dir, &previous)?
                .map(Some)
                .ok_or_else(|| failure(&runtime.runtime_id, "invocation_pending"))
        }
        Err(_) => Err(failure(&runtime.runtime_id, "result_storage_failed")),
    }
}

pub(super) fn save(root: &Path, result: &RunResult) -> Result<()> {
    let runtime = runtime_directory(root, &result.runtime_id, true)?;
    let invocation = directory(&runtime, &result.invocation_id, true, &result.runtime_id)?;
    let temporary = invocation.join(format!("{}.tmp", super::new_id("result")));
    let mut file: File = options()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|_| failure(&result.runtime_id, "result_storage_failed"))?;
    let bytes = serde_json::to_vec(result).map_err(crate::FabricError::SerializeJson)?;
    file.write_all(&bytes)
        .and_then(|_| file.sync_all())
        .and_then(|_| fs::rename(&temporary, invocation.join("run-result.json")))
        .map_err(|_| failure(&result.runtime_id, "result_storage_failed"))?;
    Ok(())
}

/// Read a saved result. Absence does not establish whether the request ran.
#[doc(hidden)]
pub fn invocation_result(
    root: &Path,
    agent: &str,
    runtime_id: &str,
    request_id: &str,
) -> Result<Option<RunResult>> {
    if request_id.trim().is_empty() || agent.trim().is_empty() {
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
    let fabric = root.join(".fabric");
    let runtime = fabric.join(runtime_id);
    let requests = runtime.join("requests");
    for path in [&fabric, &runtime, &requests] {
        match fs::symlink_metadata(path) {
            Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {}
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

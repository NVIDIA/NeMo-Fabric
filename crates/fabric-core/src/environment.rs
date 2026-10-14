// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Internal execution-environment provider boundary.

use std::collections::BTreeMap;
use std::ffi::OsString;
#[cfg(unix)]
use std::io::Write;
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
#[cfg(unix)]
use std::process::ChildStdin;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{LazyLock, Mutex};
use std::thread;
use std::time::Duration;
#[cfg(unix)]
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::{ControlLocation, EnvironmentOwnership, EnvironmentPlan, RunPlan};
use crate::error::{FabricError, Result};
use crate::runtime::{
    EnvironmentHandle, EnvironmentReference, absolute_path, new_id, resolve_path,
};

const OPEN_SHELL_PROVIDER_ID: &str = "openshell";
const OPEN_SHELL_PROVIDER_COMMAND: &str = "fabric-environment-openshell";
const OPEN_SHELL_PROVIDER_COMMAND_ENV: &str = "NEMO_FABRIC_OPEN_SHELL_PROVIDER";
const PROVIDER_PROTOCOL_VERSION: &str = "fabric.environment-provider.v1alpha1";
const MAX_PROVIDER_REQUEST_BYTES: usize = 256 * 1024;
const MAX_PROVIDER_RESPONSE_BYTES: usize = 4 * 1024 * 1024;
const DEFAULT_PROVIDER_OPERATION_TIMEOUT_SECONDS: u64 = 30;
const PROVIDER_RESPONSE_GRACE_SECONDS: u64 = 15;
const PROVIDER_GATEWAY_CALL_TIMEOUT_SECONDS: u64 = 30;

/// Internal environment preparation contract.
///
/// This boundary is intentionally crate-private while the first non-local
/// provider proves the lifecycle. It does not yet define a public provider ABI.
pub(crate) trait EnvironmentProvider: Sync {
    /// Resolve or prepare the environment represented by a run plan.
    fn prepare(&self, plan: &RunPlan) -> Result<EnvironmentHandle>;

    /// Verify and attach to a caller-owned provider resource.
    fn attach(&self, plan: &RunPlan, reference: &EnvironmentReference)
    -> Result<EnvironmentHandle>;
}

struct LocalEnvironmentProvider;
struct OpenShellEnvironmentProvider;

static LOCAL_ENVIRONMENT_PROVIDER: LocalEnvironmentProvider = LocalEnvironmentProvider;
static OPEN_SHELL_ENVIRONMENT_PROVIDER: OpenShellEnvironmentProvider = OpenShellEnvironmentProvider;
static OPEN_SHELL_PROVIDER_PROCESS: LazyLock<Mutex<Option<ProviderProcess>>> =
    LazyLock::new(|| Mutex::new(None));

/// Resolve a built-in environment provider by its stable configuration id.
pub(crate) fn resolve_environment_provider(
    provider_id: &str,
) -> Option<&'static dyn EnvironmentProvider> {
    match provider_id {
        "local" => Some(&LOCAL_ENVIRONMENT_PROVIDER),
        OPEN_SHELL_PROVIDER_ID if cfg!(unix) => Some(&OPEN_SHELL_ENVIRONMENT_PROVIDER),
        _ => None,
    }
}

impl EnvironmentProvider for LocalEnvironmentProvider {
    fn prepare(&self, plan: &RunPlan) -> Result<EnvironmentHandle> {
        let mut metadata = BTreeMap::new();
        let mut connection = BTreeMap::new();
        let (
            control_location,
            ownership,
            workspace,
            artifacts,
            environment_env,
            connection_settings,
            environment_metadata,
            settings,
        ) = if let Some(environment) = &plan.environment_plan {
            (
                environment.control_location,
                environment.ownership,
                environment.workspace.clone(),
                environment.artifacts.clone(),
                environment.env.clone(),
                environment.connection.clone(),
                environment.metadata.clone(),
                environment.settings.clone(),
            )
        } else {
            (
                ControlLocation::ExternalControl,
                EnvironmentOwnership::CallerOwned,
                Some(plan.base_dir.clone()),
                plan.config
                    .runtime
                    .artifacts
                    .as_ref()
                    .map(|artifacts| resolve_path(&plan.base_dir, artifacts)),
                BTreeMap::new(),
                serde_json::Map::new(),
                serde_json::Map::new(),
                serde_json::Map::new(),
            )
        };
        let workspace = match workspace {
            Some(path) => Some(absolute_path(path)?),
            None => None,
        };
        connection.extend(connection_settings);
        metadata.extend(settings);
        metadata.extend(environment_metadata);
        Ok(EnvironmentHandle {
            environment_id: new_id("environment"),
            provider: "local".to_string(),
            control_location,
            workspace,
            artifacts,
            env: environment_env,
            ownership,
            connection,
            metadata,
        })
    }

    fn attach(
        &self,
        _plan: &RunPlan,
        _reference: &EnvironmentReference,
    ) -> Result<EnvironmentHandle> {
        Err(FabricError::InvalidConfig {
            field: "environment_reference.provider".to_string(),
            reason: "the local provider does not attach external resources".to_string(),
        })
    }
}

impl EnvironmentProvider for OpenShellEnvironmentProvider {
    fn prepare(&self, plan: &RunPlan) -> Result<EnvironmentHandle> {
        let environment =
            plan.environment_plan
                .as_ref()
                .ok_or_else(|| FabricError::InvalidConfig {
                    field: "environment".to_string(),
                    reason: "the openshell provider requires an explicit environment configuration"
                        .to_string(),
                })?;
        validate_open_shell_profile(environment, EnvironmentOwnership::FabricOwned)?;

        let environment_id = new_id("environment");
        let prepared: PreparedEnvironment = self.request(ProviderOperation::Prepare {
            environment_id: &environment_id,
            environment,
        })?;
        let workspace =
            prepared
                .workspace
                .ok_or_else(|| FabricError::EnvironmentProviderOperation {
                    provider: OPEN_SHELL_PROVIDER_ID.to_string(),
                    operation: "prepare".to_string(),
                    code: "workspace_unavailable".to_string(),
                    message: "OpenShell provider response omitted the sandbox workspace"
                        .to_string(),
                })?;
        let mut metadata = prepared.metadata.into_iter().collect::<BTreeMap<_, _>>();
        metadata.extend(environment.metadata.clone());

        Ok(EnvironmentHandle {
            environment_id,
            provider: OPEN_SHELL_PROVIDER_ID.to_string(),
            control_location: environment.control_location,
            workspace: Some(workspace),
            artifacts: prepared.artifacts,
            env: environment.env.clone(),
            ownership: environment.ownership,
            connection: prepared.connection.into_iter().collect(),
            metadata,
        })
    }

    fn attach(
        &self,
        plan: &RunPlan,
        reference: &EnvironmentReference,
    ) -> Result<EnvironmentHandle> {
        let environment =
            plan.environment_plan
                .as_ref()
                .ok_or_else(|| FabricError::InvalidConfig {
                    field: "environment".to_string(),
                    reason: "the openshell provider requires an explicit environment configuration"
                        .to_string(),
                })?;
        validate_open_shell_profile(environment, EnvironmentOwnership::CallerOwned)?;

        let environment_id = new_id("environment");
        let prepared: PreparedEnvironment = self.request(ProviderOperation::Attach {
            environment_id: &environment_id,
            environment,
            reference,
        })?;
        let workspace =
            prepared
                .workspace
                .ok_or_else(|| FabricError::EnvironmentProviderOperation {
                    provider: OPEN_SHELL_PROVIDER_ID.to_string(),
                    operation: "attach".to_string(),
                    code: "workspace_unavailable".to_string(),
                    message: "OpenShell provider response omitted the sandbox workspace"
                        .to_string(),
                })?;
        let mut metadata = prepared.metadata.into_iter().collect::<BTreeMap<_, _>>();
        metadata.extend(environment.metadata.clone());

        Ok(EnvironmentHandle {
            environment_id,
            provider: OPEN_SHELL_PROVIDER_ID.to_string(),
            control_location: environment.control_location,
            workspace: Some(workspace),
            artifacts: prepared.artifacts,
            env: environment.env.clone(),
            ownership: environment.ownership,
            connection: prepared.connection.into_iter().collect(),
            metadata,
        })
    }
}

impl OpenShellEnvironmentProvider {
    fn request<T>(&self, operation: ProviderOperation<'_>) -> Result<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        request_provider_process(provider_command(), operation)
    }
}

/// Release or detach an environment through its registered provider.
pub(crate) fn release_environment(
    environment: &EnvironmentHandle,
) -> Result<ProviderReleaseOutput> {
    match environment.provider.as_str() {
        "local" => Ok(ProviderReleaseOutput {
            released: false,
            detached: true,
        }),
        OPEN_SHELL_PROVIDER_ID if cfg!(unix) => {
            OPEN_SHELL_ENVIRONMENT_PROVIDER.request(ProviderOperation::Release { environment })
        }
        provider => Err(FabricError::UnsupportedEnvironmentRelease {
            provider: provider.to_string(),
        }),
    }
}

fn validate_open_shell_profile(
    environment: &EnvironmentPlan,
    expected_ownership: EnvironmentOwnership,
) -> Result<()> {
    if environment.control_location != ControlLocation::InEnvControl {
        return Err(FabricError::InvalidConfig {
            field: "environment.control_location".to_string(),
            reason: "the experimental openshell provider supports only `in_env_control`"
                .to_string(),
        });
    }
    if environment.ownership != expected_ownership {
        return Err(FabricError::InvalidConfig {
            field: "environment.ownership".to_string(),
            reason: format!(
                "this operation requires `{}`",
                match expected_ownership {
                    EnvironmentOwnership::CallerOwned => "caller_owned",
                    EnvironmentOwnership::FabricOwned => "fabric_owned",
                }
            ),
        });
    }
    if environment.workspace.is_none() {
        return Err(FabricError::InvalidConfig {
            field: "environment.workspace".to_string(),
            reason: "the experimental openshell provider requires an absolute sandbox workspace"
                .to_string(),
        });
    }
    Ok(())
}

fn provider_command() -> OsString {
    std::env::var_os(OPEN_SHELL_PROVIDER_COMMAND_ENV)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| OsString::from(OPEN_SHELL_PROVIDER_COMMAND))
}

fn request_provider_process<T>(command: OsString, operation: ProviderOperation<'_>) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    let operation_name = operation.name();
    let response_timeout = operation.response_timeout();
    let request_id = new_id("environment-provider-request");
    let request = ProviderRequest {
        protocol_version: PROVIDER_PROTOCOL_VERSION,
        request_id: &request_id,
        operation,
    };
    let mut encoded = serde_json::to_vec(&request).map_err(FabricError::SerializeJson)?;
    if encoded.len() > MAX_PROVIDER_REQUEST_BYTES {
        return provider_error(
            operation_name,
            "request_too_large",
            format!(
                "request is {} bytes; the limit is {MAX_PROVIDER_REQUEST_BYTES}",
                encoded.len()
            ),
        );
    }
    encoded.push(b'\n');

    let mut slot = OPEN_SHELL_PROVIDER_PROCESS
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let replace = match slot.as_mut() {
        Some(process) if process.command != command => true,
        Some(process) => process.has_exited(operation_name)?,
        None => false,
    };
    if replace && let Some(mut process) = slot.take() {
        process.shutdown();
    }
    if slot.is_none() {
        *slot = Some(ProviderProcess::spawn(command, operation_name)?);
    }
    let process = slot.as_mut().expect("provider process initialized");
    if process
        .write_request(&encoded, Duration::from_secs(10))
        .is_err()
    {
        process.shutdown();
        *slot = None;
        return provider_error(
            operation_name,
            "provider_protocol_error",
            "could not write the bounded provider request".to_string(),
        );
    }
    let output = match process.responses.recv_timeout(response_timeout) {
        Ok(Ok(output)) => output,
        Ok(Err(_)) => {
            process.shutdown();
            *slot = None;
            return provider_error(
                operation_name,
                "provider_protocol_error",
                "could not read provider response".to_string(),
            );
        }
        Err(RecvTimeoutError::Timeout) => {
            process.shutdown();
            *slot = None;
            return provider_error(
                operation_name,
                "provider_protocol_error",
                "provider response deadline expired".to_string(),
            );
        }
        Err(RecvTimeoutError::Disconnected) => {
            process.shutdown();
            *slot = None;
            return provider_error(
                operation_name,
                "provider_exited",
                "provider closed its response stream".to_string(),
            );
        }
    };
    if output.len() > MAX_PROVIDER_RESPONSE_BYTES {
        process.shutdown();
        *slot = None;
        return provider_error(
            operation_name,
            "response_too_large",
            format!(
                "response is {} bytes; the limit is {MAX_PROVIDER_RESPONSE_BYTES}",
                output.len()
            ),
        );
    }
    if output.is_empty() || output.last() != Some(&b'\n') {
        process.shutdown();
        *slot = None;
        return provider_error(
            operation_name,
            "provider_exited",
            "provider response was incomplete".to_string(),
        );
    }

    let response: ProviderResponse = match serde_json::from_slice(&output) {
        Ok(response) => response,
        Err(_) => {
            process.shutdown();
            *slot = None;
            return provider_error(
                operation_name,
                "provider_protocol_error",
                "provider returned invalid JSON".to_string(),
            );
        }
    };
    if response.protocol_version != PROVIDER_PROTOCOL_VERSION {
        process.shutdown();
        *slot = None;
        return provider_error(
            operation_name,
            "provider_protocol_mismatch",
            "provider protocol version did not match".to_string(),
        );
    }
    if response.request_id != request_id {
        process.shutdown();
        *slot = None;
        return provider_error(
            operation_name,
            "provider_correlation_mismatch",
            "provider response request id did not match the request".to_string(),
        );
    }
    match response.outcome {
        ProviderOutcome::Succeeded { output } => {
            serde_json::from_value(output).map_err(|_| FabricError::EnvironmentProviderOperation {
                provider: OPEN_SHELL_PROVIDER_ID.to_string(),
                operation: operation_name.to_string(),
                code: "provider_protocol_error".to_string(),
                message: "provider output did not match the operation contract".to_string(),
            })
        }
        ProviderOutcome::Failed { error } => {
            provider_error(operation_name, &error.code, error.message)
        }
    }
}

struct ProviderProcess {
    command: OsString,
    child: Child,
    #[cfg(unix)]
    stdin: ChildStdin,
    responses: Receiver<std::io::Result<Vec<u8>>>,
    responses_thread: Option<thread::JoinHandle<()>>,
    stderr_thread: Option<thread::JoinHandle<()>>,
}

impl ProviderProcess {
    fn spawn(command: OsString, operation: &str) -> Result<Self> {
        let mut child = Command::new(&command)
            .args(["serve", "--stdio"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| FabricError::EnvironmentProviderOperation {
                provider: OPEN_SHELL_PROVIDER_ID.to_string(),
                operation: operation.to_string(),
                code: "provider_unavailable".to_string(),
                message: format!("could not start `{}`: {error}", command.to_string_lossy()),
            })?;
        #[cfg(unix)]
        let stdin =
            child
                .stdin
                .take()
                .ok_or_else(|| FabricError::EnvironmentProviderOperation {
                    provider: OPEN_SHELL_PROVIDER_ID.to_string(),
                    operation: operation.to_string(),
                    code: "provider_protocol_error".to_string(),
                    message: "provider stdin was unavailable".to_string(),
                })?;
        let stdout =
            child
                .stdout
                .take()
                .ok_or_else(|| FabricError::EnvironmentProviderOperation {
                    provider: OPEN_SHELL_PROVIDER_ID.to_string(),
                    operation: operation.to_string(),
                    code: "provider_protocol_error".to_string(),
                    message: "provider stdout was unavailable".to_string(),
                })?;
        let (response_sender, responses) = mpsc::sync_channel(1);
        let responses_thread = thread::spawn(move || {
            let mut stdout = BufReader::new(stdout);
            loop {
                let mut output = Vec::new();
                let read = {
                    let mut bounded = stdout
                        .by_ref()
                        .take((MAX_PROVIDER_RESPONSE_BYTES + 1) as u64);
                    bounded.read_until(b'\n', &mut output)
                };
                match read {
                    Ok(0) => break,
                    Ok(_) => {
                        if response_sender.try_send(Ok(output)).is_err() {
                            break;
                        }
                    }
                    Err(error) => {
                        let _ = response_sender.try_send(Err(error));
                        break;
                    }
                }
            }
        });
        let mut stderr =
            child
                .stderr
                .take()
                .ok_or_else(|| FabricError::EnvironmentProviderOperation {
                    provider: OPEN_SHELL_PROVIDER_ID.to_string(),
                    operation: operation.to_string(),
                    code: "provider_protocol_error".to_string(),
                    message: "provider stderr was unavailable".to_string(),
                })?;
        // Drain stderr to preserve transport liveness; routine errors never expose it.
        let stderr_thread = thread::spawn(move || {
            let mut chunk = [0_u8; 4096];
            while let Ok(read) = stderr.read(&mut chunk) {
                if read == 0 {
                    break;
                }
            }
        });
        Ok(Self {
            command,
            child,
            #[cfg(unix)]
            stdin,
            responses,
            responses_thread: Some(responses_thread),
            stderr_thread: Some(stderr_thread),
        })
    }

    #[cfg(unix)]
    fn write_request(&mut self, encoded: &[u8], timeout: Duration) -> std::io::Result<()> {
        use std::os::fd::AsRawFd;
        let descriptor = self.stdin.as_raw_fd();
        // The owned pipe remains nonblocking for subsequent bounded writes.
        let flags = unsafe { libc::fcntl(descriptor, libc::F_GETFL) };
        if flags < 0
            || unsafe { libc::fcntl(descriptor, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0
        {
            return Err(std::io::Error::last_os_error());
        }
        let deadline = Instant::now() + timeout;
        let mut remaining = encoded;
        while !remaining.is_empty() {
            if Instant::now() >= deadline {
                return Err(std::io::ErrorKind::TimedOut.into());
            }
            match self.stdin.write(remaining) {
                Ok(0) => return Err(std::io::ErrorKind::WriteZero.into()),
                Ok(written) => remaining = &remaining[written..],
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(5))
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    #[cfg(not(unix))]
    fn write_request(&mut self, _encoded: &[u8], _timeout: Duration) -> std::io::Result<()> {
        Err(std::io::ErrorKind::Unsupported.into())
    }

    fn has_exited(&mut self, operation: &str) -> Result<bool> {
        self.child
            .try_wait()
            .map(|status| status.is_some())
            .map_err(|error| FabricError::EnvironmentProviderOperation {
                provider: OPEN_SHELL_PROVIDER_ID.to_string(),
                operation: operation.to_string(),
                code: "provider_wait_failed".to_string(),
                message: error.to_string(),
            })
    }

    fn shutdown(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(thread) = self.responses_thread.take() {
            let _ = thread.join();
        }
        if let Some(thread) = self.stderr_thread.take() {
            let _ = thread.join();
        }
    }
}

fn provider_error<T>(operation: &str, code: &str, message: String) -> Result<T> {
    Err(FabricError::EnvironmentProviderOperation {
        provider: OPEN_SHELL_PROVIDER_ID.to_string(),
        operation: operation.to_string(),
        code: code.to_string(),
        message,
    })
}

#[derive(Serialize)]
struct ProviderRequest<'a> {
    protocol_version: &'static str,
    request_id: &'a str,
    #[serde(flatten)]
    operation: ProviderOperation<'a>,
}

#[derive(Serialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
enum ProviderOperation<'a> {
    Prepare {
        environment_id: &'a str,
        environment: &'a EnvironmentPlan,
    },
    Attach {
        environment_id: &'a str,
        environment: &'a EnvironmentPlan,
        reference: &'a EnvironmentReference,
    },
    Release {
        environment: &'a EnvironmentHandle,
    },
}

impl ProviderOperation<'_> {
    fn name(&self) -> &'static str {
        match self {
            Self::Prepare { .. } => "prepare",
            Self::Attach { .. } => "attach",
            Self::Release { .. } => "release",
        }
    }

    fn response_timeout(&self) -> Duration {
        let operation_seconds = match self {
            Self::Prepare { environment, .. } => {
                setting_seconds(&environment.settings, "ready_timeout_seconds", 60)
                    .saturating_add(setting_seconds(
                        &environment.settings,
                        "delete_timeout_seconds",
                        30,
                    ))
                    .saturating_add(6 * PROVIDER_GATEWAY_CALL_TIMEOUT_SECONDS)
            }
            Self::Attach { .. } => 3 * PROVIDER_GATEWAY_CALL_TIMEOUT_SECONDS,
            Self::Release { environment } => connection_seconds(
                &environment.connection,
                "delete_timeout_seconds",
                DEFAULT_PROVIDER_OPERATION_TIMEOUT_SECONDS,
            )
            .saturating_add(3 * PROVIDER_GATEWAY_CALL_TIMEOUT_SECONDS),
        };
        Duration::from_secs(operation_seconds.saturating_add(PROVIDER_RESPONSE_GRACE_SECONDS))
    }
}

fn setting_seconds(settings: &serde_json::Map<String, Value>, key: &str, default: u64) -> u64 {
    settings
        .get(key)
        .and_then(Value::as_u64)
        .unwrap_or(default)
        .min(600)
}

fn connection_seconds(connection: &BTreeMap<String, Value>, key: &str, default: u64) -> u64 {
    connection
        .get(key)
        .and_then(Value::as_u64)
        .unwrap_or(default)
        .min(600)
}

#[derive(Deserialize)]
struct ProviderResponse {
    protocol_version: String,
    request_id: String,
    #[serde(flatten)]
    outcome: ProviderOutcome,
}

#[derive(Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum ProviderOutcome {
    Succeeded { output: Value },
    Failed { error: ProviderFailure },
}

#[derive(Deserialize)]
struct ProviderFailure {
    code: String,
    message: String,
}

#[derive(Deserialize)]
struct PreparedEnvironment {
    workspace: Option<PathBuf>,
    artifacts: Option<PathBuf>,
    #[serde(default)]
    connection: serde_json::Map<String, Value>,
    #[serde(default)]
    metadata: serde_json::Map<String, Value>,
}

#[derive(Debug, Deserialize, PartialEq)]
pub(crate) struct ProviderReleaseOutput {
    pub(crate) released: bool,
    pub(crate) detached: bool,
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn prepare_response_timeout_covers_cleanup_after_readiness() {
        let environment: EnvironmentPlan = serde_json::from_value(serde_json::json!({
            "provider": OPEN_SHELL_PROVIDER_ID,
            "control_location": ControlLocation::InEnvControl,
            "ownership": EnvironmentOwnership::FabricOwned,
            "workspace": null,
            "artifacts": null,
            "env": {},
            "env_names": [],
            "connection": {},
            "metadata": {},
            "settings": {
                "ready_timeout_seconds": 60,
                "delete_timeout_seconds": 30
            }
        }))
        .expect("environment fixture");
        let operation = ProviderOperation::Prepare {
            environment_id: "environment-test",
            environment: &environment,
        };

        assert_eq!(operation.response_timeout(), Duration::from_secs(285));
    }

    fn provider_script(body: &str) -> (PathBuf, OsString) {
        let directory = std::env::temp_dir().join(new_id("fabric-provider-test"));
        fs::create_dir_all(&directory).expect("create test directory");
        let path = directory.join("provider");
        fs::write(&path, format!("#!/usr/bin/env python3\n{body}\n")).expect("write test provider");
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).expect("executable provider");
        (directory, path.into_os_string())
    }

    #[test]
    fn request_write_is_bounded_when_the_provider_does_not_read() {
        let (directory, command) = provider_script("import time\ntime.sleep(30)");
        let mut process = ProviderProcess::spawn(command, "prepare").expect("start provider");
        let started = Instant::now();
        let error = process
            .write_request(&vec![b'x'; 1024 * 1024], Duration::from_millis(30))
            .expect_err("blocked pipe write must expire");
        assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
        assert!(started.elapsed() < Duration::from_secs(1));
        process.shutdown();
        fs::remove_dir_all(directory).expect("remove fixture");
    }

    #[test]
    fn malformed_provider_frames_never_expose_stderr_or_response_data() {
        let (directory, command) = provider_script(
            r#"
import json, sys
request = json.loads(sys.stdin.readline())
print('PRIVATE stderr', file=sys.stderr, flush=True)
print(json.dumps({'protocol_version': 'PRIVATE response', 'request_id': request['request_id'], 'status': 'succeeded', 'output': {}}), flush=True)
"#,
        );
        let environment = EnvironmentHandle {
            environment_id: "environment-test".to_string(),
            provider: "openshell".to_string(),
            control_location: ControlLocation::InEnvControl,
            ownership: EnvironmentOwnership::CallerOwned,
            workspace: Some(PathBuf::from("/sandbox/workspace")),
            artifacts: None,
            env: BTreeMap::new(),
            connection: BTreeMap::new(),
            metadata: BTreeMap::new(),
        };
        let error = request_provider_process::<ProviderReleaseOutput>(
            command,
            ProviderOperation::Release {
                environment: &environment,
            },
        )
        .expect_err("reject an incompatible provider");
        assert!(
            matches!(&error, FabricError::EnvironmentProviderOperation {code, ..} if code == "provider_protocol_mismatch")
        );
        assert!(!error.to_string().contains("PRIVATE"));
        fs::remove_dir_all(directory).expect("remove fixture");
    }
}

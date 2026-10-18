// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host-side runtime coordinator for agents running in OpenShell sandboxes.
//!
//! Implements Fabric's runtime start, invoke, and stop operations by projecting
//! the host configuration into a task-local configuration and sending commands
//! through the OpenShell environment provider. It validates task SDK responses,
//! tracks runtime bindings and uncertain operations, and collects declared
//! artifacts. The environment provider handles gateway and sandbox operations;
//! the task runtime server and harness adapter execute inside the sandbox.

use super::*;
use crate::config::{DiscoveryConfig, EnvironmentConfig, ResolutionStrategy};
use crate::environment::{collect_artifacts, command_runtime};
use crate::task_command::{TaskCommand, TaskCommandResponse};

pub(super) struct OpenShellRuntimeAdapter;

#[derive(Clone, Copy, PartialEq)]
enum State {
    Starting,
    Active,
    Invoking,
    Stopping,
    Unknown,
    Stopped,
    Releasing,
}
#[derive(Clone)]
struct Binding {
    state: State,
    runtime_id: Option<String>,
}
const START_KEY: &str = "\0starting";
const UNKNOWN_START_KEY: &str = "\0unknown-start";
const RELEASE_KEY: &str = "\0releasing";
static BINDINGS: LazyLock<Mutex<BTreeMap<String, BTreeMap<String, Binding>>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

fn resource_key(environment: &EnvironmentHandle) -> Result<String> {
    let text = |name: &str| {
        environment
            .connection
            .get(name)
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
    };
    let gateway = text("gateway").ok_or_else(|| invalid_environment("connection.gateway"))?;
    let id = text("sandbox_id").ok_or_else(|| invalid_environment("connection.sandbox_id"))?;
    Ok(serde_json::json!([
        gateway.trim_end_matches('/'),
        text("workspace").unwrap_or("default"),
        id
    ])
    .to_string())
}
fn invalid_environment(field: &str) -> FabricError {
    FabricError::InvalidConfig {
        field: format!("environment.{field}"),
        reason: "a verified OpenShell binding is required".to_string(),
    }
}
fn invoke_seconds(plan: &RunPlan) -> Result<u64> {
    let seconds = plan.config.runtime.timeout_seconds.unwrap_or(240.0);
    if !seconds.is_finite() || seconds <= 0.0 || seconds > 300.0 {
        return Err(FabricError::InvalidConfig {field: "runtime.timeout_seconds".to_string(), reason: "OpenShell buffered invocation supports a timeout greater than zero and at most 300 seconds".to_string()});
    }
    Ok(seconds.ceil() as u64)
}
fn task_root(path: Option<&Path>, field: &str, default: Option<&str>) -> Result<PathBuf> {
    let value = path
        .and_then(Path::to_str)
        .or(default)
        .ok_or_else(|| invalid_environment(field))?;
    if !value.starts_with('/')
        || value.contains('\\')
        || value.contains('\0')
        || value.split('/').any(|part| part == "..")
    {
        return Err(invalid_environment(field));
    }
    Ok(PathBuf::from(value))
}
fn task_environment_config(
    workspace: PathBuf,
    artifacts: PathBuf,
    env: &BTreeMap<String, String>,
    env_names: Option<Value>,
) -> Result<EnvironmentConfig> {
    // Build through serde so additive SDK fields survive without coupling the
    // host provider to a fixed EnvironmentConfig field list.
    let mut value = serde_json::json!({
        "provider": "local",
        "control_location": ControlLocation::ExternalControl,
        "ownership": EnvironmentOwnership::CallerOwned,
        "workspace": workspace,
        "artifacts": artifacts,
        "env": env,
        "connection": {},
        "settings": {},
        "metadata": {},
    });
    if let Some(env_names) = env_names {
        value["env_names"] = env_names;
    }
    serde_json::from_value(value).map_err(|_| invalid_environment("task_config"))
}
fn task_config(plan: &RunPlan, environment: &EnvironmentHandle) -> Result<crate::FabricConfig> {
    let mut config = plan.config.clone();
    let workspace = task_root(environment.workspace.as_deref(), "workspace", None)?;
    let artifacts = task_root(
        environment.artifacts.as_deref(),
        "artifacts",
        Some("/sandbox/artifacts"),
    )?;
    let timeout = invoke_seconds(plan)?;
    let env_names = plan
        .config
        .environment
        .as_ref()
        .map(serde_json::to_value)
        .transpose()
        .map_err(|_| invalid_environment("config.environment"))?
        .and_then(|environment| environment.get("env_names").cloned());
    config.environment = Some(task_environment_config(
        workspace,
        artifacts.clone(),
        &environment.env,
        env_names,
    )?);
    config.runtime.artifacts = Some(artifacts.clone());
    config.runtime.timeout_seconds = Some(timeout as f64);
    if let Some(harness) = &mut config.harness {
        harness.resolution = Some(ResolutionStrategy::Preinstalled);
    }
    // Host discovery paths and gateway settings never become task paths.
    config.discovery = None;
    if let Some(paths) = plan
        .environment_plan
        .as_ref()
        .and_then(|plan| plan.settings.get("task_discovery_paths"))
    {
        let paths: Vec<String> = serde_json::from_value(paths.clone())
            .map_err(|_| invalid_environment("settings.task_discovery_paths"))?;
        let paths = paths
            .iter()
            .map(|path| task_root(Some(Path::new(path)), "settings.task_discovery_paths", None))
            .collect::<Result<Vec<_>>>()?;
        config.discovery = Some(DiscoveryConfig {
            local_paths: paths,
            extensions: BTreeMap::new(),
        });
    }
    if let Some(relay) = &mut config.relay {
        relay.output_dir = Some(artifacts.join("relay"));
    }
    Ok(config)
}
fn set_state(key: &str, runtime_id: &str, state: State) {
    BINDINGS
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .entry(key.to_string())
        .or_default()
        .insert(
            runtime_id.to_string(),
            Binding {
                state,
                runtime_id: Some(runtime_id.to_string()),
            },
        );
}
fn set_start_state(key: &str, state: State) {
    let mut bindings = BINDINGS.lock().unwrap_or_else(|error| error.into_inner());
    let environment_bindings = bindings.entry(key.to_string()).or_default();
    environment_bindings.remove(START_KEY);
    environment_bindings.insert(
        UNKNOWN_START_KEY.to_string(),
        Binding {
            state,
            runtime_id: None,
        },
    );
}
fn command_value(
    response: TaskCommandResponse,
    operation: AdapterLifecycleOperation,
    runtime_id: &str,
) -> Result<Value> {
    match response {
        TaskCommandResponse::Returned { value } => Ok(value),
        TaskCommandResponse::Rejected { error } => Err(lifecycle_error(
            operation,
            runtime_id,
            error.code,
            format!(
                "task command rejected during {} ({} effects)",
                error.stage, error.effects
            ),
            "",
        )),
    }
}
fn claim(runtime: &RuntimeHandle, stopping: bool) -> Result<Option<String>> {
    let key = resource_key(&runtime.environment)?;
    let mut bindings = BINDINGS.lock().unwrap_or_else(|error| error.into_inner());
    let environment_bindings = bindings.get_mut(&key).ok_or_else(|| {
        lifecycle_error(
            AdapterLifecycleOperation::Invoke,
            &runtime.runtime_id,
            "runtime_unavailable",
            "remote runtime binding is unavailable",
            "",
        )
    })?;
    if environment_bindings.contains_key(RELEASE_KEY) {
        return Err(lifecycle_error(
            AdapterLifecycleOperation::Invoke,
            &runtime.runtime_id,
            "runtime_unavailable",
            "remote environment is being released",
            "",
        ));
    }
    let binding = environment_bindings
        .get_mut(&runtime.runtime_id)
        .ok_or_else(|| {
            lifecycle_error(
                AdapterLifecycleOperation::Invoke,
                &runtime.runtime_id,
                "runtime_unavailable",
                "remote runtime binding is unavailable",
                "",
            )
        })?;
    if binding.runtime_id.as_deref() != Some(&runtime.runtime_id) {
        return Err(lifecycle_error(
            AdapterLifecycleOperation::Invoke,
            &runtime.runtime_id,
            "runtime_unavailable",
            "runtime identity does not match the resource binding",
            "",
        ));
    }
    if stopping && binding.state == State::Stopped {
        return Ok(None);
    }
    if binding.state != State::Active
        && !(stopping && matches!(binding.state, State::Unknown | State::Invoking))
    {
        return Err(lifecycle_error(
            AdapterLifecycleOperation::Invoke,
            &runtime.runtime_id,
            "runtime_unavailable",
            "remote runtime is not in a confirmed active state",
            "",
        ));
    }
    binding.state = if stopping {
        State::Stopping
    } else {
        State::Invoking
    };
    Ok(Some(key))
}

impl RuntimeAdapter for OpenShellRuntimeAdapter {
    fn start(&self, plan: &RunPlan, environment: EnvironmentHandle) -> Result<RuntimeHandle> {
        let config = serde_json::to_value(task_config(plan, &environment)?)
            .map_err(FabricError::SerializeJson)?;
        let key = resource_key(&environment)?;
        {
            let mut bindings = BINDINGS.lock().unwrap_or_else(|error| error.into_inner());
            let environment_bindings = bindings.entry(key.clone()).or_default();
            if let Some(binding) = environment_bindings
                .get(START_KEY)
                .or_else(|| environment_bindings.get(UNKNOWN_START_KEY))
                .or_else(|| environment_bindings.get(RELEASE_KEY))
            {
                return Err(FabricError::EnvironmentInUse {
                    environment_id: environment.environment_id.clone(),
                    runtime_id: binding
                        .runtime_id
                        .clone()
                        .unwrap_or_else(|| "unconfirmed".to_string()),
                });
            }
            environment_bindings.insert(
                START_KEY.to_string(),
                Binding {
                    state: State::Starting,
                    runtime_id: None,
                },
            );
        }
        // TODO(#428): Sandbox readiness can precede the Fabric server socket.
        // Poll the server-scope `fabric-agent-ctl check` here before Start.
        let command = TaskCommand::Start {
            agent: plan.agent_name.clone(),
            config,
        };
        let response = command_runtime(&environment, &command, 105);
        if let Ok(TaskCommandResponse::Rejected { error }) = &response
            && error.effects == "none"
            && !matches!(
                error.code.as_str(),
                "runtime_exists" | "cleanup_unconfirmed"
            )
        {
            BINDINGS
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .entry(key.clone())
                .or_default()
                .remove(START_KEY);
            return Err(lifecycle_error(
                AdapterLifecycleOperation::Start,
                "unconfirmed",
                error.code.clone(),
                "task rejected configuration before runtime startup",
                "",
            ));
        }
        set_start_state(&key, State::Unknown);
        let value = command_value(response?, AdapterLifecycleOperation::Start, "unconfirmed")?;
        let task: RuntimeHandle = serde_json::from_value(value).map_err(|_| {
            lifecycle_error(
                AdapterLifecycleOperation::Start,
                "unconfirmed",
                "invalid_response",
                "task returned an invalid SDK runtime handle",
                "",
            )
        })?;
        if task.runtime_id.trim().is_empty()
            || task.runtime_id.starts_with('\0')
            || task.agent_name != plan.agent_name
            || task.harness != harness(plan)
            || task.adapter_kind != adapter_kind(plan)
            || task.environment.provider != "local"
        {
            return Err(lifecycle_error(
                AdapterLifecycleOperation::Start,
                "unconfirmed",
                "invalid_response",
                "task runtime identity does not match the SDK configuration",
                "",
            ));
        }
        let runtime = RuntimeHandle {
            runtime_binding: runtime_binding(&task.runtime_id, None, plan, &environment)?,
            runtime_id: task.runtime_id,
            agent_name: task.agent_name,
            harness: task.harness,
            adapter_kind: task.adapter_kind,
            adapter_id: task.adapter_id,
            service_id: None,
            environment,
        };
        {
            let mut bindings = BINDINGS.lock().unwrap_or_else(|error| error.into_inner());
            let environment_bindings = bindings.entry(key).or_default();
            if environment_bindings.contains_key(&runtime.runtime_id) {
                return Err(lifecycle_error(
                    AdapterLifecycleOperation::Start,
                    "unconfirmed",
                    "invalid_response",
                    "task runtime ID is already bound in this environment",
                    "",
                ));
            }
            environment_bindings.remove(UNKNOWN_START_KEY);
            environment_bindings.remove(START_KEY);
            environment_bindings.insert(
                runtime.runtime_id.clone(),
                Binding {
                    state: State::Active,
                    runtime_id: Some(runtime.runtime_id.clone()),
                },
            );
        }
        Ok(runtime)
    }
    fn invoke(
        &self,
        plan: &RunPlan,
        runtime: &RuntimeHandle,
        mut request: RunRequest,
    ) -> Result<RunResult> {
        let timeout = invoke_seconds(plan)?;
        if request.request_id.is_empty() {
            request.request_id = new_id("request");
        }
        let command = TaskCommand::Invoke {
            agent: runtime.agent_name.clone(),
            runtime_id: runtime.runtime_id.clone(),
            request: serde_json::to_value(&request).map_err(FabricError::SerializeJson)?,
        };
        let key = claim(runtime, false)?.expect("active invocation binding");
        let mut confirmed = false;
        let result = (|| {
            let response = command_runtime(&runtime.environment, &command, timeout + 15)?;
            if let TaskCommandResponse::Rejected { error } = &response {
                confirmed = error.effects == "none";
            }
            let value = command_value(
                response,
                AdapterLifecycleOperation::Invoke,
                &runtime.runtime_id,
            )?;
            let mut result: RunResult = serde_json::from_value(value).map_err(|_| {
                lifecycle_error(
                    AdapterLifecycleOperation::Invoke,
                    &runtime.runtime_id,
                    "invalid_response",
                    "task returned an invalid SDK run result",
                    "",
                )
            })?;
            if result.runtime_id != runtime.runtime_id
                || result.request_id != request.request_id
                || result.agent_name != runtime.agent_name
                || result.harness != runtime.harness
                || result.adapter_kind != runtime.adapter_kind
                || result.invocation_id.is_empty()
            {
                return Err(lifecycle_error(
                    AdapterLifecycleOperation::Invoke,
                    &runtime.runtime_id,
                    "invalid_response",
                    "SDK run result identities do not match the invocation",
                    "",
                ));
            }
            adopt_task_what_ran(&mut result, &runtime.environment);
            for telemetry in &mut result.telemetry {
                telemetry.relay_enabled = telemetry
                    .metadata
                    .get("relay_enabled")
                    .and_then(Value::as_bool)
                    .unwrap_or(telemetry.provider.as_deref() == Some("relay"));
            }
            confirmed = true;
            collect_result_artifacts(plan, runtime, &mut result)?;
            Ok(result)
        })();
        if let Some(binding) = BINDINGS
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get_mut(&key)
            .and_then(|bindings| bindings.get_mut(&runtime.runtime_id))
            && binding.state == State::Invoking
        {
            binding.state = if confirmed {
                State::Active
            } else {
                State::Unknown
            };
        }
        result
    }
    fn invoke_openai_stream(
        &self,
        plan: &RunPlan,
        _runtime: &RuntimeHandle,
        _request: RunRequest,
        _transport: OpenAiStreamTransport,
    ) -> Result<RunResult> {
        Err(FabricError::UnsupportedRuntimeCapability {
            adapter_id: adapter_id(plan).unwrap_or_else(|| harness(plan)),
            capability: "streaming in an OpenShell sandbox",
        })
    }
    fn stop(&self, runtime: &RuntimeHandle) -> Result<Vec<FabricEvent>> {
        if let Some(key) = claim(runtime, true)? {
            let command = TaskCommand::Stop {
                agent: runtime.agent_name.clone(),
                runtime_id: runtime.runtime_id.clone(),
            };
            let result = (|| {
                let value = command_value(
                    command_runtime(&runtime.environment, &command, 30)?,
                    AdapterLifecycleOperation::Stop,
                    &runtime.runtime_id,
                )?;
                if !value.is_null() {
                    return Err(lifecycle_error(
                        AdapterLifecycleOperation::Stop,
                        &runtime.runtime_id,
                        "invalid_response",
                        "SDK stop must return null",
                        "",
                    ));
                }
                Ok(())
            })();
            set_state(
                &key,
                &runtime.runtime_id,
                if result.is_ok() {
                    State::Stopped
                } else {
                    State::Unknown
                },
            );
            result?;
        }
        Ok(vec![event_with_metadata(
            "runtime.stopped",
            "task SDK runtime stopped",
            BTreeMap::from([(
                "runtime_id".to_string(),
                Value::String(runtime.runtime_id.clone()),
            )]),
        )])
    }
}

/// Attribute the task core's `what_ran` to the task core, this host's
/// version and this host's environment provider. A task result without a
/// readable `what_ran`, from an older task core or one this host cannot
/// decode, keeps `what_ran` absent and is otherwise unchanged.
fn adopt_task_what_ran(result: &mut RunResult, environment: &EnvironmentHandle) {
    if let Some(what_ran) = result.what_ran.as_mut() {
        crate::what_ran::attribute_to_task_core(what_ran, &environment.provider);
    }
}

pub(super) fn release(environment: &EnvironmentHandle) -> Result<()> {
    let key = resource_key(environment)?;
    let previous = {
        let mut bindings = BINDINGS.lock().unwrap_or_else(|error| error.into_inner());
        let previous = bindings.get(&key).cloned();
        if let Some(environment_bindings) = &previous {
            if environment_bindings.contains_key(RELEASE_KEY) {
                return Err(FabricError::EnvironmentReleaseInProgress {
                    environment_id: environment.environment_id.clone(),
                });
            }
            if let Some(binding) = environment_bindings.values().find(|binding| {
                matches!(
                    binding.state,
                    State::Starting | State::Active | State::Invoking | State::Stopping
                )
            }) {
                return Err(FabricError::EnvironmentInUse {
                    environment_id: environment.environment_id.clone(),
                    runtime_id: binding
                        .runtime_id
                        .clone()
                        .unwrap_or_else(|| "unconfirmed".to_string()),
                });
            }
        }
        bindings.entry(key.clone()).or_default().insert(
            RELEASE_KEY.to_string(),
            Binding {
                state: State::Releasing,
                runtime_id: None,
            },
        );
        previous
    };
    let result = crate::environment::release_environment(environment);
    let mut bindings = BINDINGS.lock().unwrap_or_else(|error| error.into_inner());
    if result.is_ok() && environment.ownership == EnvironmentOwnership::FabricOwned {
        bindings.remove(&key);
    } else if let Some(previous) = previous {
        bindings.insert(key, previous);
    } else if result.is_err() {
        bindings.insert(
            key,
            BTreeMap::from([(
                UNKNOWN_START_KEY.to_string(),
                Binding {
                    state: State::Unknown,
                    runtime_id: None,
                },
            )]),
        );
    } else {
        bindings.remove(&key);
    }
    result.map(|_| ())
}

fn collect_result_artifacts(
    plan: &RunPlan,
    runtime: &RuntimeHandle,
    result: &mut RunResult,
) -> Result<()> {
    if result.artifacts.artifacts.is_empty() {
        return Ok(());
    }
    let task_root = task_root(
        runtime.environment.artifacts.as_deref(),
        "artifacts",
        Some("/sandbox/artifacts"),
    )?
    .to_string_lossy()
    .into_owned();
    let manifest_root = result
        .artifacts
        .root
        .as_deref()
        .and_then(Path::to_str)
        .unwrap_or(&task_root)
        .to_string();
    let relative = |path: &Path| -> Result<String> {
        let value = path
            .to_str()
            .ok_or_else(|| invalid_environment("artifact.path"))?;
        let absolute = if value.starts_with('/') {
            value.to_string()
        } else {
            format!("{}/{value}", manifest_root.trim_end_matches('/'))
        };
        let value = absolute
            .strip_prefix(&(task_root.trim_end_matches('/').to_string() + "/"))
            .ok_or_else(|| invalid_environment("artifact.path"))?;
        if value.is_empty()
            || value.contains('\\')
            || value.contains('\0')
            || value
                .split('/')
                .any(|part| part.is_empty() || part == "." || part == "..")
        {
            return Err(invalid_environment("artifact.path"));
        }
        Ok(value.to_string())
    };
    let mut declared = Vec::with_capacity(result.artifacts.artifacts.len());
    let mut paths = Vec::with_capacity(result.artifacts.artifacts.len());
    for artifact in std::mem::take(&mut result.artifacts.artifacts) {
        let path = relative(&artifact.path)?;
        if is_internal_sdk_artifact(&path) {
            continue;
        }
        paths.push(path);
        declared.push(artifact);
    }
    result.artifacts.artifacts = declared;
    if paths.is_empty() {
        result.artifacts.root = None;
        return Ok(());
    }
    let root = plan
        .config
        .runtime
        .artifacts
        .as_ref()
        .ok_or_else(|| invalid_environment("host_artifact_destination"))?;
    // Each invocation has its own destination, so earlier results remain readable.
    if result.runtime_id.contains(['/', '\\', '\0'])
        || result.invocation_id.contains(['/', '\\', '\0'])
        || matches!(result.runtime_id.as_str(), "." | "..")
        || matches!(result.invocation_id.as_str(), "." | "..")
    {
        return Err(invalid_environment("artifact.identity"));
    }
    let base = resolve_path(&plan.base_dir, root);
    let root = destination_root(&base, &result.runtime_id, &result.invocation_id)?;
    let mut seen = BTreeSet::new();
    let unique: Vec<String> = paths
        .iter()
        .filter(|path| seen.insert((*path).clone()))
        .cloned()
        .collect();
    let files = collect_artifacts(&runtime.environment, &unique)?;
    if files.len() != unique.len() {
        return Err(invalid_environment("artifact.correlation"));
    }
    for (path, file) in unique.iter().zip(files) {
        if &file.path != path {
            return Err(invalid_environment("artifact.correlation"));
        }
        write_artifact(&root, path, &file.content)?;
    }
    for (artifact, path) in result.artifacts.artifacts.iter_mut().zip(paths) {
        artifact.metadata.insert(
            "openshell.task_path".to_string(),
            Value::String(artifact.path.to_string_lossy().into_owned()),
        );
        artifact.path = root.join(path);
    }
    for telemetry in &mut result.telemetry {
        if telemetry.provider.as_deref() == Some("relay")
            && telemetry
                .uri
                .as_ref()
                .is_some_and(|uri| uri.starts_with('/') || uri.starts_with("file://"))
        {
            let directories: Vec<&Path> = result
                .artifacts
                .artifacts
                .iter()
                .filter(|artifact| matches!(artifact.kind.as_str(), "atif" | "atof"))
                .filter_map(|artifact| artifact.path.parent())
                .collect();
            if let Some(first) = directories.first() {
                let mut common = first.to_path_buf();
                while !directories
                    .iter()
                    .all(|directory| directory.starts_with(&common))
                {
                    if !common.pop() {
                        break;
                    }
                }
                if common.starts_with(&root)
                    && let Some(uri) = telemetry.uri.replace(common.to_string_lossy().into_owned())
                {
                    telemetry
                        .metadata
                        .insert("openshell.task_uri".to_string(), Value::String(uri));
                }
            }
        }
    }
    result.artifacts.root = Some(root);
    Ok(())
}

fn is_internal_sdk_artifact(path: &str) -> bool {
    path == ".fabric"
        || path.starts_with(".fabric/")
        || matches!(path, "workspace.patch" | "workspace-status.txt")
}

#[cfg(unix)]
fn child_directory(parent: &File, name: &str) -> std::io::Result<File> {
    use std::os::fd::{AsRawFd, FromRawFd};
    let name = std::ffi::CString::new(name).map_err(|_| std::io::ErrorKind::InvalidInput)?;
    // The parent is an owned directory descriptor and the C string is valid.
    let created = unsafe { libc::mkdirat(parent.as_raw_fd(), name.as_ptr(), 0o700) };
    if created < 0 && std::io::Error::last_os_error().kind() != std::io::ErrorKind::AlreadyExists {
        return Err(std::io::Error::last_os_error());
    }
    let descriptor = unsafe {
        libc::openat(
            parent.as_raw_fd(),
            name.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    if descriptor < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // openat returned a new descriptor; this File assumes sole ownership.
    Ok(unsafe { File::from_raw_fd(descriptor) })
}

#[cfg(unix)]
fn open_root(root: &Path) -> std::io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(root)
}

#[cfg(unix)]
fn destination_root(base: &Path, runtime_id: &str, invocation_id: &str) -> Result<PathBuf> {
    std::fs::create_dir_all(base).map_err(|source| FabricError::Write {
        path: base.to_path_buf(),
        source,
    })?;
    let base = base.canonicalize().map_err(|source| FabricError::Read {
        path: base.to_path_buf(),
        source,
    })?;
    let root = open_root(&base).map_err(|source| FabricError::Read {
        path: base.clone(),
        source,
    })?;
    let runtime = child_directory(&root, runtime_id).map_err(|source| FabricError::Write {
        path: base.join(runtime_id),
        source,
    })?;
    child_directory(&runtime, invocation_id).map_err(|source| FabricError::Write {
        path: base.join(runtime_id).join(invocation_id),
        source,
    })?;
    Ok(base.join(runtime_id).join(invocation_id))
}

#[cfg(unix)]
fn write_artifact(root: &Path, path: &str, bytes: &[u8]) -> Result<()> {
    use std::os::fd::{AsRawFd, FromRawFd};
    let mut parent = open_root(root).map_err(|source| FabricError::Read {
        path: root.to_path_buf(),
        source,
    })?;
    let parts: Vec<&str> = path.split('/').collect();
    for part in &parts[..parts.len() - 1] {
        parent = child_directory(&parent, part).map_err(|source| FabricError::Write {
            path: root.join(path),
            source,
        })?;
    }
    let name = std::ffi::CString::new(parts[parts.len() - 1])
        .map_err(|_| invalid_environment("artifact.destination"))?;
    // Resolve and create relative to the held parent; never follow a final symlink.
    let descriptor = unsafe {
        libc::openat(
            parent.as_raw_fd(),
            name.as_ptr(),
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            0o600,
        )
    };
    if descriptor < 0 {
        return Err(FabricError::Write {
            path: root.join(path),
            source: std::io::Error::last_os_error(),
        });
    }
    let mut output = unsafe { File::from_raw_fd(descriptor) };
    output
        .write_all(bytes)
        .map_err(|source| FabricError::Write {
            path: root.join(path),
            source,
        })
}

#[cfg(not(unix))]
fn destination_root(_base: &Path, _runtime_id: &str, _invocation_id: &str) -> Result<PathBuf> {
    Err(invalid_environment("platform"))
}
#[cfg(not(unix))]
fn write_artifact(_root: &Path, _path: &str, _bytes: &[u8]) -> Result<()> {
    Err(invalid_environment("platform"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn environment() -> EnvironmentHandle {
        EnvironmentHandle {
            environment_id: new_id("host-environment"),
            provider: "openshell".to_string(),
            control_location: ControlLocation::InEnvControl,
            ownership: EnvironmentOwnership::CallerOwned,
            workspace: Some(PathBuf::from("/sandbox/workspace")),
            artifacts: Some(PathBuf::from("/sandbox/artifacts")),
            env: BTreeMap::new(),
            metadata: BTreeMap::new(),
            connection: BTreeMap::from([
                (
                    "gateway".to_string(),
                    Value::String("https://fixture.invalid".to_string()),
                ),
                ("sandbox_id".to_string(), Value::String(new_id("sandbox"))),
            ]),
        }
    }

    /// A task result as the task core returns it, with `what_ran` from the
    /// shared reader vectors when given.
    fn task_result(what_ran: Option<Value>) -> RunResult {
        let mut result = serde_json::json!({
            "agent_name": "agent", "harness": "nvidia.fabric.claude", "adapter_kind": "python",
            "runtime_id": "runtime-task", "invocation_id": "invocation-1",
            "request_id": "request-1", "status": "succeeded", "output": "done"
        });
        if let Some(what_ran) = what_ran {
            result["what_ran"] = what_ran;
        }
        serde_json::from_value(result).expect("a task result decodes whatever what_ran holds")
    }

    fn task_what_ran() -> Value {
        let vectors: Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/what_ran/reader-vectors.json"
        ))
        .expect("reader vectors JSON");
        vectors["base"].clone()
    }

    fn facts_collected_by(value: &Value, collector: &str) -> usize {
        match value {
            Value::Object(map) if map.contains_key("source") => {
                usize::from(map.get("collected_by") == Some(&Value::from(collector)))
            }
            Value::Object(map) => map
                .values()
                .map(|value| facts_collected_by(value, collector))
                .sum(),
            _ => 0,
        }
    }

    #[test]
    fn task_what_ran_is_attributed_to_the_task_core_and_this_host() {
        for task_version in [crate::version(), "0.5.0"] {
            let mut what_ran = task_what_ran();
            what_ran["fabric"]["core_version"]["value"] = Value::from(task_version);
            let mut result = task_result(Some(what_ran));

            adopt_task_what_ran(&mut result, &environment());

            let what_ran = serde_json::to_value(result.what_ran).expect("what_ran JSON");
            assert_task_facts_are_attributed(&what_ran, task_version);
        }
    }

    fn assert_task_facts_are_attributed(what_ran: &Value, task_version: &str) {
        let host_version = serde_json::json!({"value": crate::version(), "source": "observed",
            "collected_by": "host_core", "method": "crate_version"});
        let task_core_version = serde_json::json!({"value": task_version, "source": "observed",
            "collected_by": "task_core", "method": "crate_version"});
        assert_eq!(what_ran["fabric"]["core_version"], task_core_version);
        assert_eq!(what_ran["fabric"]["host_core_version"], host_version);
        assert_eq!(
            facts_collected_by(what_ran, "host_core"),
            1,
            "only host_core_version is the host's fact"
        );
        assert_eq!(what_ran["environment"]["provider"], "openshell");
        assert_eq!(
            what_ran["models"]["default"]["requested"]["collected_by"],
            "task_core"
        );
        assert_eq!(
            what_ran["mcp"]["servers"]["docs"]["endpoint"]["collected_by"],
            "task_core"
        );
        assert_eq!(
            what_ran["harness"]["version"]["collected_by"], "adapter",
            "adapter facts keep their collector"
        );
    }

    #[test]
    fn task_result_without_readable_what_ran_is_kept_without_it() {
        let mut undecodable = task_what_ran();
        undecodable["harness"]["version"]["state"] = Value::from("withheld");
        for what_ran in [None, Some(undecodable)] {
            let mut result = task_result(what_ran);

            adopt_task_what_ran(&mut result, &environment());

            assert_eq!(result.what_ran, None);
            assert_eq!(result.status, RunStatus::Succeeded);
            assert_eq!(result.output, Value::from("done"));
        }
    }

    #[test]
    fn unknown_binding_survives_a_new_host_environment_handle() {
        let original = environment();
        let key = resource_key(&original).expect("resource key");
        let mut rebound = original.clone();
        rebound.environment_id = new_id("reattachment");
        assert_eq!(resource_key(&rebound).expect("rebound resource"), key);
        set_state(&key, "runtime-task", State::Unknown);
        let runtime = RuntimeHandle {
            runtime_id: "runtime-task".to_string(),
            runtime_binding: "unused".to_string(),
            agent_name: "echo".to_string(),
            harness: "echo".to_string(),
            adapter_kind: AdapterKind::Python,
            adapter_id: None,
            service_id: None,
            environment: rebound,
        };
        assert!(claim(&runtime, false).is_err());
        assert_eq!(
            claim(&runtime, true).expect("explicit stop of unknown runtime"),
            Some(key.clone())
        );
        assert!(claim(&runtime, false).is_err());
        BINDINGS.lock().expect("bindings").remove(&key);
    }

    #[test]
    fn runtime_bindings_are_independent_within_one_sandbox() {
        let environment = environment();
        let key = resource_key(&environment).expect("resource key");
        set_state(&key, "runtime-one", State::Active);
        set_state(&key, "runtime-two", State::Active);

        let runtime = |runtime_id: &str| RuntimeHandle {
            runtime_id: runtime_id.to_string(),
            runtime_binding: "unused".to_string(),
            agent_name: "echo".to_string(),
            harness: "echo".to_string(),
            adapter_kind: AdapterKind::Python,
            adapter_id: None,
            service_id: None,
            environment: environment.clone(),
        };

        assert_eq!(
            claim(&runtime("runtime-one"), false).expect("first runtime claim"),
            Some(key.clone())
        );
        assert_eq!(
            claim(&runtime("runtime-two"), false).expect("second runtime claim"),
            Some(key.clone())
        );
        set_state(&key, "runtime-one", State::Active);
        set_state(&key, "runtime-two", State::Active);
        BINDINGS.lock().expect("bindings").remove(&key);
    }

    #[test]
    fn internal_sdk_artifacts_are_not_exported_as_declared_files() {
        assert!(is_internal_sdk_artifact(".fabric/stdout.txt"));
        assert!(is_internal_sdk_artifact(".fabric/relay-config.json"));
        assert!(is_internal_sdk_artifact("workspace.patch"));
        assert!(is_internal_sdk_artifact("workspace-status.txt"));
        assert!(!is_internal_sdk_artifact("echo.json"));
        assert!(!is_internal_sdk_artifact("reports/review.json"));
    }

    #[test]
    fn task_environment_projection_preserves_named_environment_variables() {
        let projected = task_environment_config(
            PathBuf::from("/sandbox/workspace"),
            PathBuf::from("/sandbox/artifacts"),
            &BTreeMap::new(),
            Some(serde_json::json!(["MODEL_API_KEY"])),
        )
        .expect("task environment config");
        let value = serde_json::to_value(projected).expect("serialized task environment");
        assert_eq!(value["env_names"], serde_json::json!(["MODEL_API_KEY"]));
        assert!(value.get("connection").is_none());
    }

    #[test]
    fn sdk_telemetry_reference_keeps_public_fields_and_metadata() {
        let value = serde_json::json!({"provider": "relay", "kind": "trace", "uri": "/sandbox/relay", "trace_id": "trace-1", "metadata": {"relay_enabled": true}});
        let reference: TelemetryRef = serde_json::from_value(value).expect("public SDK telemetry");
        assert_eq!(reference.provider.as_deref(), Some("relay"));
        assert_eq!(reference.uri.as_deref(), Some("/sandbox/relay"));
        assert_eq!(reference.trace_id.as_deref(), Some("trace-1"));
        assert_eq!(reference.metadata["relay_enabled"], Value::Bool(true));
    }

    #[cfg(unix)]
    #[test]
    fn collection_never_creates_directories_through_a_host_symlink() {
        use std::os::unix::fs::symlink;
        let temporary = std::env::temp_dir().join(new_id("artifact-destination-test"));
        let base = temporary.join("host");
        let outside = temporary.join("outside");
        std::fs::create_dir_all(&outside).expect("outside directory");
        let root =
            destination_root(&base, "runtime-task", "invocation-1").expect("destination root");
        symlink(&outside, root.join("escape")).expect("parent symlink");
        assert!(write_artifact(&root, "escape/new/file", b"PRIVATE").is_err());
        assert!(!outside.join("new").exists());
        symlink(&outside, base.join("another-runtime")).expect("runtime symlink");
        assert!(destination_root(&base, "another-runtime", "invocation-2").is_err());
        assert!(!outside.join("invocation-2").exists());
        std::fs::remove_dir_all(temporary).expect("remove fixture");
    }

    #[cfg(unix)]
    #[test]
    fn collected_files_are_exclusive_and_preserve_earlier_results() {
        let temporary = std::env::temp_dir().join(new_id("artifact-exclusive-test"));
        let root =
            destination_root(&temporary, "runtime-task", "invocation-1").expect("destination root");
        write_artifact(&root, "nested/receipt", b"FIRST").expect("write artifact");
        assert!(write_artifact(&root, "nested/receipt", b"SECOND").is_err());
        assert_eq!(
            std::fs::read(root.join("nested/receipt")).expect("read preserved artifact"),
            b"FIRST"
        );
        std::fs::remove_dir_all(temporary).expect("remove fixture");
    }
}

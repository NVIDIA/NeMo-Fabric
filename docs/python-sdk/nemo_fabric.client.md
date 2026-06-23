


# <kbd>module</kbd> `nemo_fabric.client`
Python client for NeMo Fabric. 

The SDK uses the native Rust binding when the package is installed with maturin. It falls back to the Fabric CLI when the native extension is not available or when a CLI command is configured explicitly. 



---


## <kbd>class</kbd> `FabricCliError`
Raised when the Fabric CLI exits unsuccessfully. 


### <kbd>method</kbd> `__init__`

```python
__init__(
    command: 'Sequence[str]',
    returncode: 'int',
    stdout: 'str',
    stderr: 'str'
) → None
```









---


## <kbd>class</kbd> `FabricNativeUnavailableError`
Raised when a typed-config SDK method needs the native extension. 





---


## <kbd>class</kbd> `FabricClient`
Primary Python entrypoint for NeMo Fabric. 

Resolve an agent's configuration, plan it, diagnose it, and run it through a harness, returning normalized results -- without importing harness-specific code. Every method accepts either an agent-package directory (``agent.yaml`` plus ``profiles/``) or, via the ``*_config`` methods, an in-memory typed config, so a consumer can build the Fabric slice from its own job/deployment config without materializing an agent directory. 

The client uses the native Rust binding when installed and otherwise falls back to the ``fabric`` CLI; pass ``command`` to force a specific CLI invocation. 



**Attributes:**
 
 - <b>`command`</b>:  Explicit CLI command to use instead of the native binding,  e.g. ``("cargo", "run", "-q", "-p", "fabric-cli", "--")``. 
 - <b>`cwd`</b>:  Working directory for CLI invocations. 



**Example:**
 ``` import asyncio```
    >>> from nemo_fabric import FabricClient
    >>> async def main() -> None:
    ...     client = FabricClient()
    ...     result = await client.run(
    ...         "examples/code-review-agent",
    ...         profile="hermes_sdk",
    ...         input_text="review this diff",
    ...     )
    ...     print(result["status"])
    >>> asyncio.run(main())



### <kbd>method</kbd> `__init__`

```python
__init__(
    command: 'tuple[str, ] | None' = None,
    cwd: 'Path | None' = None
) → None
```








---


### <kbd>method</kbd> `doctor`

```python
doctor(
    path: 'str | Path',
    profile: 'str | Sequence[str] | None' = None
) → dict[str, Any]
```

Diagnose a run plan without installing or running the harness. 

Checks adapter availability, capability mappings, and requirements. 



**Args:**
 
 - <b>`path`</b>:  Agent package directory or config file. 
 - <b>`profile`</b>:  A profile name or an ordered sequence of profile names. 



**Returns:**
 
 - <b>`A diagnostics report dict (e.g. ``{"checks"`</b>:  [...]}``). 

---


### <kbd>method</kbd> `doctor_config`

```python
doctor_config(
    config: 'Mapping[str, Any] | Any',
    profile_configs: 'Sequence[Mapping[str, Any] | Any] | None' = None,
    base_dir: 'str | Path | None' = None
) → dict[str, Any]
```

Diagnose an in-memory typed config without running the harness. 

Requires the native binding. 



**Args:**
 
 - <b>`config`</b>:  A mapping or Pydantic-like object describing the agent config. 
 - <b>`profile_configs`</b>:  Ordered profile-config overrides applied in order. 
 - <b>`base_dir`</b>:  Directory used to resolve relative paths the config references. 



**Returns:**
 A diagnostics report dict. 



**Raises:**
 
 - <b>`FabricNativeUnavailableError`</b>:  If the native binding is not installed. 

---


### <kbd>method</kbd> `inspect`

```python
inspect(
    path: 'str | Path',
    profile: 'str | Sequence[str] | None' = None
) → dict[str, Any]
```

Resolve and return the effective config after applying profiles. 



**Args:**
 
 - <b>`path`</b>:  Agent package directory or config file. 
 - <b>`profile`</b>:  A profile name or an ordered sequence of profile names. 



**Returns:**
 The effective config as a dict (``effective-config`` schema). 

---


### <kbd>method</kbd> `plan`

```python
plan(
    path: 'str | Path',
    profile: 'str | Sequence[str] | None' = None
) → dict[str, Any]
```

Resolve an agent and profiles into an executable run plan. 

Does not run the harness. 



**Args:**
 
 - <b>`path`</b>:  Agent package directory or config file. 
 - <b>`profile`</b>:  A profile name or an ordered sequence of profile names. 



**Returns:**
 A run plan dict (``run-plan`` schema). 

---


### <kbd>method</kbd> `plan_config`

```python
plan_config(
    config: 'Mapping[str, Any] | Any',
    profile_configs: 'Sequence[Mapping[str, Any] | Any] | None' = None,
    base_dir: 'str | Path | None' = None
) → dict[str, Any]
```

Resolve an in-memory typed config into a run plan. 

The typed-config path lets a consumer build the Fabric slice in code, with no agent directory on disk. Requires the native binding. 



**Args:**
 
 - <b>`config`</b>:  A mapping or Pydantic-like object (exposing ``model_dump()``  or ``dict()``) describing the agent config. 
 - <b>`profile_configs`</b>:  Ordered profile-config overrides applied in order. 
 - <b>`base_dir`</b>:  Directory used to resolve any relative paths the config  references (skills, repos); omit for self-contained configs. 



**Returns:**
 A run plan dict (``run-plan`` schema). 



**Raises:**
 
 - <b>`FabricNativeUnavailableError`</b>:  If the native binding is not installed. 

---


### <kbd>method</kbd> `run`

```python
run(
    path: 'str | Path',
    profile: 'str | Sequence[str] | None' = None,
    input_text: 'str' = '',
    input_file: 'str | Path | None' = None,
    request: 'dict[str, Any] | None' = None,
    request_file: 'str | Path | None' = None
) → dict[str, Any]
```

Run an agent/profile through the selected adapter. 

The per-invocation request is shaped by the ``run-request`` schema; supply it through ``input_text``, ``input_file``, a ``request`` dict, or ``request_file`` (in increasing precedence). 



**Args:**
 
 - <b>`path`</b>:  Agent package directory or config file. 
 - <b>`profile`</b>:  A profile name or an ordered sequence of profile names. 
 - <b>`input_text`</b>:  Text input for the harness. 
 - <b>`input_file`</b>:  Path to a file whose contents are used as the input. 
 - <b>`request`</b>:  A full request dict (``run-request`` schema). 
 - <b>`request_file`</b>:  Path to a JSON file containing the request. 



**Returns:**
 A normalized result dict (``run-result`` schema). 

---


### <kbd>method</kbd> `run_config`

```python
run_config(
    config: 'Mapping[str, Any] | Any',
    profile_configs: 'Sequence[Mapping[str, Any] | Any] | None' = None,
    base_dir: 'str | Path | None' = None,
    input_text: 'str' = '',
    input_file: 'str | Path | None' = None,
    request: 'dict[str, Any] | None' = None,
    request_file: 'str | Path | None' = None
) → dict[str, Any]
```

Run an in-memory typed config through the selected adapter. 

The typed-config path for consumers that build the Fabric slice in code with no agent directory. Requires the native binding. 



**Args:**
 
 - <b>`config`</b>:  A mapping or Pydantic-like object describing the agent config. 
 - <b>`profile_configs`</b>:  Ordered profile-config overrides applied in order. 
 - <b>`base_dir`</b>:  Directory used to resolve relative paths the config references. 
 - <b>`input_text`</b>:  Text input for the harness. 
 - <b>`input_file`</b>:  Path to a file whose contents are used as the input. 
 - <b>`request`</b>:  A full request dict (``run-request`` schema). 
 - <b>`request_file`</b>:  Path to a JSON file containing the request. 



**Returns:**
 A normalized result dict (``run-result`` schema). 



**Raises:**
 
 - <b>`FabricNativeUnavailableError`</b>:  If the native binding is not installed. 

---


### <kbd>method</kbd> `validate`

```python
validate(path: 'str | Path') → str
```

Validate an agent directory or config file. 



**Args:**
 
 - <b>`path`</b>:  Agent package directory or config file. 



**Returns:**
 A human-readable validation status message. 



**Raises:**
 
 - <b>`FabricCliError`</b>:  If validation fails (CLI backend). 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

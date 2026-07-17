# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer experimentation interface backed by the NeMo Fabric SDK."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemo_fabric.client import Fabric
from nemo_fabric.errors import FabricError
from nemo_fabric.factories import load_config_factory
from nemo_fabric.models import FabricConfig, RunRequest
from nemo_fabric.presets import EXAMPLES, PRESETS


class SourceError(ValueError):
    """Invalid CLI source selector."""


@dataclass(frozen=True)
class SelectedSource:
    """A complete typed config and its path-resolution base."""

    config: FabricConfig
    base_dir: Path
    label: str


def load_factory(spec: str) -> FabricConfig:
    """Import and invoke a ``module:callable`` Fabric config factory."""

    try:
        return load_config_factory(spec)
    except Exception as error:
        raise SourceError(str(error)) from error


def select_source(args: argparse.Namespace) -> SelectedSource:
    """Resolve one mutually exclusive CLI selector."""

    override = Path(args.base_dir).resolve() if args.base_dir else None
    if args.variant and not args.example:
        raise SourceError("--variant requires --example")
    if args.preset:
        try:
            preset = PRESETS[args.preset]
        except KeyError as error:
            raise SourceError(_unknown("preset", args.preset, PRESETS)) from error
        return SelectedSource(preset.factory(), override or preset.base_dir, f"preset:{preset.name}")
    if args.example:
        try:
            example = EXAMPLES[args.example]
        except KeyError as error:
            raise SourceError(_unknown("example", args.example, EXAMPLES)) from error
        variant = args.variant or example.default_variant
        try:
            factory = example.variants[variant]
        except KeyError as error:
            choices = ", ".join(sorted(example.variants))
            raise SourceError(f"unknown variant {variant!r} for {example.name}; available: {choices}") from error
        return SelectedSource(factory(), override or example.base_dir, f"example:{example.name}:{variant}")
    if args.factory:
        return SelectedSource(load_factory(args.factory), override or Path.cwd(), f"factory:{args.factory}")
    raise SourceError("select exactly one of --preset, --example, or --factory")


def _unknown(kind: str, name: str, registry: dict[str, Any]) -> str:
    return f"unknown {kind} {name!r}; available: {', '.join(sorted(registry))}"


def _add_selector(parser: argparse.ArgumentParser) -> None:
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--preset", help="Use a maintained, complete built-in config.")
    sources.add_argument("--example", help="Use an installed example config.")
    sources.add_argument("--factory", help="Import a custom module:callable returning FabricConfig.")
    parser.add_argument("--variant", help="Variant for --example (defaults to the example default).")
    parser.add_argument("--base-dir", help="Override the source base directory for relative paths.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nemo-fabric",
        description=__doc__,
        epilog="For stable application integration, construct FabricConfig and use the SDK directly.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    for name in ("plan", "doctor", "run", "chat"):
        command = subcommands.add_parser(name)
        _add_selector(command)
        if name in {"run", "chat"}:
            inputs = command.add_mutually_exclusive_group()
            inputs.add_argument(
                "--input",
                default=None,
                help="Invocation text; defaults to an empty string.",
            )
            inputs.add_argument(
                "--input-file",
                type=Path,
                help="Read invocation text from this path, or use - for stdin.",
            )
        if name == "run":
            command.add_argument("--request-json", help="JSON object or @path containing a RunRequest.")
        command.add_argument("--output", type=Path, help="Write JSON output to this path instead of stdout.")

    preset = subcommands.add_parser("preset", help="Discover built-in presets.")
    preset_commands = preset.add_subparsers(dest="preset_command", required=True)
    preset_commands.add_parser("list")
    preset_show = preset_commands.add_parser("show")
    preset_show.add_argument("name")

    example = subcommands.add_parser("example", help="Discover installed examples.")
    example_commands = example.add_subparsers(dest="example_command", required=True)
    example_commands.add_parser("list")
    example_show = example_commands.add_parser("show")
    example_show.add_argument("name")
    example_init = example_commands.add_parser("init")
    example_init.add_argument("name")
    example_init.add_argument("destination", type=Path)

    subcommands.add_parser("version", help="Show the installed Fabric runtime version.")
    return parser


def _input(args: argparse.Namespace) -> str:
    if args.input_file is None:
        return args.input or ""
    if args.input_file == Path("-"):
        return sys.stdin.read()
    return args.input_file.read_text(encoding="utf-8")


def _request(args: argparse.Namespace) -> RunRequest:
    if args.request_json and (args.input is not None or args.input_file is not None):
        raise SourceError("--input/--input-file and --request-json are mutually exclusive")
    if not args.request_json:
        return RunRequest(input=_input(args))
    raw = args.request_json
    if raw.startswith("@"):
        raw = Path(raw[1:]).read_text(encoding="utf-8")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SourceError(f"invalid request JSON: {error}") from error
    if not isinstance(value, dict):
        raise SourceError("request JSON must be an object")
    return RunRequest.from_mapping(value)


def _emit(value: Any, output: Path | None) -> None:
    mapping = value.to_mapping() if hasattr(value, "to_mapping") else value
    rendered = json.dumps(mapping, indent=2)
    if output is None:
        print(rendered)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")


async def _execute(args: argparse.Namespace) -> int:
    selected = select_source(args)
    fabric = Fabric()
    if args.command == "plan":
        result = fabric.plan(selected.config, base_dir=selected.base_dir)
    elif args.command == "doctor":
        result = await fabric.doctor(selected.config, base_dir=selected.base_dir)
    elif args.command == "run":
        result = await fabric.run(selected.config, base_dir=selected.base_dir, request=_request(args))
    else:
        return await _chat(fabric, selected, args)
    _emit(result, args.output)
    return 1 if getattr(result, "status", None) == "fail" else 0


async def _chat(fabric: Fabric, selected: SelectedSource, args: argparse.Namespace) -> int:
    transcript: list[dict[str, Any]] = []
    async with await fabric.start_runtime(selected.config, base_dir=selected.base_dir) as runtime:
        pending = _input(args) if args.input is not None or args.input_file is not None else None
        while True:
            if pending is None:
                try:
                    pending = input("you> ")
                except EOFError:
                    break
            if pending.strip().lower() in {"/exit", "/quit"}:
                break
            result = await runtime.invoke(input=pending)
            transcript.append(result.to_mapping())
            response = getattr(result.output, "response", None)
            print(f"agent> {response if response is not None else result.status}")
            pending = None
    if args.output is not None:
        _emit(transcript, args.output)
    return 0


def _discovery(args: argparse.Namespace) -> int | None:
    if args.command == "version":
        print(_runtime_version())
        return 0
    if args.command == "preset":
        if args.preset_command == "list":
            for name in sorted(PRESETS):
                print(name)
        else:
            try:
                item = PRESETS[args.name]
            except KeyError as error:
                raise SourceError(_unknown("preset", args.name, PRESETS)) from error
            _emit(item.discovery(), None)
        return 0
    if args.command == "example":
        if args.example_command == "list":
            for name in sorted(EXAMPLES):
                print(name)
        elif args.example_command == "show":
            try:
                item = EXAMPLES[args.name]
            except KeyError as error:
                raise SourceError(_unknown("example", args.name, EXAMPLES)) from error
            _emit(
                {
                    "name": item.name,
                    "description": item.description,
                    "default_variant": item.default_variant,
                    "variants": sorted(item.variants),
                    "init_supported": item.template_dir is not None,
                },
                None,
            )
        else:
            _init_example(args.name, args.destination)
        return 0
    return None


def _runtime_version() -> str:
    try:
        return importlib.metadata.version("nemo-fabric-runtime")
    except importlib.metadata.PackageNotFoundError:
        from nemo_fabric import _native

        return _native.version()


def _init_example(name: str, destination: Path) -> None:
    try:
        item = EXAMPLES[name]
    except KeyError as error:
        raise SourceError(_unknown("example", name, EXAMPLES)) from error
    if item.template_dir is None:
        raise SourceError(f"example {name!r} cannot be copied")
    destination = destination.resolve()
    if not destination.name.isidentifier():
        raise SourceError("destination name must be a valid Python package name")
    if destination.exists():
        raise SourceError(f"destination already exists: {destination}")
    shutil.copytree(item.template_dir, destination)
    print(f"Created {name} in {destination}")
    print(
        f"From {destination.parent}, run: nemo-fabric run "
        f"--factory {destination.name}.config:build_config "
        f"--base-dir {destination} --input 'Review the workspace'"
    )


def cli(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    arguments = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    if not arguments:
        parser.print_help()
        return 0
    args = parser.parse_args(arguments)
    try:
        discovery = _discovery(args)
        return discovery if discovery is not None else asyncio.run(_execute(args))
    except KeyboardInterrupt:
        print("nemo-fabric: interrupted", file=sys.stderr)
        return 130
    except (FabricError, SourceError, OSError, ValueError) as error:
        print(f"nemo-fabric: {error}", file=sys.stderr)
        return 2


def main() -> None:
    """Console-script entrypoint."""

    raise SystemExit(cli())


if __name__ == "__main__":
    main()

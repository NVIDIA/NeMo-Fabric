# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install packaged plugins into an explicitly selected Hermes Agent checkout."""

import argparse
import json
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory


def install_tavily(hermes_root: Path) -> Path:
    """Install Tavily without enabling it or overwriting a different installation.

    The caller must provide an existing Hermes checkout. The returned path is
    Hermes' native ``web/tavily`` plugin directory. Repeating installation of
    identical files is a no-op; changed files require explicit removal first.
    No credentials, runtime state, or network access are needed.
    """
    root = hermes_root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    parent = root
    for component in ("plugins", "web"):
        parent /= component
        if parent.is_symlink():
            raise FileExistsError(f"plugin directory is a symlink: {parent}")
        parent.mkdir(exist_ok=True)
    destination = parent / "tavily"
    if destination.is_symlink():
        raise FileExistsError(f"plugin directory is a symlink: {destination}")
    source = files("nemo_fabric_adapters.hermes").joinpath("plugins", "tavily")
    contents = {
        name: source.joinpath(name).read_bytes()
        for name in ("__init__.py", "plugin.yaml")
    }
    if destination.exists():
        if all(
            not (destination / name).is_symlink()
            and (destination / name).is_file()
            and (destination / name).read_bytes() == content
            for name, content in contents.items()
        ):
            return destination
        raise FileExistsError(f"plugin directory has different contents: {destination}")
    # Assemble the complete plugin before making it discoverable by Hermes.
    with TemporaryDirectory(prefix=".fabric-tavily-", dir=parent) as temporary:
        staging = Path(temporary)
        for name, content in contents.items():
            path = staging / name
            path.write_bytes(content)
            path.chmod(0o644)
        staging.chmod(0o755)
        staging.rename(destination)
    return destination


def main() -> None:
    """Install packaged Hermes plugins and print their identifiers and paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hermes-root", type=Path, required=True, help="existing Hermes Agent checkout"
    )
    args = parser.parse_args()
    try:
        destination = install_tavily(args.hermes_root)
    except OSError as error:
        parser.exit(2, f"Cannot install Hermes plugin: {error}\n")
    print(json.dumps({"web/tavily": str(destination)}))


if __name__ == "__main__":
    main()

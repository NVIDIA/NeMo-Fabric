# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Console-script bridge to the Rust experimentation CLI."""

from __future__ import annotations

import sys

from nemo_fabric import _native


def main() -> None:
    """Run the single Rust/Clap CLI implementation."""

    try:
        _native.cli_main(sys.argv[1:])
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run ``mcp_server_time`` and record each ``get_current_time`` call.

Each call appends one JSON line to the file named by ``TIME_MCP_LOG``, so a
test can prove the agent reached the real MCP server.
"""

import json
import os

from mcp_server_time import main
from mcp_server_time import server

_original = server.TimeServer.get_current_time


def _logged(self, timezone_name):
    """Record the call, then answer it normally."""
    with open(os.environ["TIME_MCP_LOG"], "a", encoding="utf-8") as log:
        log.write(json.dumps({"tool": "get_current_time", "timezone": timezone_name}))
        log.write("\n")
    return _original(self, timezone_name)


server.TimeServer.get_current_time = _logged

if __name__ == "__main__":
    main()

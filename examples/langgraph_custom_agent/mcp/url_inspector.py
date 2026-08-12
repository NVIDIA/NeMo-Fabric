# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small deterministic stdio MCP server for the phishing example."""

import ipaddress
from typing import TypedDict
from urllib.parse import urlparse

from mcp.server.fastmcp import FastMCP


class UrlInspection(TypedDict):
    """Non-authoritative URL metadata returned to the example graph."""

    hostname: str
    indicators: list[str]


server = FastMCP("email-link-inspector", log_level="ERROR")


@server.tool()
def inspect_url(url: str) -> UrlInspection:
    """Inspect URL syntax for a few deterministic phishing indicators."""

    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    indicators: list[str] = []
    if parsed.scheme != "https":
        indicators.append("unencrypted_link")
    if "@" in parsed.netloc:
        indicators.append("embedded_credentials")
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        indicators.append("ip_literal_host")
    if hostname.startswith("xn--") or ".xn--" in hostname:
        indicators.append("punycode_host")
    if hostname.endswith(".invalid"):
        indicators.append("reserved_test_domain")
    return {"hostname": hostname, "indicators": indicators}


if __name__ == "__main__":
    server.run(transport="stdio")

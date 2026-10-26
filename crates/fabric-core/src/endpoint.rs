// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Secret-free identities for configured endpoints.

use std::net::Ipv6Addr;

/// The endpoint identity of `url`: its lowercase scheme and host, and its
/// port as a number. An omitted port becomes the scheme's default, 80 for
/// `http` and `ws` and 443 for `https` and `wss`; other schemes keep a port
/// only when one is written. An IPv6 host keeps its brackets in canonical,
/// compressed lowercase form, such as `[2001:db8::1]`. The user info, path,
/// query and fragment never enter it, because a hosted service can carry a
/// key in any of them.
///
/// `None` when `url` has no `scheme://host`, when an `@` follows the
/// authority (a password holding an unencoded `/`, `\`, `?` or `#` ends
/// the authority early, so the host cannot be told apart from the user
/// info), when the host holds a character other than ASCII letters, digits,
/// `-`, `.`, `_` and `~`, when a bracketed host is not an IPv6 address
/// (a scoped address with a zone ID is rejected too), or when the port is
/// not a number below 65,536.
pub(crate) fn endpoint_identity(url: &str) -> Option<String> {
    let (scheme, rest) = url.trim().split_once("://")?;
    if !is_scheme(scheme) {
        return None;
    }
    let authority_end = rest.find(['/', '\\', '?', '#']).unwrap_or(rest.len());
    let (authority, tail) = rest.split_at(authority_end);
    if tail.contains('@') {
        return None;
    }
    let host_port = authority
        .rsplit_once('@')
        .map_or(authority, |(_, host_port)| host_port);
    let (host, port) = split_host_port(host_port)?;
    let scheme = scheme.to_ascii_lowercase();
    let mut identity = format!("{scheme}://{host}");
    if let Some(port) = port.or_else(|| default_port(&scheme)) {
        identity.push_str(&format!(":{port}"));
    }
    Some(identity)
}

/// Whether `scheme` is a URL scheme: a letter, then letters, digits, `+`,
/// `-` or `.`.
fn is_scheme(scheme: &str) -> bool {
    let mut characters = scheme.chars();
    characters
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic())
        && characters.all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '+' | '-' | '.')
        })
}

/// Split `host[:port]` into the lowercase host, with a bracketed IPv6
/// address in canonical form, and the port. `None` when the host is empty,
/// holds a character no host name holds, or is bracketed but not an IPv6
/// address, or when the port is not a number below 65,536. An empty port
/// reads as no port.
fn split_host_port(host_port: &str) -> Option<(String, Option<u16>)> {
    let (host, port) = match host_port.strip_prefix('[') {
        Some(bracketed) => {
            let (address, after) = bracketed.split_once(']')?;
            let address: Ipv6Addr = address.parse().ok()?;
            let port = match after {
                "" => None,
                after => Some(after.strip_prefix(':')?),
            };
            (format!("[{address}]"), port)
        }
        None => {
            let (host, port) = match host_port.split_once(':') {
                Some((host, port)) => (host, Some(port)),
                None => (host_port, None),
            };
            let host_character = |character: char| {
                character.is_ascii_alphanumeric() || matches!(character, '-' | '.' | '_' | '~')
            };
            if host.is_empty() || !host.chars().all(host_character) {
                return None;
            }
            (host.to_ascii_lowercase(), port)
        }
    };
    match port.filter(|port| !port.is_empty()) {
        None => Some((host, None)),
        Some(port) if port.bytes().all(|byte| byte.is_ascii_digit()) => {
            port.parse::<u16>().ok().map(|port| (host, Some(port)))
        }
        Some(_) => None,
    }
}

fn default_port(scheme: &str) -> Option<u16> {
    match scheme {
        "http" | "ws" => Some(80),
        "https" | "wss" => Some(443),
        _ => None,
    }
}

/// The file name of a `stdio` server's executable: the part after the last
/// `/` or `\\`.
///
/// `None` when the executable holds an `=` or any whitespace anywhere, or
/// when the file name is empty or holds a character other than ASCII
/// letters, digits, `.`, `_`, `+` and `-`. The plan keeps arguments in their
/// own field, so whitespace in the executable field means arguments may
/// have been written into it, and an argument can put a secret after a `/`
/// (`npx --token a/SECRET`). A leading `NAME=value` can do the same
/// (`TOKEN=a/SECRET server`). Both checks run before the folder is dropped.
/// A path with a space in a folder is rejected too. This is a conservative
/// extractor, not a shell parser.
pub(crate) fn executable_name(executable: &str) -> Option<String> {
    if executable.contains('=') || executable.chars().any(char::is_whitespace) {
        return None;
    }
    let name = executable.rsplit(['/', '\\']).next().unwrap_or_default();
    let plain = |character: char| {
        character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '+' | '-')
    };
    (!name.is_empty() && name.chars().all(plain)).then(|| name.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_identity_keeps_scheme_host_and_port_only() {
        for (url, identity) in [
            (
                "HTTP://user:pw@10.0.0.5:8000/v1/?key=abc#frag",
                "http://10.0.0.5:8000",
            ),
            ("https://API.example.com/", "https://api.example.com:443"),
            ("http://[::1]/v1", "http://[::1]:80"),
            (
                "https://[2001:DB8:0:0::1]:8443/",
                "https://[2001:db8::1]:8443",
            ),
            ("https://example.com", "https://example.com:443"),
            ("https://example.com:443", "https://example.com:443"),
            (
                "https://actions.zapier.com/mcp/sk-ak-SECRET/sse",
                "https://actions.zapier.com:443",
            ),
            ("https://u%40x:p%2Fq@host:8443/v1", "https://host:8443"),
            ("https://host:/v1", "https://host:443"),
            ("https://host:0443/v1", "https://host:443"),
            ("WSS://Host.Example/s", "wss://host.example:443"),
            ("grpc://host:9000", "grpc://host:9000"),
        ] {
            assert_eq!(endpoint_identity(url).as_deref(), Some(identity), "{url}");
        }
    }

    /// A URL whose host cannot be told apart from its user info, or that is
    /// not a URL, has no identity, so no part of a password becomes a host.
    #[test]
    fn endpoint_identity_rejects_ambiguous_or_invalid_urls() {
        for url in [
            "https://user:123/SECRET@host/x",
            "https://u:p#SECRET@host/x",
            "https://u:p?SECRET@host/x",
            "https://user:pw@host/p@x",
            r"https://api.example.com\SECRET@secret-host",
            "https://ärzte.example/mcp",
            "https://server.example.com/@owner/server/mcp",
            "https://u:p/x",
            "https://host:99999",
            "https://host:8a",
            "https://",
            "1http://host",
            "http://ho st/",
            "http://[::1/",
            "http://[]/",
            "https://[SECRET]/mcp",
            "http://[::::]/",
            "http://[fe80::1%25eth0]/",
            "http://[fe80::1%eth0]/",
            "http://[::1]x/",
            "localhost:8000",
            "not a url",
        ] {
            assert_eq!(endpoint_identity(url), None, "{url}");
        }
    }

    #[test]
    fn executable_name_is_the_file_name_without_its_folder() {
        for (executable, name) in [
            ("db-mcp", Some("db-mcp")),
            ("/home/alice/.venv/bin/python", Some("python")),
            (r"C:\tools\npx.cmd", Some("npx.cmd")),
            (r"C:\MCP\server.exe", Some("server.exe")),
        ] {
            assert_eq!(executable_name(executable).as_deref(), name, "{executable}");
        }
    }

    /// An `=` or whitespace anywhere rejects the executable before its
    /// folder is dropped, so no part of an assignment or an argument
    /// becomes the name.
    #[test]
    fn executable_name_rejects_assignments_and_unplain_names() {
        for executable in [
            "OPENAI_API_KEY=SECRET npx",
            "TOKEN=prefix/SECRET",
            r"TOKEN=prefix\SECRET",
            "TOKEN=https://SECRET",
            "/opt/server --token=SECRET",
            "npx --api-key SECRET",
            "npx -y @org/server-github --token SECRET",
            "/opt/My Tools/db server",
            "npx --token a/SECRET",
            r"npx --token a\SECRET",
            r"C:\Program Files\MCP\server.exe",
            "/opt/My Tools/db-server",
            "db-mcp\t",
            " db-mcp",
            "'quoted server'",
            "/opt/tools/ärzte",
            "",
            "/opt/tools/",
        ] {
            assert_eq!(executable_name(executable), None, "{executable}");
        }
    }
}

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reconcile PR limits. Default to read-only; use only trusted default-branch code."""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


ARTIFACT = "pr-cleanup-state"
MARKER = "<!-- fabric-pr-cleanup:"


class CleanupError(RuntimeError):
    pass


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise CleanupError("Activity timestamps must include a timezone")
    return parsed


def iso(value):
    return value.astimezone(timezone.utc).isoformat()


def load_policy(path):
    policy = json.loads(Path(path).read_text())
    for key in ("max_open_prs", "inactive_days"):
        if type(policy.get(key)) is not int or policy[key] < 1:
            raise CleanupError(f"Invalid {key}")
    for key in ("exemption_org", "exemption_team"):
        if not re.fullmatch(r"[A-Za-z0-9-]+", policy.get(key, "")):
            raise CleanupError(f"Invalid {key}")
    return policy


class GitHub:
    """Use the runner's gh CLI for authenticated API calls and artifact redirects."""

    def __init__(self, repo, token):
        self.repo = repo
        self.token = token

    def raw(self, path, method="GET", payload=None):
        command = ["gh", "api", "--hostname", "github.com", "--method", method, path]
        if payload is not None:
            command += ["--input", "-"]
        try:
            result = subprocess.run(
                command,
                input=json.dumps(payload).encode() if payload is not None else None,
                capture_output=True,
                timeout=90,
                env={**os.environ, "GH_TOKEN": self.token},
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise CleanupError(
                f"GitHub API could not complete: {method} {path.split('?')[0]}"
            ) from error
        if result.returncode:
            # Do not print response bodies, credentials, or attacker-controlled titles.
            raise CleanupError(f"GitHub API failed: {method} {path.split('?')[0]}")
        return result.stdout

    def request(self, path, method="GET", payload=None):
        return json.loads(self.raw(path, method, payload))

    def pages(self, path, key=None):
        items = []
        separator = "&" if "?" in path else "?"
        for page in range(1, 1001):
            response = self.request(f"{path}{separator}per_page=100&page={page}")
            batch = response[key] if key else response
            if not isinstance(batch, list):
                raise CleanupError("Expected a paginated list")
            items.extend(batch)
            if len(batch) < 100:
                return items
        raise CleanupError("Pagination limit reached; refusing incomplete data")

    def pulls(self):
        return self.pages(f"repos/{self.repo}/pulls?state=open")

    def maintainers(self, policy):
        path = f"orgs/{policy['exemption_org']}/teams/{policy['exemption_team']}/members?role=maintainer"
        return {member["login"].casefold() for member in self.pages(path)}

    def previous_state(self):
        """Accept artifacts only from this trusted workflow's completed runs."""
        path = (
            f"repos/{self.repo}/actions/workflows/pr-cleanup.yml/runs?status=completed"
        )
        # A newly merged workflow has no previous runs. The workflow now exists,
        # so its runs endpoint returns an empty list rather than needing a 404 fallback.
        for run in self.pages(path, "workflow_runs"):
            event = run["event"]
            if event not in {"schedule", "workflow_dispatch", "pull_request_target"}:
                continue
            if event != "pull_request_target" and run["head_branch"] != os.environ.get(
                "DEFAULT_BRANCH", "main"
            ):
                continue
            artifacts = self.pages(
                f"repos/{self.repo}/actions/runs/{run['id']}/artifacts", "artifacts"
            )
            for artifact in artifacts:
                if artifact["name"] != ARTIFACT or artifact["expired"]:
                    continue
                archive = self.raw(
                    f"repos/{self.repo}/actions/artifacts/{artifact['id']}/zip"
                )
                with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
                    entry = zipped.getinfo("state.json")
                    if entry.file_size > 10_000_000:
                        raise CleanupError("State artifact is too large")
                    state = json.loads(zipped.read(entry))
                if state["repository"] != self.repo or state["version"] != 1:
                    raise CleanupError("Unexpected state artifact identity")
                timestamp(state["observed_at"])
                return state
        return None


def excess_prs(pulls, maintainers, limit):
    authors = defaultdict(list)
    for pr in pulls:
        if pr["user"]["login"].casefold() not in maintainers:
            authors[pr["user"]["login"].casefold()].append(pr)
    excess = []
    for group in authors.values():
        ordered = sorted(
            group, key=lambda pr: (timestamp(pr["created_at"]), pr["number"])
        )
        excess.extend(ordered[limit:])
    return sorted(
        excess, key=lambda pr: (timestamp(pr["created_at"]), pr["number"]), reverse=True
    )


def observe_heads(pulls, previous, now, days, event=None):
    """Bootstrap conservatively: commit author/committer dates are not push times."""
    continuous = previous and timedelta(0) <= now - timestamp(
        previous["observed_at"]
    ) < timedelta(days=days)
    records = {}
    for pr in pulls:
        number = str(pr["number"])
        old = previous.get("pulls", {}).get(number) if continuous else None
        head = pr["head"]["sha"]
        changed = not old or old["head"] != head
        # Also record synchronize even if a force-push returns to a previously seen SHA.
        synchronized = (
            event
            and event.get("action") == "synchronize"
            and event.get("number") == pr["number"]
        )
        records[number] = {
            "head": head,
            "last_push_observed_at": iso(now)
            if changed or synchronized
            else old["last_push_observed_at"],
        }
    return {"version": 1, "observed_at": iso(now), "pulls": records}


def human(item):
    user = item.get("user") or {}
    return user.get("type") == "User" and not user.get("login", "").endswith("[bot]")


def last_activity(pr, record, comments, reviews, review_comments, timeline):
    dates = [timestamp(pr["created_at"]), timestamp(record["last_push_observed_at"])]
    for event in timeline:
        if event.get("event") in {"reopened", "head_ref_force_pushed"}:
            dates.append(timestamp(event["created_at"]))
    for item in [*comments, *reviews, *review_comments]:
        if human(item):
            value = (
                item.get("updated_at")
                or item.get("submitted_at")
                or item.get("created_at")
            )
            # Pending, unsubmitted reviews do not constitute public activity.
            if value:
                dates.append(timestamp(value))
    return max(dates)


class Cleanup:
    def __init__(self, api, members_api, policy, state, now, apply=False):
        self.api = api
        self.members_api = members_api
        self.policy = policy
        self.state = state
        self.now = now
        self.apply = apply
        self.rows = []
        self.errors = []

    def log(self, number, status, reason):
        self.rows.append({"number": number, "status": status, "reason": reason})

    def activity(self, pr):
        number = pr["number"]
        base = f"repos/{self.api.repo}"
        comments = self.api.pages(f"{base}/issues/{number}/comments")
        timeline = self.api.pages(f"{base}/issues/{number}/timeline")
        reviews = self.api.pages(f"{base}/pulls/{number}/reviews")
        review_comments = self.api.pages(f"{base}/pulls/{number}/comments")
        record = self.state["pulls"][str(number)]
        if record["head"] != pr["head"]["sha"]:
            raise CleanupError("PR head changed after observation; retry next run")
        return (
            last_activity(pr, record, comments, reviews, review_comments, timeline),
            comments,
            timeline,
        )

    def close(self, pr, reason, explanation):
        """One comment per reason and reopen cycle, including after partial failures."""
        number = pr["number"]
        base = f"repos/{self.api.repo}"
        comments = self.api.pages(f"{base}/issues/{number}/comments")
        timeline = self.api.pages(f"{base}/issues/{number}/timeline")
        cycle = max(
            (event["id"] for event in timeline if event.get("event") == "closed"),
            default=0,
        )
        marker = f"{MARKER}{reason}:{cycle} -->"
        if not any(
            (comment.get("body") or "").startswith(marker)
            and (comment.get("user") or {}).get("login") == "github-actions[bot]"
            for comment in comments
        ):
            self.api.request(
                f"{base}/issues/{number}/comments",
                "POST",
                {"body": f"{marker}\n{explanation}"},
            )
        # If closure fails, the existing marker allows a retry without another comment.
        self.api.request(f"{base}/pulls/{number}", "PATCH", {"state": "closed"})
        self.log(number, "closed", explanation)

    def reconcile(self, pulls):
        cutoff = self.now - timedelta(days=self.policy["inactive_days"])
        simulated_closed = set()
        for pr in pulls:
            number = pr["number"]
            try:
                activity, _, _ = self.activity(pr)
                if activity >= cutoff:
                    continue
                explanation = (
                    f"No qualifying activity since {iso(activity)} (more than {self.policy['inactive_days']} days). "
                    "Reopen when ready to resume work and tag the maintainers for review."
                )
                if self.apply:
                    fresh = self.api.request(f"repos/{self.api.repo}/pulls/{number}")
                    if fresh["state"] != "open" or self.activity(fresh)[0] >= cutoff:
                        self.log(
                            number,
                            "skipped",
                            "State or activity changed before closure",
                        )
                        continue
                    self.close(fresh, "inactive", explanation)
                else:
                    self.log(number, "would close", explanation)
                simulated_closed.add(number)
            except (CleanupError, KeyError, ValueError) as error:
                self.errors.append(f"PR #{number}: inactivity skipped: {error}")

        # Failed inactivity closures still count toward the real cap.
        remaining = (
            self.api.pulls()
            if self.apply
            else [pr for pr in pulls if pr["number"] not in simulated_closed]
        )
        try:
            maintainers = self.members_api.maintainers(self.policy)
        except (CleanupError, KeyError, ValueError) as error:
            self.errors.append(
                f"Cap enforcement skipped: team maintainer lookup failed: {error}"
            )
            return
        limit = self.policy["max_open_prs"]
        for pr in excess_prs(remaining, maintainers, limit):
            number = pr["number"]
            try:
                current = remaining
                if self.apply:
                    maintainers = self.members_api.maintainers(self.policy)
                    current = self.api.pulls()
                    if number not in {
                        item["number"]
                        for item in excess_prs(current, maintainers, limit)
                    }:
                        self.log(
                            number,
                            "skipped",
                            "Capacity, team role, or PR state changed",
                        )
                        continue
                author = pr["user"]["login"].casefold()
                retained = sorted(
                    [
                        item
                        for item in current
                        if item["user"]["login"].casefold() == author
                    ],
                    key=lambda item: (timestamp(item["created_at"]), item["number"]),
                )[:limit]
                links = ", ".join(f"#{item['number']}" for item in retained)
                explanation = (
                    f"The limit is {limit} open PRs per author, including drafts. "
                    f"Keeping the oldest: {links}. Merge or close one before reopening this PR."
                )
                if self.apply:
                    self.close(pr, "cap", explanation)
                else:
                    self.log(number, "would close", explanation)
            except (CleanupError, KeyError, ValueError) as error:
                self.errors.append(
                    f"PR #{number}: cap closure skipped or failed: {error}"
                )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--policy", default=".github/pr-cleanup.json")
    parser.add_argument("--state", default=".tmp/pr-cleanup/state.json")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Explicitly enable PR comments and closures",
    )
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repo):
        parser.error("Invalid repository")
    policy = load_policy(args.policy)
    api = GitHub(args.repo, os.environ["GH_TOKEN"])
    members_api = GitHub(
        args.repo, os.environ.get("PR_CLEANUP_MEMBERS_TOKEN") or os.environ["GH_TOKEN"]
    )
    now = datetime.now(timezone.utc)
    pulls = api.pulls()
    errors = []
    try:
        previous = api.previous_state()
    except (CleanupError, KeyError, ValueError, zipfile.BadZipFile) as error:
        previous = None
        errors.append(
            f"Push observation history unavailable; starting a fresh observation period: {error}"
        )
    event = None
    if os.environ.get("GITHUB_EVENT_NAME") == "pull_request_target":
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    state = observe_heads(pulls, previous, now, policy["inactive_days"], event)
    state["repository"] = args.repo
    cleanup = Cleanup(api, members_api, policy, state, now, args.apply)
    cleanup.reconcile(pulls)
    # Persist even on partial failures: missing history never makes a PR look older.
    path = Path(args.state)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n")
    report = {
        "mode": "apply" if args.apply else "dry-run",
        "decisions": cleanup.rows,
        "errors": errors + cleanup.errors,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        # JSON is indented Markdown, so untrusted API text cannot inject headings/HTML.
        with open(summary, "a") as stream:
            stream.write(
                "## PR cleanup\n\n"
                + "\n".join("    " + line for line in rendered.splitlines())
                + "\n"
            )
    return int(bool(report["errors"]))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (
        CleanupError,
        KeyError,
        ValueError,
        OSError,
        subprocess.TimeoutExpired,
    ) as error:
        print(f"Cleanup stopped without further mutations: {error}", file=sys.stderr)
        sys.exit(1)

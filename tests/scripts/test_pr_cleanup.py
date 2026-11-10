# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Policy and API integration tests; run with unittest without project dependencies."""

from __future__ import annotations

import copy
import io
import json
import sys
import tempfile
import unittest
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "ci"))
import pr_cleanup as cleanup  # noqa: E402


NOW = datetime(2026, 11, 9, 12, tzinfo=timezone.utc)
POLICY = {
    "max_open_prs": 3,
    "inactive_days": 28,
    "exemption_org": "NVIDIA",
    "exemption_team": "nemo-fabric-developers",
}


def pr(number, author="developer", days=40, draft=False):
    return {
        "number": number,
        "user": {"login": author},
        "created_at": cleanup.iso(NOW - timedelta(days=days)),
        "head": {"sha": f"head-{number}"},
        "state": "open",
        "draft": draft,
    }


def comment(days=1, bot=False, body="review feedback"):
    return {
        "user": {
            "login": "github-actions[bot]" if bot else "reviewer",
            "type": "Bot" if bot else "User",
        },
        "updated_at": cleanup.iso(NOW - timedelta(days=days)),
        "body": body,
    }


def state(pulls, days=40):
    return {
        "version": 1,
        "repository": "NVIDIA/NeMo-Fabric",
        "observed_at": cleanup.iso(NOW - timedelta(hours=1)),
        "pulls": {
            str(item["number"]): {
                "head": item["head"]["sha"],
                "last_push_observed_at": cleanup.iso(NOW - timedelta(days=days)),
            }
            for item in pulls
        },
    }


class FakeGitHub:
    repo = "NVIDIA/NeMo-Fabric"

    def __init__(self, pulls, maintainers=()):
        self.items = copy.deepcopy(pulls)
        self.exempt = set(maintainers)
        self.comments = {}
        self.timeline = {}
        self.reviews = {}
        self.review_comments = {}
        self.writes = []
        self.fail = set()
        self.promote = None
        self.members_calls = 0
        self.fresh_hook = None

    def pulls(self):
        return copy.deepcopy([item for item in self.items if item["state"] == "open"])

    def maintainers(self, policy):
        self.members_calls += 1
        if "members" in self.fail:
            raise cleanup.CleanupError("membership read failed")
        if self.promote and self.members_calls > 1:
            self.exempt.add(self.promote)
        return self.exempt

    def pages(self, path):
        if any(fragment in path for fragment in self.fail):
            raise cleanup.CleanupError("lookup failed")
        number = int(path.split("/")[-2])
        if path.endswith("/timeline"):
            return copy.deepcopy(self.timeline.get(number, []))
        if path.endswith("/reviews"):
            return copy.deepcopy(self.reviews.get(number, []))
        if "/pulls/" in path:
            return copy.deepcopy(self.review_comments.get(number, []))
        return copy.deepcopy(self.comments.get(number, []))

    def request(self, path, method="GET", payload=None):
        number = int(
            path.split("/")[-2] if path.endswith("/comments") else path.split("/")[-1]
        )
        item = next(item for item in self.items if item["number"] == number)
        if method == "GET":
            if self.fresh_hook:
                self.fresh_hook(self, number)
            return copy.deepcopy(item)
        if method == "PATCH" and "close" in self.fail:
            raise cleanup.CleanupError("closure failed")
        self.writes.append((method, number, payload))
        if method == "POST":
            self.comments.setdefault(number, []).append(
                comment(bot=True, body=payload["body"])
            )
        else:
            item["state"] = payload["state"]
            self.timeline.setdefault(number, []).append(
                {"event": "closed", "id": 100 + len(self.writes)}
            )
        return copy.deepcopy(item)


class CleanupTests(unittest.TestCase):
    def run_policy(self, api, *, apply=False, days=0):
        worker = cleanup.Cleanup(api, api, POLICY, state(api.items, days), NOW, apply)
        worker.reconcile(api.pulls())
        return worker

    def test_cap_drafts_all_branches_and_newest_first(self):
        items = [pr(i, days=50 - i, draft=i % 2 == 0) for i in range(1, 7)]
        items[3]["base"] = {"ref": "release/1"}
        self.assertEqual(
            [item["number"] for item in cleanup.excess_prs(items, set(), 3)], [6, 5, 4]
        )
        self.assertEqual(cleanup.excess_prs(items[:3], set(), 3), [])

    def test_ties_reopening_and_case_insensitive_authors(self):
        items = [
            pr(i, author="Developer" if i % 2 else "developer") for i in range(1, 5)
        ]
        items[0]["updated_at"] = cleanup.iso(
            NOW
        )  # reopening does not change original order
        self.assertEqual(
            [item["number"] for item in cleanup.excess_prs(items, set(), 3)], [4]
        )
        self.assertEqual(cleanup.excess_prs(items, {"developer"}, 3), [])

    def test_cap_exemption_is_team_role_not_repository_role(self):
        api = FakeGitHub([pr(i, author="repo-admin") for i in range(1, 5)])
        api.items[-1]["permissions"] = {"admin": True}
        worker = self.run_policy(api)
        self.assertEqual([row["number"] for row in worker.rows], [4])
        api.exempt.add("repo-admin")
        self.assertEqual(self.run_policy(api).rows, [])

    def test_dry_run_has_no_writes_and_apply_closes_newest(self):
        api = FakeGitHub([pr(i) for i in range(1, 6)])
        self.run_policy(api)
        self.assertEqual(api.writes, [])
        self.run_policy(api, apply=True)
        self.assertEqual(
            [number for method, number, _ in api.writes if method == "PATCH"], [5, 4]
        )
        self.assertEqual(len(api.pulls()), 3)

    def test_missing_team_permission_blocks_only_cap(self):
        api = FakeGitHub([pr(i) for i in range(1, 5)])
        api.fail.add("members")
        worker = self.run_policy(api, apply=True)
        self.assertTrue(worker.errors)
        self.assertEqual(api.writes, [])
        worker = self.run_policy(api, apply=True, days=40)
        self.assertEqual(len(api.pulls()), 0)  # inactivity is independent of exemptions
        self.assertTrue(worker.errors)

    def test_promotion_before_cap_closure(self):
        api = FakeGitHub([pr(i) for i in range(1, 5)])
        api.promote = "developer"
        self.run_policy(api, apply=True)
        self.assertEqual(api.writes, [])

    def test_inactivity_boundary_and_team_maintainers(self):
        api = FakeGitHub([pr(1, draft=True)], {"developer"})
        self.assertEqual(self.run_policy(api, days=28).rows, [])
        self.assertEqual(self.run_policy(api, days=29).rows[0]["number"], 1)

    def test_human_comments_edits_reviews_and_review_comments_reset(self):
        for surface in ("comments", "reviews", "review_comments"):
            with self.subTest(surface=surface):
                api = FakeGitHub([pr(1)])
                getattr(api, surface)[1] = [comment()]
                self.assertEqual(self.run_policy(api, days=40).rows, [])

    def test_bot_ci_label_and_draft_changes_do_not_reset(self):
        api = FakeGitHub([pr(1, draft=True)])
        api.items[0]["updated_at"] = cleanup.iso(NOW)
        api.comments[1] = [comment(bot=True)]
        api.timeline[1] = [{"event": "labeled", "created_at": cleanup.iso(NOW)}]
        self.assertEqual(self.run_policy(api, days=40).rows[0]["number"], 1)

    def test_reopening_and_force_push_reset(self):
        for kind in ("reopened", "head_ref_force_pushed"):
            api = FakeGitHub([pr(1)])
            api.timeline[1] = [{"event": kind, "created_at": cleanup.iso(NOW)}]
            self.assertEqual(self.run_policy(api, days=40).rows, [])

    def test_inactivity_before_cap_and_dry_run_parity(self):
        items = [pr(i, days=50 - i) for i in range(1, 5)]
        dry_api, real_api = FakeGitHub(items), FakeGitHub(items)
        for api in (dry_api, real_api):
            for number in (2, 3, 4):
                api.comments[number] = [comment()]
        dry = self.run_policy(dry_api, days=40)
        real = self.run_policy(real_api, apply=True, days=40)
        self.assertEqual([row["number"] for row in dry.rows], [1])
        self.assertEqual([row["number"] for row in real.rows], [1])
        self.assertEqual(len(real_api.pulls()), 3)

    def test_activity_lookup_failure_skips_inactivity(self):
        api = FakeGitHub([pr(1)])
        api.fail.add("timeline")
        worker = self.run_policy(api, apply=True, days=40)
        self.assertTrue(worker.errors)
        self.assertEqual(api.writes, [])

    def test_last_minute_comment_merge_and_head_change(self):
        for change in ("comment", "merge", "push"):
            api = FakeGitHub([pr(1)])

            def update(api, number):
                if change == "comment":
                    api.comments[number] = [comment()]
                elif change == "merge":
                    api.items[0]["state"] = "closed"
                else:
                    api.items[0]["head"]["sha"] = "new-head"

            api.fresh_hook = update
            self.run_policy(api, apply=True, days=40)
            self.assertEqual(api.writes, [])

    def test_partial_failure_retry_deduplicates_and_reopen_comments_again(self):
        api = FakeGitHub([pr(1)])
        worker = cleanup.Cleanup(api, api, POLICY, state(api.items), NOW, True)
        api.fail.add("close")
        with self.assertRaises(cleanup.CleanupError):
            worker.close(api.items[0], "cap", "reason")
        api.fail.clear()
        worker.close(api.items[0], "cap", "reason")
        self.assertEqual(sum(method == "POST" for method, _, _ in api.writes), 1)
        api.items[0]["state"] = "open"
        worker.close(api.items[0], "cap", "reason")
        self.assertEqual(sum(method == "POST" for method, _, _ in api.writes), 2)

    def test_bootstrap_changed_heads_gap_and_synchronize(self):
        pulls = [pr(1)]
        first = cleanup.observe_heads(pulls, None, NOW, 28)
        self.assertEqual(first["pulls"]["1"]["last_push_observed_at"], cleanup.iso(NOW))
        previous = state(pulls)
        continued = cleanup.observe_heads(pulls, previous, NOW, 28)
        self.assertEqual(
            continued["pulls"]["1"]["last_push_observed_at"],
            previous["pulls"]["1"]["last_push_observed_at"],
        )
        for event in (None, {"action": "synchronize", "number": 1}):
            changed = copy.deepcopy(pulls)
            if event is None:
                changed[0]["head"]["sha"] = "new"
            updated = cleanup.observe_heads(changed, previous, NOW, 28, event)
            self.assertEqual(
                updated["pulls"]["1"]["last_push_observed_at"], cleanup.iso(NOW)
            )
        previous["observed_at"] = cleanup.iso(NOW - timedelta(days=28))
        reset = cleanup.observe_heads(pulls, previous, NOW, 28)
        self.assertEqual(reset["pulls"]["1"]["last_push_observed_at"], cleanup.iso(NOW))

    def test_policy_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            for key, value in (
                ("max_open_prs", 0),
                ("inactive_days", True),
                ("exemption_team", "../../other"),
            ):
                path.write_text(json.dumps({**POLICY, key: value}))
                with self.assertRaises(cleanup.CleanupError):
                    cleanup.load_policy(path)

    def test_pagination_and_team_role_query(self):
        api = cleanup.GitHub("NVIDIA/NeMo-Fabric", "not-a-real-token")
        with patch.object(
            api, "request", side_effect=[[{"login": "A"}] * 100, [{"login": "B"}]]
        ) as request:
            self.assertEqual(api.maintainers(POLICY), {"a", "b"})
            self.assertIn(
                "role=maintainer&per_page=100&page=2", request.call_args.args[0]
            )
        with patch.object(
            api, "request", side_effect=cleanup.CleanupError("page failed")
        ):
            with self.assertRaises(cleanup.CleanupError):
                api.pages("repos/example/pulls")

    def test_state_artifacts_ignore_untrusted_run_events_and_expired_state(self):
        api = cleanup.GitHub("NVIDIA/NeMo-Fabric", "not-a-real-token")
        runs = [
            {"id": 1, "event": "pull_request", "head_branch": "main"},
            {"id": 2, "event": "workflow_dispatch", "head_branch": "attacker"},
            {"id": 3, "event": "schedule", "head_branch": "main"},
        ]
        with patch.object(
            api,
            "pages",
            side_effect=[runs, [{"id": 1, "name": cleanup.ARTIFACT, "expired": True}]],
        ):
            self.assertIsNone(api.previous_state())
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("state.json", json.dumps(state([pr(1)])))
        with (
            patch.object(
                api,
                "pages",
                side_effect=[
                    runs,
                    [{"id": 4, "name": cleanup.ARTIFACT, "expired": False}],
                ],
            ),
            patch.object(api, "raw", return_value=archive.getvalue()),
        ):
            self.assertEqual(api.previous_state()["version"], 1)

    def test_missing_or_expired_history_starts_fresh(self):
        pulls = [pr(1)]
        for previous in (
            None,
            {**state(pulls), "observed_at": cleanup.iso(NOW - timedelta(days=90))},
        ):
            observed = cleanup.observe_heads(pulls, previous, NOW, 28)
            self.assertEqual(
                observed["pulls"]["1"]["last_push_observed_at"], cleanup.iso(NOW)
            )

    def test_spoofed_cleanup_marker_is_still_human_activity(self):
        api = FakeGitHub([pr(1)])
        api.comments[1] = [comment(body="<!-- fabric-pr-cleanup:fake -->")]
        self.assertEqual(self.run_policy(api, days=40).rows, [])

    def test_spoofed_comment_does_not_suppress_closure_explanation(self):
        api = FakeGitHub([pr(1)])
        api.comments[1] = [comment(body="<!-- fabric-pr-cleanup:cap:0 -->")]
        worker = cleanup.Cleanup(api, api, POLICY, state(api.items), NOW, True)
        worker.close(api.items[0], "cap", "real reason")
        self.assertEqual(sum(method == "POST" for method, _, _ in api.writes), 1)

    def test_main_defaults_to_no_mutations_and_saves_observations(self):
        api = FakeGitHub([pr(i) for i in range(1, 5)])
        api.previous_state = lambda: None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "policy.json"
            policy.write_text(json.dumps(POLICY))
            with (
                patch.object(cleanup, "GitHub", return_value=api),
                patch.dict(
                    cleanup.os.environ,
                    {
                        "GH_TOKEN": "test-token",
                        "GITHUB_EVENT_NAME": "workflow_dispatch",
                        "GITHUB_STEP_SUMMARY": str(root / "summary.md"),
                    },
                ),
                patch("sys.stdout", new_callable=io.StringIO) as output,
            ):
                code = cleanup.main(
                    [
                        "--repo",
                        api.repo,
                        "--policy",
                        str(policy),
                        "--state",
                        str(root / "state.json"),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertEqual(api.writes, [])
            self.assertEqual(json.loads(output.getvalue())["mode"], "dry-run")
            self.assertEqual(
                json.loads((root / "state.json").read_text())["repository"], api.repo
            )
            self.assertTrue((root / "summary.md").exists())

    def test_api_timeout_is_a_reportable_lookup_failure(self):
        api = cleanup.GitHub("NVIDIA/NeMo-Fabric", "not-a-real-token")
        with patch.object(
            cleanup.subprocess,
            "run",
            side_effect=cleanup.subprocess.TimeoutExpired("gh", 90),
        ):
            with self.assertRaises(cleanup.CleanupError):
                api.pulls()

    def test_gh_errors_do_not_echo_token_or_response_body(self):
        api = cleanup.GitHub("NVIDIA/NeMo-Fabric", "secret-token")
        response = type(
            "Result", (), {"returncode": 1, "stdout": b"", "stderr": b"secret-token"}
        )()
        with patch.object(cleanup.subprocess, "run", return_value=response):
            with self.assertRaises(cleanup.CleanupError) as error:
                api.raw("repos/NVIDIA/NeMo-Fabric/pulls")
        self.assertNotIn("secret-token", str(error.exception))


if __name__ == "__main__":
    unittest.main()

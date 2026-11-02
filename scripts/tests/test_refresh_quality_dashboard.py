"""Evidence attribution and failure behavior for the explicit refresh command."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refresh_quality_dashboard as refresh

SHA = "a" * 40
NOW = "2026-10-17T12:00:00Z"
RUN_TIME = "2026-10-15T10:00:00Z"
RUN = {
    "id": 101,
    "conclusion": "success",
    "head_branch": "main",
    "event": "push",
    "head_sha": SHA,
    "updated_at": RUN_TIME,
    "html_url": "https://github.com/owner/repo/actions/runs/101",
}
ARTIFACT = {
    "id": 202,
    "name": "executable-code-coverage-report",
    "expired": False,
    "workflow_run": {"id": 101, "head_sha": SHA},
}


def archive(rows=None, **overrides):
    data = {
        "schema_version": 1,
        "skills": [{"id": "alpha", "actual": 82.5}] if rows is None else rows,
    }
    data.update(overrides)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as zf:
        zf.writestr("per-skill-coverage.json", json.dumps(data))
    return output.getvalue()


class FakeGitHub:
    base = "repos/owner/repo"

    def __init__(self):
        self.runs = [dict(RUN)]
        self.artifacts = [{**ARTIFACT, "workflow_run": dict(ARTIFACT["workflow_run"])}]
        self.payload = archive()
        self.downloaded = []

    def pages(self, endpoint):
        assert endpoint == "issues?state=open&per_page=100"
        return [
            {"title": "[P0] failure", "labels": []},
            {"title": "a bug", "labels": [{"name": "P1"}]},
            {"title": "[P1] change", "pull_request": {}, "labels": []},
            {"title": "[P2] feature", "labels": []},
        ]

    def get(self, endpoint):
        if endpoint.startswith("actions/workflows/ci.yml/runs?"):
            return {"workflow_runs": self.runs}
        assert endpoint == "actions/runs/101/artifacts?per_page=100"
        return {"total_count": len(self.artifacts), "artifacts": self.artifacts}

    def artifact(self, artifact_id):
        self.downloaded.append(artifact_id)
        return self.payload


@pytest.fixture
def root(tmp_path):
    (tmp_path / "skills-index.yaml").write_text(
        yaml.safe_dump(
            {
                "skills": [
                    {
                        "id": "alpha",
                        "status": "production",
                        "verification": {"unit_tests": "passed"},
                    },
                    {"id": "beta", "status": "beta"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "docs/assets").mkdir(parents=True)
    return tmp_path


def test_collected_evidence_keeps_historical_measurement_separate(root):
    gh = FakeGitHub()
    snapshot = refresh.collect(root, gh, NOW, "b" * 40)
    assert snapshot["as_of"] == NOW
    assert snapshot["per_skill_coverage_pct"] == {"alpha": 82.5}
    assert snapshot["open_high_severity_issues"] == 2
    assert snapshot["input_provenance"]["per_skill_coverage_pct"] == {
        "commit": SHA,
        "measured_at": RUN_TIME,
        "source": "https://github.com/owner/repo/actions/runs/101/artifacts/202",
    }
    assert snapshot["input_provenance"]["catalog"]["commit"] == "b" * 40
    assert gh.downloaded == [202]
    assert snapshot["production_verification_declarations"]["unit_tests"] == {
        "passed": 1,
        "not_verified": 0,
        "not_applicable": 0,
    }
    assert snapshot["executed_smoke_count"] == refresh.NOT_YET_MEASURED
    assert snapshot["latest_successful_live_sample"] == refresh.NOT_YET_MEASURED


@pytest.mark.parametrize("mode", ["expired", "missing", "no-runs"])
def test_unavailable_evidence_is_unknown_and_never_carries_old_values(root, mode):
    (root / refresh.DEFAULT_SNAPSHOT_PATH).write_text('{"per_skill_coverage_pct":{"alpha":99}}')
    gh = FakeGitHub()
    if mode == "expired":
        gh.artifacts[0]["expired"] = True
    elif mode == "missing":
        gh.artifacts = []
    else:
        gh.runs = []
    snapshot = refresh.collect(root, gh, NOW, SHA)
    assert snapshot["per_skill_coverage_pct"] == {}
    assert "per_skill_coverage_pct" not in snapshot["input_provenance"]
    assert not gh.downloaded


@pytest.mark.parametrize("value", [True, -1, 101, float("nan"), float("inf"), "82"])
def test_invalid_percentages_rejected(value):
    with pytest.raises(refresh.RefreshError):
        refresh.coverage_values(archive([{"id": "alpha", "actual": value}]), {"alpha"})


@pytest.mark.parametrize(
    "rows",
    [
        [{"id": "alpha", "actual": 10}, {"id": "alpha", "actual": 20}],
        [{"id": "unknown", "actual": 10}],
        [{}],
        None,
    ],
)
def test_bad_rows_rejected(rows):
    if rows is None:
        payload = archive(skills=None)
    else:
        payload = archive(rows)
    with pytest.raises(refresh.RefreshError):
        refresh.coverage_values(payload, {"alpha"})


@pytest.mark.parametrize("schema", [True, 2, "1", None])
def test_invalid_schema_rejected(schema):
    with pytest.raises(refresh.RefreshError):
        refresh.coverage_values(archive(schema_version=schema), {"alpha"})


def test_nullable_coverage_and_repo_scripts_not_published():
    assert (
        refresh.coverage_values(
            archive(
                [
                    {"id": "alpha", "actual": None},
                    {"id": "repo-scripts", "actual": 91},
                ]
            ),
            {"alpha"},
        )
        == {}
    )


def test_artifact_identity_mismatch_rejected(root):
    gh = FakeGitHub()
    gh.artifacts[0]["workflow_run"]["id"] = 999
    with pytest.raises(refresh.RefreshError, match="identity"):
        refresh.collect(root, gh, NOW, SHA)
    assert gh.downloaded == []


@pytest.mark.parametrize(
    "field,value", [("conclusion", "failure"), ("event", "pull_request"), ("head_branch", "other")]
)
def test_untrusted_ci_run_rejected(root, field, value):
    gh = FakeGitHub()
    gh.runs[0][field] = value
    with pytest.raises(refresh.RefreshError):
        refresh.collect(root, gh, NOW, SHA)


def test_duplicate_artifact_rejected(root):
    gh = FakeGitHub()
    gh.artifacts *= 2
    with pytest.raises(refresh.RefreshError, match="ambiguous"):
        refresh.collect(root, gh, NOW, SHA)


def test_archive_paths_are_never_extracted(tmp_path):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as zf:
        zf.writestr("../per-skill-coverage.json", "{}")
    with pytest.raises(refresh.RefreshError):
        refresh.coverage_values(output.getvalue(), {"alpha"})
    assert list(tmp_path.iterdir()) == []


def test_late_api_failure_leaves_snapshot_and_pages_unchanged(root, monkeypatch):
    paths = [
        root / refresh.DEFAULT_SNAPSHOT_PATH,
        root / "docs/en/quality-dashboard.md",
        root / "docs/ja/quality-dashboard.md",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("old content")
    gh = FakeGitHub()

    def fail_download(_):
        raise refresh.RefreshError("API failed")

    gh.artifact = fail_download
    monkeypatch.setattr(refresh, "GitHub", lambda _: gh)
    monkeypatch.setattr(refresh, "command", lambda _: SHA.encode())
    assert refresh.main(["--project-root", str(root)]) == 1
    assert all(p.read_text() == "old content" for p in paths)
    assert not (root / refresh.DEFAULT_SNAPSHOT_PATH).with_suffix(".tmp").exists()


def test_github_pagination_flattens_all_pages(monkeypatch):
    calls = []

    def fake_command(args):
        calls.append(args)
        return b'[[{"title":"a"}],[{"title":"b"}]]'

    monkeypatch.setattr(refresh, "command", fake_command)
    assert refresh.GitHub("owner/repo").pages("issues?per_page=100") == [
        {"title": "a"},
        {"title": "b"},
    ]
    assert "--paginate" in calls[0] and "--slurp" in calls[0]


def test_command_timeout_and_errors_hide_response_bodies(monkeypatch):
    monkeypatch.setattr(
        refresh.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, b"", b"secret")
    )
    with pytest.raises(refresh.RefreshError) as error:
        refresh.command(["gh", "api"])
    assert "secret" not in str(error.value)

    def timeout(*args, **kwargs):
        assert kwargs["timeout"] == 60
        raise subprocess.TimeoutExpired(args[0], 60)

    monkeypatch.setattr(refresh.subprocess, "run", timeout)
    with pytest.raises(refresh.RefreshError, match="timed out"):
        refresh.command(["gh", "api"])


def test_workflow_validates_bot_updates_without_recursive_ci():
    workflow = yaml.safe_load(
        (
            Path(__file__).resolve().parents[2] / ".github/workflows/quality-dashboard-refresh.yml"
        ).read_text()
    )
    job = workflow["jobs"]["refresh"]
    assert job["permissions"] == {"contents": "write", "pull-requests": "write", "actions": "read"}
    scripts = "\n".join(step.get("run", "") for step in job["steps"])
    for required in (
        "pytest",
        "ruff check",
        "ruff format --check",
        "--check",
        "check_docs_site.py source",
        "gh pr create",
        "gh pr list",
    ):
        assert required in scripts
    assert "--force" not in scripts
    assert "--body-file" in scripts


@pytest.mark.parametrize(
    "scenario,status,deleted,summary",
    [
        ("policy", 23, True, "Allow GitHub Actions to create and approve pull requests"),
        ("transport", 23, True, "PR creation failed"),
        ("success", 0, False, ""),
        ("pending", 0, False, "already open"),
        ("clean", 0, False, "No dashboard changes"),
        ("commit-failure", 19, False, ""),
        ("push-failure", 19, False, ""),
        ("pr-exists", 23, False, "already has a PR"),
        ("lookup-failure", 23, False, "could not be safely completed"),
        ("delete-failure", 23, True, "Manual cleanup required"),
    ],
)
def test_workflow_proposal_cleanup(tmp_path, scenario, status, deleted, summary):
    workflow = yaml.safe_load(
        (
            Path(__file__).resolve().parents[2] / ".github/workflows/quality-dashboard-refresh.yml"
        ).read_text()
    )
    script = next(
        step["run"]
        for step in workflow["jobs"]["refresh"]["steps"]
        if step.get("name") == "Propose reviewed update"
    )
    # An isolated PATH and environment prevent real git, gh, or credentials being used.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = (
        f"#!{sys.executable}\n"
        + """\
import json
import os
import sys
from pathlib import Path

name = Path(sys.argv[0]).name
args = sys.argv[1:]
scenario = os.environ["SCENARIO"]
with open(os.environ["COMMAND_LOG"], "a") as log:
    log.write(json.dumps([name, *args]) + "\\n")
if name == "cat":
    sys.stdout.write(sys.stdin.read())
elif name == "git":
    if args[:3] == ["diff", "--cached", "--quiet"]:
        sys.exit(0 if scenario == "clean" else 1)
    if args[0] == "commit" and scenario == "commit-failure":
        sys.exit(19)
    if args == ["push", "origin", os.environ["REFRESH_BRANCH"]] and scenario == "push-failure":
        sys.exit(19)
    if "--delete" in args and scenario == "delete-failure":
        sys.exit(31)
elif name == "gh":
    if args[:2] == ["pr", "create"]:
        if scenario != "success":
            print("GitHub Actions is not permitted to create or approve pull requests."
                  if scenario == "policy" else "API transport failed", file=sys.stderr)
            sys.exit(23)
    elif args[:2] == ["pr", "list"]:
        if "--head" in args:
            if scenario == "lookup-failure":
                print("PR lookup failed", file=sys.stderr)
                sys.exit(29)
            print("1" if scenario == "pr-exists" else "0")
        else:
            print("1" if scenario == "pending" else "0")
    else:
        sys.exit("Unexpected gh command")
else:
    sys.exit("Unexpected executable")
"""
    )
    for name in ("git", "gh", "cat"):
        executable = bin_dir / name
        executable.write_text(stub)
        executable.chmod(0o755)
    summary_path = tmp_path / "summary"
    summary_path.touch()
    command_log = tmp_path / "commands.jsonl"
    branch = "automation/quality-dashboard-123-1"
    result = subprocess.run(
        ["/bin/bash", "--noprofile", "--norc", "-c", script],
        cwd=tmp_path,
        env={
            "PATH": str(bin_dir),
            "HOME": str(tmp_path),
            "REFRESH_BRANCH": branch,
            "GITHUB_STEP_SUMMARY": str(summary_path),
            "RUNNER_TEMP": str(tmp_path),
            "COMMAND_LOG": str(command_log),
            "SCENARIO": scenario,
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    commands = [json.loads(line) for line in command_log.read_text().splitlines()]
    assert result.returncode == status, result.stderr
    assert summary in summary_path.read_text()
    deletions = [cmd for cmd in commands if cmd[:2] == ["git", "push"] and "--delete" in cmd]
    assert deletions == ([["git", "push", "origin", "--delete", branch]] if deleted else [])
    if scenario == "success":
        assert ["git", "push", "origin", branch] in commands
        assert any(cmd[:3] == ["gh", "pr", "create"] for cmd in commands)
        assert not any("--head" in cmd for cmd in commands if cmd[:3] == ["gh", "pr", "list"])
    if scenario == "pending":
        assert not any(cmd[:2] == ["git", "switch"] for cmd in commands)
    if scenario in ("pending", "clean", "commit-failure", "push-failure"):
        assert not any(cmd[:3] == ["gh", "pr", "create"] for cmd in commands)
    if scenario in ("pending", "clean", "commit-failure"):
        assert not any(cmd[:2] == ["git", "push"] for cmd in commands)
    if status == 23:
        lookup = next(
            cmd for cmd in commands if cmd[:3] == ["gh", "pr", "list"] and "--head" in cmd
        )
        assert lookup[lookup.index("--head") + 1] == branch
        assert lookup[lookup.index("--state") + 1] == "all"
        assert "::error::PR creation failed" in result.stdout
        assert (
            "Allow GitHub Actions to create and approve pull requests" in summary_path.read_text()
        )
        assert branch in summary_path.read_text()
        assert (
            "not permitted" if scenario == "policy" else "API transport failed"
        ) in result.stderr
    if scenario == "lookup-failure":
        assert "PR lookup failed" in result.stderr
    if scenario == "delete-failure":
        assert f"::error::Manual cleanup required for branch {branch}" in result.stdout

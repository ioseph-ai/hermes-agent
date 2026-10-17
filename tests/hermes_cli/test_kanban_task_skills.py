"""Task-skills at card creation: joined strings and toolset names never reach a
card, and a well-formed name that cannot resolve in the ASSIGNEE profile's skill
trees is refused up front (either failure persists to ``tasks.skills`` and aborts
the dispatched worker at boot, on every retry)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_db_skills import _normalize_task_skills


def _plant_skill(skills_dir: Path, name: str) -> None:
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: test skill {name}\n---\n\n# {name}\n",
        encoding="utf-8",
    )


def _live_profile(home: Path, name: str) -> Path:
    """Profile dir carrying an identity marker (config.yaml) — live for dispatch."""
    profile = home / "profiles" / name
    profile.mkdir(parents=True, exist_ok=True)
    (profile / "config.yaml").write_text("{}\n", encoding="utf-8")
    return profile


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.mark.parametrize("joined, refusal", [
    ("github-pr-workflow,github-code-review", "cannot contain comma"),
    ("github-pr-workflow/github-code-review", "cannot contain '/'"),
    ("skills/github-code-review", "cannot contain '/'"),
    ("github-code-review/", "cannot contain '/'"),
])
def test_refuses_joined_strings(joined, refusal):
    """A joined multi-name string must never land in one argv slot: it persists to
    the card verbatim and the dispatched worker then fails to resolve it at boot."""
    with pytest.raises(ValueError, match=refusal):
        _normalize_task_skills([joined])


def test_slash_refusal_names_the_fix():
    """The '/' refusal must be actionable — name separate names, not a path form."""
    with pytest.raises(ValueError, match="pass separate names"):
        _normalize_task_skills(["github-pr-workflow/github-code-review"])


def test_rejects_toolset_names():
    with pytest.raises(ValueError, match="are toolset names, not skill name"):
        _normalize_task_skills(["web", "terminal"])


def test_strips_and_dedupes():
    assert _normalize_task_skills([" a ", "", "b", "a"]) == ["a", "b"]
    assert _normalize_task_skills(None) is None


def test_create_task_refuses_slash_joined_skills_before_persist(kanban_home):
    """The refusal must fire at the persistence boundary: no card may be created
    carrying a slash-joined pseudo-skill for a worker to crash on later."""
    with kbc.connect() as conn:
        with pytest.raises(ValueError, match="cannot contain '/'"):
            kb.create_task(conn, title="poisoned card", assignee="worker-a",
                           skills=["github-pr-workflow/github-code-review"])
        assert kb.list_tasks(conn) == []


# --- assignee-profile resolution: a card's skills must load where the card runs ------

def test_create_rejects_skill_missing_from_assignee_profile(kanban_home):
    """A well-formed name that resolves nowhere in the assignee's trees is refused at
    the persistence boundary, naming both the skill and the profile — previously it
    persisted fine and aborted the dispatched worker at boot on every retry."""
    worker = _live_profile(kanban_home, "worker")
    _plant_skill(worker / "skills", "bark")
    with kbc.connect() as conn:
        kb.create_task(conn, title="healthy card", assignee="worker", skills=["bark"])
        with pytest.raises(ValueError, match=r"'nope'.*assignee profile 'worker'"):
            kb.create_task(conn, title="poisoned card", assignee="worker", skills=["bark", "nope"])
        assert len(kb.list_tasks(conn)) == 1


def test_create_refuses_skill_installed_only_in_the_creating_profile(kanban_home):
    """Profiles are islands: a skill installed in the CREATING (default) profile must
    not pass validation for an assignee that lacks it — the exact cross-profile trap
    that crash-looped cards whose skills came from the orchestrator's own profile."""
    _plant_skill(kanban_home / "skills", "director-only-skill")
    _live_profile(kanban_home, "analyst")
    with kbc.connect() as conn:
        with pytest.raises(ValueError, match=r"director-only-skill.*profile 'analyst'"):
            kb.create_task(conn, title="cross-profile card", assignee="analyst",
                           skills=["director-only-skill"])


def test_unassigned_create_defers_and_plugin_qualified_names_skip(kanban_home):
    """No assignee yet → nothing to resolve against (the assign path validates later);
    a plugin-qualified name resolves from the worker's plugin registry, not these
    trees, so it is not existence-checked here."""
    _live_profile(kanban_home, "worker")
    with kbc.connect() as conn:
        kb.create_task(conn, title="unassigned", skills=["not-installed-anywhere"])
        kb.create_task(conn, title="plugin skill", assignee="worker", skills=["acme:helper"])


def test_assign_refuses_card_skills_the_new_profile_cannot_load(kanban_home):
    """Reassignment moves the card onto another profile's island: its skills must
    resolve THERE, at the moment of the handoff, not at the next dispatch."""
    _plant_skill(_live_profile(kanban_home, "worker") / "skills", "bark")
    _live_profile(kanban_home, "other")
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="roaming card", skills=["bark"])
        with pytest.raises(ValueError, match=r"'bark'.*assignee profile 'other'"):
            kb.assign_task(conn, task_id, "other")
        assert kb.assign_task(conn, task_id, "worker") is True


def test_assignee_external_dirs_are_searched(kanban_home):
    """The guard searches the trees the WORKER searches: ``skills.external_dirs``
    from the assignee's own config count, so shared read-only skill trees neither
    false-reject nor get bypassed."""
    worker = _live_profile(kanban_home, "worker")
    shared = kanban_home.parent / "shared-skills"
    _plant_skill(shared, "bark")
    (worker / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {shared}\n", encoding="utf-8")
    with kbc.connect() as conn:
        kb.create_task(conn, title="external skill card", assignee="worker", skills=["bark"])

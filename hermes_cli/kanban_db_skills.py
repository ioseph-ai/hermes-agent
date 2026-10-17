"""Task-skills normalization + assignee-profile resolution for the Kanban DB layer.

Split out of ``hermes_cli.kanban_db`` (FILE_LINES offset): the skills
normalizer, its toolset-name set, and the create/assign-time guard that every
task skill resolves in the ASSIGNEE profile's skill trees live here; the facade
imports them for ``create_task``/``assign_task``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

from toolsets import get_toolset_names

KNOWN_TOOLSET_NAMES = frozenset(name.casefold() for name in get_toolset_names())


def _normalize_task_skills(skills: Optional[Iterable[str]]) -> Optional[list[str]]:
    """Strip/dedupe a skills list. Commas and slashes are refused (a joined
    string must not land in one argv slot — a slash-joined phrase is two names,
    not a path-style skill reference); toolset names are rejected all at once
    because agents that confuse the two usually pass several."""
    if skills is None:
        return None
    cleaned: list[str] = []
    seen: set[str] = set()
    toolset_typos: list[str] = []
    for s in skills:
        if not s:
            continue
        name = str(s).strip()
        if not name:
            continue
        if "," in name:
            raise ValueError(
                f"skill name cannot contain comma: {name!r} "
                f"(pass a list of separate names instead of a comma-joined string)"
            )
        if "/" in name:
            raise ValueError(
                f"skill name cannot contain '/': {name!r} "
                "(a slash-joined phrase is not one skill — pass separate names, "
                "e.g. ['github-pr-workflow', 'github-code-review'])"
            )
        if name.casefold() in KNOWN_TOOLSET_NAMES:
            toolset_typos.append(name)
            continue
        if name in seen:
            continue
        seen.add(name)
        cleaned.append(name)
    if toolset_typos:
        quoted = ", ".join(repr(n) for n in toolset_typos)
        noun = "is a toolset name" if len(toolset_typos) == 1 else "are toolset names"
        raise ValueError(
            f"{quoted} {noun}, not skill name(s). "
            "Put toolsets in the assignee profile's `toolsets:` config "
            "instead of per-task skills. Skills are named skill bundles "
            "(e.g. `blogwatcher`, `github-code-review`); toolsets are runtime "
            "capabilities (e.g. `web`, `browser`, `terminal`)."
        )
    return cleaned


# --- assignee-profile resolution (create/assign-time guard) -------------------------
#
# A well-formed skill name that is installed only in the CREATING profile (or nowhere)
# used to abort the dispatched worker at boot — ``ValueError: kanban_skills_unavailable``
# in gateway.session_kanban — with zero messages, burning the card's whole retry budget.
# Profiles are islands: resolution must happen against the ASSIGNEE's trees, never the
# creating session's. Dispatch-time degradation (session_kanban) is the safety net for
# names that slip past this guard (pre-existing cards, DB edits, disabled skills).


def _assignee_home(assignee: str) -> Optional[Path]:
    """HERMES_HOME the *assignee* runs under, or None when that profile isn't live on
    this root (then nothing can be verified up front and degradation owns it). Uses the
    launch-root derivation from ``hermes_cli.profiles.resolve_profile_env`` so a
    profile-shaped ``HERMES_HOME`` (creating session already inside ``profiles/<x>``)
    still finds sibling profiles."""
    from hermes_cli.profiles import (
        _get_default_hermes_home, named_profile_is_live, normalize_profile_name, profile_root_for_env_home,
    )

    canon = normalize_profile_name(assignee)
    root = profile_root_for_env_home(os.environ.get("HERMES_HOME", ""), _get_default_hermes_home())
    if canon == "default":
        return root
    profile_dir = root / "profiles" / canon
    return profile_dir if named_profile_is_live(profile_dir) else None


def _assignee_skill_roots(home: Path) -> list[tuple[int, Path]]:
    """``(tier, dir)`` roots a worker as the assignee would search, minus the cwd
    project tiers — the workspace doesn't exist yet at create/assign time. Mirrors
    ``agent.skill_utils.get_skill_search_roots`` precedence (local, then
    ``skills.create_dir``, then ``skills.external_dirs``) but reads the ASSIGNEE's
    config, not the creating profile's: profiles are islands."""
    from agent.skill_utils import TIER_CREATE_DIR, TIER_EXTERNAL, TIER_LOCAL, _config_str_list, _expand_path

    def anchor(entry: str) -> Optional[Path]:
        text = str(entry).strip()
        if not text:
            return None
        p = _expand_path(text)
        return p if p.is_absolute() else home / p

    roots: list[tuple[int, Path]] = [(TIER_LOCAL, home / "skills")]
    skills_cfg: dict = {}
    try:
        from agent.skill_utils import yaml_load

        loaded = yaml_load((home / "config.yaml").read_text(encoding="utf-8-sig"))
        if isinstance(loaded, dict) and isinstance(loaded.get("skills"), dict):
            skills_cfg = loaded["skills"]
    except OSError:
        pass
    create_dir = anchor(skills_cfg.get("create_dir") or "")
    if create_dir is not None and create_dir != roots[0][1]:
        roots.append((TIER_CREATE_DIR, create_dir))
    for entry in _config_str_list(skills_cfg.get("external_dirs")):
        d = anchor(entry)
        if d is not None and d.is_dir() and all(d != existing for _t, existing in roots):
            roots.append((TIER_EXTERNAL, d))
    return roots


def validate_task_skills_for_assignee(skills: Optional[Iterable[str]], assignee: Optional[str]) -> None:
    """Refuse cards whose skills cannot load where they will run. Raises ``ValueError``
    naming every unresolvable skill and the assignee profile, so the mistake surfaces
    at the operator's fingertips instead of as a silent worker crash-loop.

    Existence check only: a skill the assignee has DISABLED, a plugin-qualified
    ``plugin:skill`` name (resolved from the worker's plugin registry, not these
    trees) and project-tier skills of a workspace that doesn't exist yet are left to
    the dispatch-time degradation, which reports and proceeds. No-op when the card
    carries no skills, is unassigned (validated on assign), or the assignee profile
    is not live on this root."""
    names = [n for n in (skills or []) if n and ":" not in n]
    if not names or not assignee:
        return
    home = _assignee_home(assignee)
    if home is None:
        return
    roots = _assignee_skill_roots(home)
    from tools.skills_tool import _locate_skill

    missing = [n for n in names if _locate_skill(n, None, roots)[2] is None]
    if missing:
        quoted = ", ".join(repr(n) for n in missing)
        searched = ", ".join(str(d) for _t, d in roots)
        raise ValueError(
            f"task skill(s) {quoted} do not resolve in assignee profile '{assignee}' "
            f"(searched: {searched}). Install the skill into that profile, or remove it "
            "from the card's skills — the dispatched worker can only preload skills "
            "installed in its own profile."
        )

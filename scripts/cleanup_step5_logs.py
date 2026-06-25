#!/usr/bin/env python3
"""Dry-run-first cleanup helper for Step 5 experiment logs.

This script never touches the formal Step 5B folder. By default it only prints
planned operations. Pass --execute to move archive candidates. Pass both
--execute and --delete to permanently delete delete candidates.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path


LOG_ROOT = Path("experiments/logs")
FORMAL_DIR = LOG_ROOT / "formal_step5b_chunked_20260621"

KEEP_FORMAL = {
    LOG_ROOT / "formal_step5b_chunked_20260621",
    LOG_ROOT / "formal_step5b_chunked_20260621.tar.gz",
}

KEEP_DIAGNOSTIC = {
    LOG_ROOT / "step5b_virtual_feedback_response",
    LOG_ROOT / "step5a_mutual_hil_smoke" / "20260620_182810_step5a_mutual_hil_smoke",
}

ARCHIVE_OPTIONAL = {
    LOG_ROOT / "step5b_mutual_model_comparison",
    LOG_ROOT / "step5_hil_model_comparison_preview",
    LOG_ROOT / "leader_ui_acceptance_stderr.log",
    LOG_ROOT / "step5b_no_pi_visual_check_stderr.log",
    LOG_ROOT / "step5a_mutual_hil_smoke" / "20260617_200414_step5a_mutual_hil_smoke",
    LOG_ROOT / "step5a_mutual_hil_smoke" / "20260619_224400_step5a_mutual_hil_smoke",
    LOG_ROOT / "step5a_mutual_hil_smoke" / "20260620_172859_step5a_mutual_hil_smoke",
    LOG_ROOT / "step5a_mutual_hil_smoke" / "20260620_175219_step5a_mutual_hil_smoke",
}

DELETE_CANDIDATES = {
    LOG_ROOT / "step5b_dryrun_validation",
    LOG_ROOT / "tmp_step5b_phase_check",
    LOG_ROOT / "tmp_step5a_debug_trace_schema_check",
    LOG_ROOT / "tmp_step5a_metric_schema_check",
    LOG_ROOT / "tmp_step5b_metric_schema_check",
    LOG_ROOT / "tmp_step5a_threaded_sanity",
    LOG_ROOT / "leader_ui_acceptance_stdout.log",
    LOG_ROOT / "step5b_no_pi_visual_check_stdout.log",
    LOG_ROOT / "step5a_mutual_hil_smoke" / "20260619_224500_step5a_mutual_hil_smoke",
    LOG_ROOT / "step5a_mutual_hil_smoke" / "20260620_172851_step5a_mutual_hil_smoke",
}


def _default_archive_dir() -> Path:
    return LOG_ROOT / f"_archive_step5_{datetime.now().strftime('%Y%m%d')}"


def _size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _relative_destination(path: Path, archive_dir: Path) -> Path:
    try:
        rel = path.relative_to(LOG_ROOT)
    except ValueError:
        rel = Path(path.name)
    return archive_dir / rel


def _assert_safe_candidate(path: Path) -> None:
    resolved = path.resolve()
    root = LOG_ROOT.resolve()
    formal = FORMAL_DIR.resolve()
    if resolved == formal or formal in resolved.parents:
        raise RuntimeError(f"Refusing to operate on formal evidence: {path}")
    if root not in resolved.parents and resolved != root:
        raise RuntimeError(f"Refusing to operate outside log root: {path}")


def _move_to_archive(path: Path, archive_dir: Path, execute: bool) -> None:
    if not path.exists():
        print(f"SKIP missing: {path}")
        return
    _assert_safe_candidate(path)
    dest = _relative_destination(path, archive_dir)
    print(f"ARCHIVE {'EXEC' if execute else 'DRY '} {path} -> {dest} ({_size_bytes(path)} bytes)")
    if not execute:
        return
    if dest.exists():
        raise RuntimeError(f"Archive destination already exists: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(dest))


def _delete_candidate(path: Path, execute: bool) -> None:
    if not path.exists():
        print(f"SKIP missing: {path}")
        return
    _assert_safe_candidate(path)
    print(f"DELETE {'EXEC' if execute else 'DRY '} {path} ({_size_bytes(path)} bytes)")
    if not execute:
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run-first Step 5 log cleanup helper.")
    parser.add_argument("--archive-dir", type=Path, default=_default_archive_dir())
    parser.add_argument("--execute", action="store_true", help="Actually perform planned operations")
    parser.add_argument("--delete", action="store_true", help="Delete delete-candidate items; requires --execute")
    args = parser.parse_args()

    if not FORMAL_DIR.exists():
        raise SystemExit(
            f"ERROR: required formal folder is missing: {FORMAL_DIR}. "
            "Refusing cleanup."
        )
    if args.delete and not args.execute:
        raise SystemExit("ERROR: --delete requires --execute.")

    print("Step 5 log cleanup")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY-RUN'}")
    print(f"Delete candidates: {'DELETE' if args.delete else 'ARCHIVE'}")
    print(f"Archive dir: {args.archive_dir}")
    print()

    print("Protected formal evidence:")
    for path in sorted(KEEP_FORMAL):
        status = "present" if path.exists() else "missing"
        print(f"  KEEP_FORMAL {status}: {path}")
    print()

    print("Kept diagnostics:")
    for path in sorted(KEEP_DIAGNOSTIC):
        status = "present" if path.exists() else "missing"
        print(f"  KEEP_DIAGNOSTIC {status}: {path}")
    print()

    print("Archive optional:")
    for path in sorted(ARCHIVE_OPTIONAL):
        _move_to_archive(path, args.archive_dir, args.execute)
    print()

    print("Delete candidates:")
    for path in sorted(DELETE_CANDIDATES):
        if args.delete:
            _delete_candidate(path, args.execute)
        else:
            _move_to_archive(path, args.archive_dir, args.execute)

    print()
    if not args.execute:
        print("Dry-run complete. Re-run with --execute to move candidates to archive.")
        print("Use --execute --delete only after approving permanent deletion.")
    else:
        print("Cleanup operations complete.")


if __name__ == "__main__":
    main()

# Step 5 Log Cleanup Plan

Generated: 2026-06-21

This is a dry-run cleanup report only. No experiment data has been moved,
deleted, renamed, or modified.

## Summary

The Step 5 log tree contains one formal result folder, several diagnostic
folders, an earlier non-chunked batch, and several dry-run/schema/temp folders.
The formal folder is small and must be kept. The largest space users are
Step 5A smoke dry-run subfolders inside `step5a_mutual_hil_smoke`, not the
formal Step 5B results.

Important warning: `formal_step5b_chunked_20260621` currently contains the
formal run nested under
`formal_step5b_chunked_20260621/experiments/logs/step5b_mutual_model_comparison/formal_step5b_chunked_20260621/`.
This report does not propose reorganising it because the folder is formal
evidence.

## Folder Inventory

| Folder | Recommendation | Size | Files | Newest Modified | Aggregate | Summary | Metadata | Trial Metrics | Debug Traces | Reason |
|---|---:|---:|---:|---|---|---|---|---:|---|---|
| `formal_step5b_chunked_20260621` | KEEP_FORMAL | 4.483 MB | 215 | 2026-06-21 19:57:53 | yes | yes | batch/run/chunk | 18 | `kuramoto_debug.csv` x9 | Formal chunked Step 5B evidence. Never delete. |
| `step5a_mutual_hil_smoke` | KEEP_DIAGNOSTIC | 104.953 MB | 31 | 2026-06-20 18:28:10 | no | no | per-trial | 5 | none | Keep parent folder; review subfolders individually. |
| `step5b_virtual_feedback_response` | KEEP_DIAGNOSTIC | 0.025 MB | 8 | 2026-06-21 19:13:22 | no | no | no | 0 | none | Synthetic feedback/API response diagnostics, including Kuramoto server check. |
| `step5b_mutual_model_comparison` | ARCHIVE_OPTIONAL | 3.037 MB | 203 | 2026-06-21 18:58:52 | yes | yes | batch | 18 | none | Earlier non-chunked batch with known frontend/polling issue; useful historical evidence but not formal. |
| `step5_hil_model_comparison_preview` | ARCHIVE_OPTIONAL | 0.371 MB | 8 | 2026-06-17 19:50:22 | no | no | no | 0 | none | Step 5 preview figures/tables; not formal HIL evidence. |
| `step5b_dryrun_validation` | DELETE_CANDIDATE | 0.044 MB | 19 | 2026-06-21 19:33:41 | yes | yes | batch/run/chunk | 1 | none | Created for runner dry-run/seed validation; disposable after review. |
| `tmp_step5a_debug_trace_schema_check` | DELETE_CANDIDATE | 0.061 MB | 16 | 2026-06-20 22:27:35 | no | no | per-trial | 2 | `event_trace.csv` x2 | Temporary schema/debug trace check. Keep only if you want examples of event-trace schema. |
| `tmp_step5a_metric_schema_check` | DELETE_CANDIDATE | 0.025 MB | 7 | 2026-06-20 22:14:15 | no | no | per-trial | 1 | none | Temporary metric schema check. |
| `tmp_step5a_threaded_sanity` | DELETE_CANDIDATE | 0.023 MB | 7 | 2026-06-20 18:48:41 | no | no | per-trial | 1 | none | Temporary camera-threading sanity output. |
| `tmp_step5b_metric_schema_check` | DELETE_CANDIDATE | 0.001 MB | 1 | 2026-06-20 22:14:08 | no | no | batch | 0 | none | Temporary Step 5B metric schema check. |
| `tmp_step5b_phase_check` | DELETE_CANDIDATE | 0.001 MB | 1 | 2026-06-20 23:01:12 | no | no | batch | 0 | none | Temporary Step 5B phase schedule check. |

Top-level Step 5 related files, not folders:

| File | Recommendation | Size | Reason |
|---|---:|---:|---|
| `formal_step5b_chunked_20260621.tar.gz` | KEEP_FORMAL | 2.268 MB | Archive copy of formal folder; keep until formal evidence is backed up elsewhere. |
| `leader_ui_acceptance_stderr.log` | ARCHIVE_OPTIONAL | 0.013 MB | Acceptance/debug terminal output. |
| `leader_ui_acceptance_stdout.log` | DELETE_CANDIDATE | 0 MB | Empty output log. |
| `step5b_no_pi_visual_check_stderr.log` | ARCHIVE_OPTIONAL | 0.008 MB | Diagnostic terminal output. |
| `step5b_no_pi_visual_check_stdout.log` | DELETE_CANDIDATE | 0 MB | Empty output log. |

## Step 5A Smoke Subfolder Review

Policy says not to delete the whole `step5a_mutual_hil_smoke` folder. In this
workspace, all visible Step 5A smoke subfolders are dry-run or empty; I did not
find feedback-ON full EAPF validation, random-phase smoke validation, 1.2 Hz
runaway fix validation, `suspicious_window.csv`, or important debug traces in
this parent folder.

| Subfolder | Recommendation | Size | Key Flags | Reason |
|---|---:|---:|---|---|
| `20260617_200414_step5a_mutual_hil_smoke` | ARCHIVE_OPTIONAL | 52.890 MB | dry-run, EAPF, V=2.0/P=1.5 | Large dry-run output; archive or delete only after manual confirmation. |
| `20260619_224400_step5a_mutual_hil_smoke` | ARCHIVE_OPTIONAL | 51.222 MB | dry-run, feedback OFF, V=2.0/P=1.5 | Large dry-run output; archive or delete only after manual confirmation. |
| `20260619_224500_step5a_mutual_hil_smoke` | DELETE_CANDIDATE | 0 MB | empty | Empty temporary folder. |
| `20260620_172851_step5a_mutual_hil_smoke` | DELETE_CANDIDATE | 0 MB | empty | Empty temporary folder. |
| `20260620_172859_step5a_mutual_hil_smoke` | ARCHIVE_OPTIONAL | 0.140 MB | dry-run, feedback OFF | Older small dry-run smoke. |
| `20260620_175219_step5a_mutual_hil_smoke` | ARCHIVE_OPTIONAL | 0.183 MB | dry-run, Kuramoto, feedback OFF | Older small dry-run smoke. |
| `20260620_182810_step5a_mutual_hil_smoke` | KEEP_DIAGNOSTIC | 0.518 MB | dry-run, EAPF, detect-only, no Pi feedback post | Best visible Step 5A smoke diagnostic in this folder; keep unless superseded elsewhere. |

Manual uncertainty: the requested categories mention feedback-OFF normal HIL,
feedback-ON no-pi-feedback-post, feedback-ON full EAPF, 1.2 Hz runaway debug,
and random-phase smoke validation. I did not find those as clearly labelled
non-dry-run folders under `step5a_mutual_hil_smoke`; do not delete this parent
until you confirm whether those runs are stored elsewhere.

## Space Estimate

Conservative delete candidates:

- Top-level temporary/dry-run folders: about 0.156 MB.
- Empty Step 5A smoke subfolders: about 0 MB.
- Empty top-level stdout logs: about 0 MB.

Archive-optional candidates:

- `step5b_mutual_model_comparison`: about 3.037 MB.
- `step5_hil_model_comparison_preview`: about 0.371 MB.
- Large Step 5A smoke dry-run subfolders: about 104.435 MB.
- Small diagnostic logs: about 0.021 MB.

Estimated immediate deletion space is tiny unless you also approve archiving or
deleting the large Step 5A dry-run subfolders.

## Dry-Run Cleanup Commands

Inspect planned operations only:

```powershell
python scripts/cleanup_step5_logs.py
```

Dry-run with explicit archive destination:

```powershell
python scripts/cleanup_step5_logs.py --archive-dir experiments/logs/_archive_step5_20260621
```

Archive optional and delete-candidate items by moving them to the archive
directory, without permanent deletion:

```powershell
python scripts/cleanup_step5_logs.py --execute --archive-dir experiments/logs/_archive_step5_20260621
```

Permanently delete delete-candidate items and archive optional items:

```powershell
python scripts/cleanup_step5_logs.py --execute --delete --archive-dir experiments/logs/_archive_step5_20260621
```

## Safety Rules in the Script

- Default mode is dry-run.
- `--execute` is required for any move or delete.
- `--delete` is additionally required for permanent deletion.
- `formal_step5b_chunked_20260621` is never moved or deleted.
- The script refuses to run if `formal_step5b_chunked_20260621` is missing.
- The script prints every planned or executed operation.


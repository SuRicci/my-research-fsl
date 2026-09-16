# September 16 incremental research archive

Repository: https://github.com/SuRicci/my-research-fsl
Release tag: quest-012-pause-20260916

The earlier September14 release retains pre-resumption history. This release retains all currently reachable research Git refs plus supplemental ignored/uncommitted research files from every live worktree. Runtime/login state and conversation logs are excluded from upload and remain local where needed.

1. Download both archive files and SHA256SUMS. Verify with `shasum -a 256 -c SHA256SUMS`.
2. Run `git clone research-history.bundle restored-research`. All saved refs are listed in `refs.json`. Check out the desired branch (the latest evidence is `run/ordered-crop-correspondence-20260916`).
3. Extract `research-supplement.tar.gz` into a separate empty folder. `supplement-manifest.json` maps each original workspace-relative path to an archived member and SHA256. Files marked supplemental capture untracked or locally modified research evidence; apply only to their matching worktree/ref after inspecting provenance.
4. For source features already retrieved from September14, use that release and its RESTORE.md. The cleanup receipt lists which redundant workspaces/downloads were removed. Absolute local paths in old scripts need remapping on another machine.

Research remains paused. This is a reproducible evidence checkpoint, not a submission-ready manuscript.

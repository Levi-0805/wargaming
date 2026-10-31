# CSSIM source of truth

The user requires CSSIM Python changes to be made in the desktop Git repository first, pushed to GitHub, and only then copied here.

- Source repository on this machine: `C:\Users\aaa\Desktop\wargaming`.
- Edit source under that repository's `PythonCode` directory.
- `<CssimRoot>\Client\Data\AlgData\PythonCode` and `<CssimRoot>\Server\Data\AlgData\PythonCode` are deployment copies. This machine uses `C:\CSSIM`; `E:\CSSIM` remains supported. The desktop `CSSIM20260915` directory is installation media.
- Read the source repository's `AGENTS.md`. Review changes, validate, commit and push there, then use its `tools\publish.ps1` workflow to deploy to both copies.
- If the platform generates new code, import it into the repository before editing. Preserve existing uncommitted work.
- If validation or GitHub push fails, do not deploy. Never claim files are backed up on GitHub until the remote commit is verified.
- Do not force push or delete unrelated platform files. Renamed/deleted source files need a reviewed deployment cleanup and backup.

- Match records must include terminal destruction score plus `Reward`, every round, and the source commit. Historical timestamps alone do not establish which code ran; offline decisions are not new simulated outcomes.
- For each code/model version, accumulate ten completed games across client interruptions. Preserve terminal games and retry only the remaining budget. Summarize results and the modification plan before publishing the next version; do not require ten consecutive games or mix versions.
- Continuous learning writes a `.txt` brief after each ten-game round and a `.txt` overview after every four rounds in the desktop `迭代报告` folder. Keep machine-readable provenance in the source repository, and logs/checkpoints out of Git.
- Consume an undelivered `logs/learning/report-ready.json` score signal before treating training as a quiet wait; candidate training must not hide completed scores. Review/publish only after training ends. The local observer is diagnostic only and its images require scale/time validation.
- Review bounded public-observation tactical search and offline IQL separately; retain champion/rollback evidence. Only real match returns evaluate the three-calendar-day search, and offline agreement/replay checks never establish score gains. Publish and deploy before releasing the next round.

This guide is versioned at `C:\Users\aaa\Desktop\wargaming\tools\CSSIM-AGENTS.md`.

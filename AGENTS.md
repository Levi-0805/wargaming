# CSSIM development and publishing rules

The user requires all CSSIM algorithm and Python framework changes to be versioned here and pushed to GitHub before deployment.

- Canonical source: this repository's `PythonCode` (currently `C:\Users\aaa\Desktop\wargaming\PythonCode`).
- Deployment copies: `<CssimRoot>\Client\Data\AlgData\PythonCode` and `<CssimRoot>\Server\Data\AlgData\PythonCode`. This machine uses `C:\CSSIM`; `E:\CSSIM` remains supported. The desktop `CSSIM20260915` folder contains installation media, not the runtime source.
- Make source changes in this repository first. Never use the deployment copies as the development source.
- When CSSIM creates or downloads a new team algorithm, inspect repository status and import those files into the repository before editing them. Preserve any uncommitted work; do not blindly overwrite it with an import.
- Review the diff, run the appropriate checks, commit, push to the configured GitHub remote, verify the remote commit, then deploy. Use `tools\publish.ps1 -Message "describe the change"` for the supported workflow.
- If checks or push fail, stop deployment and report the error accurately. A local commit alone is not a GitHub backup. Do not force push to resolve divergence.
- Keep credentials, virtual environments, runtime logs and generated caches out of Git.
- Deploy by copying repository files to both CSSIM copies; preserve unrelated platform-generated files. File deletions/renames require explicit review of the affected deployed paths and a backup, because the copy workflow does not delete destination-only files.
- The tracked `tools\CSSIM-AGENTS.md` is the corresponding workspace guide for `<CssimRoot>\AGENTS.md`; keep it consistent when the workflow changes.
- Record total score as terminal destruction score plus `Reward`, with every round and the source commit. Never infer a historical running version from timestamps alone or present replayed decisions as new match outcomes.
- Count ten completed games cumulatively for each code/model version across client interruptions. Preserve valid terminal games, run only the remaining budget after restart, and summarize results and the modification plan before publishing the next version. Do not require ten consecutive games or mix code/model versions.
- Continuous learning writes a `.txt` brief after each ten-game round and a `.txt` overview after every four rounds in the desktop `迭代报告` folder. Keep machine-readable provenance in the repository; runtime logs and local checkpoints remain ignored.
- The three-calendar-day tactical search uses observed match returns and bounded public-observation profiles. Keep champion/rollback evidence, distinguish IQL held-out data from gradient training, and never equate replay checks or offline agreement with live score gains. A completed round requires review/publish/deploy before release of the next version.


# CSSIM source of truth

The user requires CSSIM Python changes to be made in the desktop Git repository first, pushed to GitHub, and only then copied here.

- Source repository: `C:\Users\admin\Desktop\wargaming`.
- Edit source under `C:\Users\admin\Desktop\wargaming\PythonCode`.
- `E:\CSSIM\Client\Data\AlgData\PythonCode` and `E:\CSSIM\Server\Data\AlgData\PythonCode` are deployment copies, not development folders.
- Read the source repository's `AGENTS.md`. Review changes, validate, commit and push there, then use its `tools\publish.ps1` workflow to deploy to both copies.
- If the platform generates new code, import it into the repository before editing. Preserve existing uncommitted work.
- If validation or GitHub push fails, do not deploy. Never claim files are backed up on GitHub until the remote commit is verified.
- Do not force push or delete unrelated platform files. Renamed/deleted source files need a reviewed deployment cleanup and backup.

This guide is versioned at `C:\Users\admin\Desktop\wargaming\tools\CSSIM-AGENTS.md`.

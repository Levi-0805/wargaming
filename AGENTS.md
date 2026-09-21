# CSSIM development and publishing rules

The user requires all CSSIM algorithm and Python framework changes to be versioned here and pushed to GitHub before deployment.

- Canonical source: `C:\Users\admin\Desktop\wargaming\PythonCode`.
- Deployment copies: `E:\CSSIM\Client\Data\AlgData\PythonCode` and `E:\CSSIM\Server\Data\AlgData\PythonCode`.
- Make source changes in this repository first. Never use the deployment copies as the development source.
- When CSSIM creates or downloads a new team algorithm, inspect repository status and import those files into the repository before editing them. Preserve any uncommitted work; do not blindly overwrite it with an import.
- Review the diff, run the appropriate checks, commit, push to the configured GitHub remote, verify the remote commit, then deploy. Use `tools\publish.ps1 -Message "describe the change"` for the supported workflow.
- If checks or push fail, stop deployment and report the error accurately. A local commit alone is not a GitHub backup. Do not force push to resolve divergence.
- Keep credentials, virtual environments, runtime logs and generated caches out of Git.
- Deploy by copying repository files to both CSSIM copies; preserve unrelated platform-generated files. File deletions/renames require explicit review of the affected deployed paths and a backup, because the copy workflow does not delete destination-only files.
- The tracked `tools\CSSIM-AGENTS.md` is the corresponding workspace guide installed at `E:\CSSIM\AGENTS.md`; keep it consistent when the workflow changes.


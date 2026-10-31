# Ten-game learning workflow

Run `tools/install_learning_shortcuts.ps1` once. **CSSIM Training** starts a hidden,
idempotent worker. **Stop CSSIM Training** requests STOP and preserves completed
games. Keep the computer on, logged in, and its desktop available to CSSIM.

`logs/learning/state.json` is the small control record. Each published code/model
version collects exactly ten completed games across native client interruptions.
Failed exits remain recorded. Three attempts with no new completed games stop for
attention. No source changes or Git commits while games are running.

The desktop folder **迭代报告** receives a readable `.txt` brief
with all ten destruction scores + Rewards, exact source/model/profile hashes,
actual exit codes, observed defence groups, low-score fraction and tail statistics.
Every fourth round also produces a four-round report. Reports are written before
the next version is staged. Compact reviews/results are pushed to GitHub; large
protocols and local checkpoints stay out of Git.

After collection, offline IQL training runs on verified complete sessions. The last
two sessions are validation and test, not gradient training. This is ground target
selection, not learned navigation or end-to-end strategic reinforcement learning.
The newest data can become training data in later rounds as the source split moves;
each manifest records the exact roles. No offline metric is presented as match gain.

At `awaiting_review`, Codex reads the local report queue. A reviewer checks the
candidate, chooses one experiment, then runs:

```
python tools/cssim_learning.py prepare-next
# Or --use-candidate for an isolated IQL trial with low/high integration replays.
# Review the diff and generated results/learning/*-next-version.json.
tools/publish.ps1 -Message "Describe reviewed next iteration" -CssimRoot C:\CSSIM
python tools/cssim_learning.py release --commit FULL_VERIFIED_GITHUB_COMMIT
```

Release verifies clean source, GitHub HEAD, both deployments and staged artifact
hashes again. The same worker then starts the next ten games. No arbitrary code is
generated or executed from reports. STOP blocks release. Interrupted coordinators
require recovery rather than silently losing their completed-game budget.

Three-calendar-day search uses Asia/Shanghai dates, temperatures 800/300/100 and a
reheat at each new three-day block. One defence class/one bounded parameter changes
per tactical proposal. Actual ten-game returns determine acceptance with a lower
tail guard; the best evaluated checkpoint is retained separately. Small unpaired
samples and changing defenders cannot establish causal superiority. The 5000 goal
is assessed using low-score frequency and tail performance, not mean alone.

The known HQ front U-turn, HQ-first rule, local vehicle fire eligibility and finite
formation waiting limits remain in force. Defence classes use public capture-box
counts plus recent visible contacts; unknown fields never imply an empty objective.
The stored label is an observed class, not access to hidden enemy deployment.

Codex's scheduled chat can consume pending reports while the desktop app is running.
With Codex closed the independent worker still collects, trains and writes the
current round, then waits at review. There is no supported automatic attachment
upload here, no promise of an immediate app-start trigger, and no offline AI code
review. Reopen Codex for the next scheduled check, or manually send the report.

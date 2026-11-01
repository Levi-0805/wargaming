# Ten-game learning workflow

Run `tools/install_learning_shortcuts.ps1` once. **CSSIM Training** starts a hidden,
idempotent worker. **Stop CSSIM Training** requests STOP and preserves completed
games. Keep the computer on, logged in, and its desktop available to CSSIM.

`logs/learning/state.json` is the small control record. Each published code/model
version collects exactly ten completed games across native client interruptions.
Failed exits remain recorded. Three attempts with no new completed games stop for
attention. No source changes or Git commits while games are running.

Each round uses its own desktop folder **迭代报告/第NNNN轮/** for its `.txt` brief,
observation notes, modification plans and current status. It receives a readable brief
with all ten destruction scores + Rewards, exact source/model/profile hashes,
actual exit codes, observed defence groups, low-score fraction and tail statistics.
Every fourth round also produces a four-round report in that ending round's folder. Reports are written before
the next version is staged. Compact reviews/results are pushed to GitHub; large
protocols and local checkpoints stay out of Git.

After collection, offline IQL training runs on verified complete sessions. The last
two sessions are validation and test, not gradient training. This is ground target
selection, not learned navigation or end-to-end strategic reinforcement learning.
The newest data can become training data in later rounds as the source split moves;
each manifest records the exact roles. No offline metric is presented as match gain.

A cumulative session can contain an extra score packet without a confirmed Done.
The verified report's episode selection must pass through dataset extraction and
manifest binding: retain its valid completed games, exclude unconfirmed episodes,
and preserve the full original protocol hash. A score-only extra episode must not
enter training merely because the same file also contains verified games.

Before candidate training, `logs/learning/report-ready.json` points to an early
ten-game summary. The scheduled review checks undelivered score signals even while
the worker is training, reports the scores once, then marks delivery. This is a
local inbox consumed by the existing chat; it does not upload an attachment or
instantly wake a closed app. Publishing remains blocked until training ends.

The learner starts at most one versioned local observer for the first ten rounds.
It writes evidence only, uses local Windows OCR for viewer focus, respects both
STOP files, and makes no AI/API calls. `VIEW_PAUSE` disables camera interaction
without stopping unit/event recording. Its images still require HUD/scale review;
a unit-focus request never proves that the image is a useful close-up.

At `awaiting_review`, Codex reads the local report queue. A reviewer checks the
candidate, chooses one experiment, then runs:

```
python tools/cssim_learning.py prepare-next
# Or --use-candidate for an isolated IQL trial with low/high integration replays.
# Or --restore-profiles reviewed.json for a bounded profile rollback with unchanged weights.
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

The reviewed HQ garrison experiment keeps the current IQL weights and defence
profiles fixed. With a known residual HQ occupant, controllable soldiers fill the
existing basic reserve required for departure; dependent dogs can still fight as provisional guards but do not replace that soldier reserve. A tied full
bar preserves eligible previous guards without asserting ownership or unlocking
a new departure. Clear-point and explicit remnant fallback rules are unchanged.
The experiment includes two related guard branches; a ten-game return cannot
identify their separate causal effects. Log guard control links, HQ reversals,
terminal capture Rewards and repeated return journeys alongside destruction score.

`tools/diagnostics/audit_completed_failure_types.py` audits the archived first four
round reports and verifies their complete protocol hashes and selected episodes.
`tools/diagnostics/audit_hq_guard_contract.py` compares explicitly supplied old and
new route files against frozen public states. Pass `--baseline-route`, `--source`,
`--episode`, `--model` and `--out`; the canonical route is the default candidate.
The old route can be exported from the recorded baseline Git commit. Runtime
protocols and model checkpoints must exist locally. These diagnostics do not
control units or generate new match outcomes.

Historical baseline reproduction selects `policy_baseline: "6cb5a79"` in the
published policy template. Its exact Git-exported tactical files remain isolated
under `Algorithm/Champion6cb5a79`; each file is checked against snapshot.json
before policy instantiation. The current policy remains available as `current`.
This restores the old tactic and original IQL settings without reverting the
runtime, exit handling, result provenance or report pipeline. Effective configs
and batch manifests bind the baseline selector; actual reset metadata must match.
Defence profiles are not applied inside the historical tactic. Any score is a
fresh reproduction attempt, not a guaranteed repeat of the historical 4496 mean.

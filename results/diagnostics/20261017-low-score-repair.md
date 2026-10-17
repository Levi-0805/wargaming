# Low-score investigation and objective-progress repair

Source: `C:\CSSIM\Client\Data\AlgData\PythonCode\logs\20261017_011401\protocol.jsonl`.
Frozen size: 1,229,543,582 bytes. SHA256: `4ad0f34ced26948539510a9b10fffaa6b5631084565aeeb5a685403c9b4c4bf3`.
Baseline: `5e789df47cd86a401940f98ba23a5f3281ae3725`.
Twenty completed rounds: mean 2833, median 3055, range 970–4220. Four below 2000; two above 4000.
Total means terminal destruction score plus capture Reward.

## Reproduced evidence

Public-observation replays of rounds 4, 10, 11, 16, 18, 19 reproduced all 105,250 recorded unit commands with zero mismatches before editing. This establishes the baseline behavior without inferring a version from timestamps.

- Round 11: 1570 = 770 + 800. At step 175 B had 8 red and 0 blue occupants, full capture, but no reserved guards because the selected target was A. By step 225 B had 10 blue occupants and was lost. Only A contributed terminal capture reward.
- Round 16: 970 = 970 + 0. The HQ foothold at step 125 still contained 6 blue units. Capture-capable red units fell from 31 at step 125 to 3 at step 175 and 1 at step 250. The full progress bar did not establish a cleared, durable HQ holding.
- Round 18: 1150 = 1150 + 0. At step 175 B had 14 red and 0 blue occupants, full capture, no reserved guards. At step 225 B was lost to 9 blue occupants. No capture-capable red units survived at step 250.
- Round 19: 1700 = 1700 + 0. A was selected at step 83, but its approach crossed B, which had 16 blue occupants at step 100. Destination-only threat scoring treated empty A as cheap. Capture-capable red strength fell from 36 at step 150 to 3 at step 200 and 0 at step 250.
- Round 10: 4220 = 2620 + 1600. The reward was A+B, not HQ. At the end, the remaining mobile soldier 5041 was still assigned an old A route point (-34023,-34923,5763), although the selected objective was HQ. Four other capturers were reserved as guards.
- Round 4: 4200 = 2600 + 1600, also A+B. At the end all three surviving capturers were guards, leaving no mobile capturer for HQ.

## Changes

1. Add public threat along crossed objective boxes to route cost, using the existing known-map corridor and a representative mobile-unit position. An empty A no longer erases the cost of crossing defended B.
2. Recognize incidentally captured objectives after four consecutive empty-enemy observations with enough physically present capturers. Reserve guards and retain the current attack destination.
3. After adaptive replanning, advance mobile-unit cursors to the current objective instead of completing obsolete earlier legs.
4. With at most six surviving capturers and another objective pending, reserve one guard per holding only after eight consecutive clear observations. Nearby visible threats restore the normal reserve.

Regression coverage checks crossed B versus already-passed B, incidental holding confirmation and observation gaps, old-route cancellation, and conditional small-force guard release. The former expectation that empty A is preferable behind defended B is intentionally replaced.

## Validation and limits

323 tests passed, including compilation and the existing UAV/perception regressions. Candidate replays use historical observations and cannot predict new combat outcomes: changed actions would change future observations. The HQ attrition in round 16 remains a combat risk; these changes do not prove that every round will exceed 4000. Fresh matches against both defensive patterns are required to measure the lower tail, capture rewards and survival.

Candidate replay completed on all six selected rounds without errors. Under round 19's recorded observations it retains HQ instead of selecting the A route through defended B. At step 175 it assigns four B guards in round 11 and two in round 18 (both were zero in the baseline). Round 10's final holdings use two guards instead of four, releasing two additional capturers; round 4's three survivors no longer all remain assigned as guards. Candidate command differences total 6389; they are behavior changes, not score improvements. Round 16 also retains HQ instead of first redirecting to A and back.

Reproduce the selected public-view audit with:

`python tools/audit_objective_progress.py <protocol.jsonl> --episodes 4 10 11 16 18 19 --out logs/audit.json`

Raw replay snapshots stay in ignored logs. The compact comparison is recorded alongside this report; it distinguishes original scores from candidate decisions.

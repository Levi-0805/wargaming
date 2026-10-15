# Rule intervention rollback

This change withdraws three decision overrides from the default rule policy.
It does not claim a new match score or improvement in survival. Previous source
baseline: `145cc7fd0398eb8051b531d464df2cb00982bd4a`.

## Evidence and changes

The completed `20261015_200211` run scored 1640, 1210, 870, 1190 and 930 for red;
all five final capture rewards were zero. Its 1,757 public observations were
processed with the three legacy switches enabled. All 87,600 recorded red
action strings matched, with no missing commands or mismatches. The frame count
includes terminal observations which need not have a following action packet.

- Disable `enable_shared_contact_maneuver`: 10,183 legacy engagement decisions
  were based only on team reports, with no local contact. Those decisions mixed
  speculative attack requests and repeated short repositioning. Reports no
  longer invoke that branch or replace objective waypoints with an enemy's
  coordinates. Public information remains available to aircraft and objective
  threat/occupancy assessment. Local engagements and obstruction recovery remain.
- Disable `enable_preservation_pause`: the old policy returned 109 pause actions
  after its force-strength threshold was crossed. That rule did not establish
  that the stopped position offered protection. It no longer issues actions by
  default. This is separate from the existing bounded opening rendezvous.
- Disable `enable_forced_entry_under_contact`: the old entrance policy returned
  120 navigation actions despite an armed unit having a local ground contact.
  The entrance route now yields for that contact. Its cursor survives the
  interruption so navigation can resume afterwards. This does not remove the
  entrance geometry or prove that the retained route is safe.

With all three disabled, these intervention counts are zero on the same public
observations. Local attack requests change from 527 to 624. These are requests,
not confirmed shots, hits or kills. UAV commands match the legacy combination
on every compared frame. Different ground movement in a live game can still
change later UAV decisions through their existing support conditions.

The single-rule variants are included in `20261015-intervention-comparison.json`.
They isolate decision changes on a fixed recording; they cannot establish the
score contribution of a rule, because the simulator does not respond to the
new decisions during replay. The much smaller pause and entrance counts also
mean neither alone explains the entire collapse.

## Verification and reproducibility

`tools/compare_rule_interventions.py` reads a bounded protocol snapshot, builds
only the red team's public view, serializes all actions, and compares the legacy,
three single-rule removals, and combined default variants. It never connects to
the platform. `--require-recorded-match` rejects an incompatible baseline.

```powershell
& 'C:\CSSIM\Client\Data\AlgData\PythonEnv\python.exe' tools\compare_rule_interventions.py `
  C:\CSSIM\Client\Data\AlgData\PythonCode\logs\20261015_200211\protocol.jsonl `
  --output results\diagnostics\20261015-intervention-comparison.json --require-recorded-match
```

Regression coverage includes default platform configuration, independent switch
selection, rejection of string booleans, shared reports not diverting ground
navigation, contact interrupts/resumption for all four ground types, and
casualties not activating the retired pause. Existing legacy shared-probe tests
now explicitly enable the retired option; they do not define the default policy.
The full test suite passed: 275 tests.

The earlier `20261015_150032` five-round run was also checked in full (1,933
observations, 96,650 serialized actions per variant). All three retired
interventions are absent in the default variant there too, including the
observations from the recorded 3610-point round. See
`20261016-intervention-cross-run.json`. That run used an older version, so its
5,025 action mismatches against the latest legacy combination are expected;
it is a cross-run validity check, not an exact reproduction or a new score.

## Live comparison

All three options default to `false`; the current TaskConfig has no overriding
section. An experiment can set one option to `true` in the section named
`TeamAlg.FsLutk7wD2RkNgbs.Algorithm.RLAgentAlgorithm.py->AlgorithmConfig`.
All three `true` reproduce the previous combination. Configuration changes must
follow the same commit, push, remote verification and deployment workflow.

Keep the map, opponent, round count, step limit and timing unchanged between
variants. Record the commit, switches and complete round results. Compare the
mean, median, lowest score, proportion above 3000 and final capture reward;
do not select a variant using its best game alone. The configured Python seed
is not proof of identical simulator randomness. Repeated live comparisons are
still required before claiming a stable score improvement.

Connection repair, ownership/occupancy checks, garrison bounds, motion recovery,
opening assembly, vehicle/UAV parameters and scoring are not retuned in this
change. Source files are retained for reversible comparisons, avoiding deployed
file deletion. Newly started Python sessions load the new defaults; an already
running interpreter may retain its previous imported classes.

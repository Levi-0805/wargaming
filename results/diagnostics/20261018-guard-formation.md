# Guard vehicle fire, three-way scouting and phased objective capture

## Latest recorded scores

Run `20261017_165218` contains nine completed red rounds: 2050, 3210, 3420,
3400, 3340, 2170, 2880, 3220, 1530. Mean 2802.22, median 3210, minimum
1530, maximum 3420; one below 2000 and none reaching 4000. The preceding
13-round batch averaged 3358.46. Six completed rounds received only the HQ's
1600 capture reward, one received A's 800, and two received no capture reward.
Round 10 stopped at step 76 without settlement, with a cancelled gRPC connection.

Score definition: terminal destruction score plus Reward. The log does not
bind its running source to a commit; deployment timing alone is not version
proof. Source SHA256:
`cd075b0ed3e2e416d1613680c3e8a6b7c8ab257c479e8ee6060566716ca67b12`.
Frozen source size: 599,266,248 bytes.

## Requested behavior

- Five UAVs receive stable HQ/A/B scouting assignments (2/2/1) at the start.
  They launch ahead of the infantry toward actual observation destinations.
  Each objective retains an observer; only spare aircraft replace lost groups.
  Three consecutive, connected, genuinely in-perception observations with a
  known public occupant count establish a survey. Evidence expires after 12
  steps. Ground objective changes do not pull all scouts to the same point.
- The actual roster has four wheeled vehicles and two Lynx vehicles. Three
  vehicles lead, including both chassis types. Three are initially reserved
  for A/B (one/two); when HQ is cleared, living lead vehicles can replenish
  each reserve to two while retaining at least one mobile lead vehicle.
- Vehicles patrol the existing road/entrance coordinates with supported
  `Guard` commands. Explicit fire first aims using directional Guard, then
  targets an actual UID. It requires that vehicle's current local perception,
  a living ground target, usable ammunition, connectivity and a distance check.
  Shared sightings and air targets cannot trigger explicit vehicle fire.
- Two unconfirmed attack requests trigger an eight-step cooldown; observing
  both ammunition consumption and target HP loss allows useful fire to continue.
  Patrol can resume during cooldown. This is not a claim of two bullets: UE
  determines how many rounds a single attack request consumes.
- Infantry/dogs retain the opening rendezvous, while vehicles go first. Early
  foot arrivals hold reached positions; rear units keep advancing. Reserves
  wait behind the first objective rather than spending all vehicles on HQ.
  Each objective gets a reconnaissance/vehicle/foot handoff. Actual subsequent
  ammunition consumption, not merely issuing an attack, confirms a vehicle
  volley. Cleared surveyed objectives also allow handoff after vehicles arrive.
  Local combat/damage, casualties and bounded waits prevent permanent deadlock.
  Reconnaissance waiting is capped at 18 steps and handoff at 28; absent UAVs
  do not cause a vehicle reconnaissance wait. These are fallback deadlines,
  not claims that reconnaissance or combat has succeeded by those times.

## Range evidence and limits

All six vehicles advertise `fireRange=1000`, as do unarmed commanders. The
Oct08 investigation already showed that treating this generic value as a
9.5–10 m firing gate suppressed almost all fire. Training material gives
vehicle perception of 200 m but no verified weapon maximum; scenario weapon
JSON also lacks a maximum range. Perception is not substituted for weapon range.

The latest log's explicitly attributed vehicle kill events include preceding
separations of 102.9–111.5 m, 117.2–134 m and 118.9–126.4 m. They demonstrate
long-distance kills, but do not establish an exact firing instant, line of
sight or maximum range. For the known generic 1000 placeholder only, the new
policy uses a 150 m bounded trial envelope, covering these observations while
remaining below the 200 m perception radius. Other finite positive reported
ranges are honored; missing/invalid range disables explicit fire.

The protocol has Guard direction/path and NormalAttacking by UID. It does not
provide directional normal fire. Guard's physical travel and automatic weapon
behavior are engine-controlled and require live-match confirmation.

## Validation

Regression tests cover range and local perception, no ammunition, burst limits,
Guard serialization, navigation recovery, three simultaneous scout destinations,
survey freshness, fixed assignments, reserves, bounded waits and actual-ammo
handoff. Existing tests were updated where the requested behavior intentionally
changes Moving into Guard patrol or adds one aiming frame before firing.

Offline validation uses latest rounds 1/3/6/9 and historical round 11 of
`20261016_110326` (the original 5630-point match). Every action is serialized
through the real protocol; explicit vehicle fire is checked against local
eligibility and UAV fire against actual local perception. Companion JSON
records source hashes, candidate file hashes, roster assignments and counts.
The final candidate completed 2,293 recorded frames across those five rounds
with zero invalid explicit vehicle-fire or UAV-local-contact checks. Its
vehicle commands included 44 explicit attack requests across 206 frames with
eligible local targets; other commands include patrol, aiming and cooldown.
These counts do not measure the engine's automatic fire or actual ammunition
savings under Guard.
Historical scores remain historical: fixed-observation replay cannot predict
the movement, ammunition consumption, survival or scores caused by new actions.

Changes are published only through the repository's verify → commit → push →
remote verification → backed-up client/server deployment workflow. New matches
must load a fresh algorithm process. No new live score or stable 4000+ outcome
is claimed by these checks.

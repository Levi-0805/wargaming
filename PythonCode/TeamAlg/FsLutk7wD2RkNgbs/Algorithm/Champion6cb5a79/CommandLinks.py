from __future__ import annotations

import logging
import math

from cssim.environment import TeamState
from cssim.protocol import Action, ChangeParentAction


logger = logging.getLogger(__name__)


class CommandLinks:
    """Repair command links using observed parent UIDs, not assumed RPC success.

    UE coordinates are centimetres. Range is only an upper bound: rSVD1 remains
    the authority for obstruction-related outages inside that range.
    """

    SOLDIER = "BP_BaseSoldier_C"
    MAX_RANGE = 100_000.0
    RETRY_STEPS = 4
    REJECT_STEPS = 12

    def __init__(self):
        self._episode: int | None = None
        self._step: int | None = None
        self._pending: dict[int, tuple[int, int]] = {}
        self._rejected: dict[tuple[int, int], int] = {}
        self._warnings: dict[int, int] = {}

    def begin(self, state: TeamState) -> None:
        if state.episode != self._episode or (
            self._step is not None and state.step < self._step
        ):
            self._pending.clear()
            self._rejected.clear()
            self._warnings.clear()
        self._episode, self._step = state.episode, state.step

    @staticmethod
    def distance(left, right) -> float:
        return math.dist(tuple(left.position), tuple(right.position))

    @classmethod
    def eligible(cls, state: TeamState, child, parent) -> bool:
        if parent is None or parent.uid == child.uid:
            return False
        if not parent.alive or parent.team != child.team:
            return False
        commander_uids = {item.uid for item in state.commanders}
        if parent.entity_type != cls.SOLDIER and parent.uid not in commander_uids:
            return False
        if parent.communication_ok() is False:
            return False
        # Only allow soldier/commander roots, and reject a candidate whose
        # existing ancestor chain leads back to the child or a dead node.
        node, visited = parent, set()
        while node is not None:
            if node.uid == child.uid or node.uid in visited:
                return False
            visited.add(node.uid)
            uid = node.commander_uid()
            if uid is None or uid == node.uid:
                return True
            node = state.entity_by_uid(uid)
            if node is not None and (not node.alive or node.team != child.team):
                return False
        return True

    def reassign(self, state: TeamState, child) -> ChangeParentAction | None:
        self.begin(state)
        if not child.alive or child.entity_type == self.SOLDIER:
            return None
        current = state.current_commander(child)
        current_valid = self.eligible(state, child, current)
        disconnected = child.communication_ok() is False
        current_distance = self.distance(child, current) if current_valid else math.inf
        emergency = not current_valid or disconnected or current_distance >= self.MAX_RANGE

        pending = self._pending.get(child.uid)
        if pending is not None:
            parent_uid, sent_step = pending
            requested = state.entity_by_uid(parent_uid)
            if current_valid and current.uid == parent_uid and not disconnected:
                del self._pending[child.uid]
            elif self.eligible(state, child, requested) and self.distance(child, requested) < self.MAX_RANGE:
                if state.step - sent_step < self.RETRY_STEPS:
                    return None
                # The next frame did not confirm a usable link. Give another
                # nearby node a chance instead of consuming every action slot.
                self._rejected[(child.uid, parent_uid)] = state.step + self.REJECT_STEPS
                del self._pending[child.uid]
            else:
                # A requested parent died: do not wait for the retry timer.
                del self._pending[child.uid]

        candidates = [
            parent for parent in (*state.agents, *state.commanders)
            if self.eligible(state, child, parent)
            and self.distance(child, parent) < self.MAX_RANGE
            and self._rejected.get((child.uid, parent.uid), -1) <= state.step
        ]
        if disconnected:
            alternatives = [item for item in candidates if item.uid != child.commander_uid()]
            if alternatives:
                candidates = alternatives
        parent = min(candidates, key=lambda item: (self.distance(child, item), item.uid), default=None)
        if parent is None:
            if emergency and state.step - self._warnings.get(child.uid, -100) >= 20:
                logger.warning("指挥链无法恢复: child=%d step=%d 当前无1公里内存活候选上级", child.uid, state.step)
                self._warnings[child.uid] = state.step
            return None

        distance = self.distance(child, parent)
        # Healthy early reparenting correlates with complete motion loss in
        # the 230451 replay. Do not disturb a working chain just to shorten it.
        if not emergency:
            return None
        self._pending[child.uid] = (parent.uid, state.step)
        logger.info("动态改挂上级: child=%d old=%s parent=%d distance_m=%.1f offline=%s step=%d",
                    child.uid, child.commander_uid(), parent.uid, distance / 100, disconnected, state.step)
        return ChangeParentAction(parent, child)

    def constrain(self, state: TeamState, child, action: Action) -> Action:
        """Recover only on reported link loss; do not turn healthy units back.

        Never clamp a bomb's coordinates onto another location. A disconnected
        aircraft first repairs its link before trying to bomb.
        """
        if child.entity_type == self.SOLDIER or not child.alive:
            return action
        if action.command == "SelfDestruct" and child.communication_ok() is False:
            return Action.guard_position(tuple(child.position))
        if action.command != "Moving":
            return action
        parent = state.current_commander(child)
        if not self.eligible(state, child, parent):
            return action
        if child.communication_ok() is not False:
            return action
        if action.points is not None and math.dist(tuple(parent.position), action.points[:3]) <= self.distance(child, parent):
            return action
        # Only when a handoff has not succeeded and the link is down, move
        # back toward the confirmed parent. Direction control avoids creating
        # arbitrary altitude points by clipping a sphere around that parent.
        delta = parent.position - child.position
        axes = ("X", "Y", "Z") if child.entity_type in {"BP_Base_UAV_C", "BP_Helicopter_C"} else ("X", "Y")
        direction = "".join(
            ("+" if delta[i] > 0 else "-") + axis
            for i, axis in enumerate(axes) if abs(delta[i]) > 400
        )
        return Action.move(direction) if direction else Action.guard_position(tuple(child.position))

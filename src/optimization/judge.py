"""Stage 3 judge subsystem for GEPA+."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.optimization.proposer import PromptProposal


@dataclass(slots=True)
class JudgeResult:
    """Scored proposal with explanation."""

    proposal: PromptProposal
    score: float
    explanation: str


class Judge:
    """Rubric-based deterministic judge for prompt proposals."""

    def score(
        self,
        proposal: PromptProposal,
        *,
        held_out_signals: Iterable[str] | None = None,
    ) -> JudgeResult:
        signals = list(held_out_signals or [])
        score = 0.0
        reasons: list[str] = []

        if "contract" in proposal.edited_prompt.lower():
            score += 1.0
            reasons.append("preserves contract language")

        if "minimal edits" in proposal.edited_prompt.lower() or "minimal" in proposal.delta.lower():
            score += 0.75
            reasons.append("favors minimal change")

        if "schema" in proposal.edited_prompt.lower():
            score += 0.75
            reasons.append("makes schema explicit")

        if "ambiguous" in proposal.edited_prompt.lower():
            score += 0.5
            reasons.append("guards against ambiguity")

        if signals:
            overlap = sum(1 for signal in signals if signal.lower() in proposal.edited_prompt.lower())
            score += overlap * 0.5
            if overlap:
                reasons.append(f"matches {overlap} held-out signal(s)")

        if score == 0.0:
            reasons.append("no rubric match; proposal is weak")

        return JudgeResult(proposal=proposal, score=score, explanation="; ".join(reasons))

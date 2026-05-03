"""Stage 3 merger for GEPA+ proposals."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.optimization.judge import JudgeResult
from src.optimization.proposer import PromptProposal
from src.optimization.verifier import VerificationResult


@dataclass(slots=True)
class MergeResult:
    """Merged prompt delta and audit trail."""

    module_name: str
    merged_prompt: str
    kept: list[PromptProposal] = field(default_factory=list)
    rejected: list[PromptProposal] = field(default_factory=list)
    deferred: list[PromptProposal] = field(default_factory=list)
    delta_summary: str = ""


class Merger:
    """Contract-preserving merger for optimized prompt proposals."""

    def merge(
        self,
        module_name: str,
        base_prompt: str,
        judged: list[JudgeResult],
        verified: dict[str, VerificationResult],
        *,
        topn: int = 2,
    ) -> MergeResult:
        ranked = sorted(judged, key=lambda item: (-item.score, item.proposal.origin))
        kept: list[PromptProposal] = []
        rejected: list[PromptProposal] = []
        deferred: list[PromptProposal] = []
        merged_lines = [line.strip() for line in base_prompt.splitlines() if line.strip()]
        seen_additions: set[str] = set()

        for item in ranked:
            proposal = item.proposal
            verification = verified.get(proposal.origin)
            if verification is None:
                deferred.append(proposal)
                continue
            if not verification.ok:
                rejected.append(proposal)
                continue
            added_line = proposal.metadata.get("added_line")
            if not isinstance(added_line, str):
                rejected.append(proposal)
                continue
            if added_line in seen_additions:
                rejected.append(proposal)
                continue
            kept.append(proposal)
            seen_additions.add(added_line)
            merged_lines.append(added_line)
            if len(kept) >= topn:
                break

        for item in ranked:
            proposal = item.proposal
            if proposal not in kept and proposal not in rejected and proposal not in deferred:
                deferred.append(proposal)

        merged_lines.append(f"Merge contract: preserve {module_name} schema and minimal prompt drift.")
        merged_prompt = "\n".join(merged_lines)
        delta_summary = "; ".join(proposal.delta for proposal in kept) if kept else "no changes kept"
        return MergeResult(
            module_name=module_name,
            merged_prompt=merged_prompt,
            kept=kept,
            rejected=rejected,
            deferred=deferred,
            delta_summary=delta_summary,
        )

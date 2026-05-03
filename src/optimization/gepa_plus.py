"""Stage 3 GEPA+ optimization loop."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Iterable

from src.optimization.judge import Judge, JudgeResult
from src.optimization.merger import MergeResult, Merger
from src.optimization.proposer import ProposalGenerator, PromptProposal
from src.optimization.verifier import VerificationResult, Verifier


@dataclass(slots=True)
class OptimizationRound:
    """Audit record for a single optimization round."""

    module_name: str
    proposal_count: int
    judged: list[JudgeResult] = field(default_factory=list)
    verified: list[VerificationResult] = field(default_factory=list)
    merge_result: MergeResult | None = None
    elapsed_ms: float = 0.0
    tokens_consumed: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class OptimizationResult:
    """Final output of a GEPA+ optimization run."""

    module_name: str
    base_prompt: str
    optimized_prompt: str
    rounds: list[OptimizationRound] = field(default_factory=list)
    best_score: float = 0.0
    final_delta: str = ""


class GEPAPlus:
    """Deterministic Stage 3 prompt optimizer."""

    def __init__(
        self,
        *,
        proposer: ProposalGenerator | None = None,
        judge: Judge | None = None,
        verifier: Verifier | None = None,
        merger: Merger | None = None,
        seed: int = 0,
    ) -> None:
        self.proposer = proposer or ProposalGenerator(seed=seed)
        self.judge = judge or Judge()
        self.verifier = verifier or Verifier()
        self.merger = merger or Merger()
        self.seed = seed

    def optimize(
        self,
        module_name: str,
        base_prompt: str,
        *,
        held_out_signals: Iterable[str] | None = None,
        rounds: int = 1,
        proposals_per_round: int = 4,
        topn: int = 2,
        max_tokens: int | None = None,
    ) -> OptimizationResult:
        current_prompt = base_prompt
        audit: list[OptimizationRound] = []
        best_score = float("-inf")
        final_delta = ""
        signals = list(held_out_signals or [])

        for round_index in range(rounds):
            started = time.perf_counter()
            proposals = self.proposer.generate(module_name, current_prompt, k=proposals_per_round)
            judged = [self.judge.score(proposal, held_out_signals=signals) for proposal in proposals]
            verified_map = {proposal.origin: self.verifier.verify(proposal) for proposal in proposals}
            merge_result = self.merger.merge(module_name, current_prompt, judged, verified_map, topn=topn)
            elapsed_ms = (time.perf_counter() - started) * 1000
            tokens_consumed = self._estimate_tokens(current_prompt, proposals)

            round_result = OptimizationRound(
                module_name=module_name,
                proposal_count=len(proposals),
                judged=judged,
                verified=list(verified_map.values()),
                merge_result=merge_result,
                elapsed_ms=elapsed_ms,
                tokens_consumed=tokens_consumed,
            )
            audit.append(round_result)

            if merge_result.kept:
                current_prompt = merge_result.merged_prompt
                round_best = max(item.score for item in judged if item.proposal in merge_result.kept)
                if round_best > best_score:
                    best_score = round_best
                    final_delta = merge_result.delta_summary
            else:
                round_result.notes.append("no verified proposals were promoted")

            if max_tokens is not None and tokens_consumed > max_tokens:
                round_result.notes.append("token budget exceeded; stopping optimization")
                break

            if not merge_result.kept:
                break

        if best_score == float("-inf"):
            best_score = 0.0

        return OptimizationResult(
            module_name=module_name,
            base_prompt=base_prompt,
            optimized_prompt=current_prompt,
            rounds=audit,
            best_score=best_score,
            final_delta=final_delta,
        )

    def _estimate_tokens(self, base_prompt: str, proposals: list[PromptProposal]) -> int:
        total_chars = len(base_prompt) + sum(len(proposal.edited_prompt) for proposal in proposals)
        return max(1, total_chars // 4)

from __future__ import annotations

from src.optimization import GEPAPlus, Judge, Merger, ProposalGenerator, Verifier
from src.prompts.seed_prompts import ATOMIZER_PROMPT, EXECUTOR_WRITE_PROMPT


def test_proposal_generator_emits_delta_candidates() -> None:
    generator = ProposalGenerator(seed=7)
    proposals = generator.generate("atomizer", ATOMIZER_PROMPT, k=3)

    assert len(proposals) == 3
    assert all(proposal.delta.startswith("+ ") for proposal in proposals)
    assert len({proposal.origin for proposal in proposals}) == 3


def test_judge_and_verifier_gate_promotions() -> None:
    generator = ProposalGenerator(seed=11)
    proposal = generator.generate("write", EXECUTOR_WRITE_PROMPT, k=1)[0]
    judged = Judge().score(proposal, held_out_signals=["schema", "contract"])
    verified = Verifier().verify(proposal)

    assert judged.score > 0
    assert verified.ok is True
    assert "contract" in judged.explanation


def test_merger_deduplicates_and_preserves_top_candidates() -> None:
    generator = ProposalGenerator(seed=13)
    proposals = generator.generate("planner", ATOMIZER_PROMPT, k=4)
    judge = Judge()
    verifier = Verifier()
    judged = [judge.score(proposal) for proposal in proposals]
    verified = {proposal.origin: verifier.verify(proposal) for proposal in proposals}

    merge = Merger().merge("planner", ATOMIZER_PROMPT, judged, verified, topn=2)

    assert len(merge.kept) <= 2
    assert "schema" in merge.merged_prompt.lower() or "contract" in merge.merged_prompt.lower()
    assert merge.delta_summary


def test_gepa_plus_optimizes_prompt_with_audit_trail() -> None:
    optimizer = GEPAPlus(seed=21)

    result = optimizer.optimize(
        "write",
        EXECUTOR_WRITE_PROMPT,
        held_out_signals=["schema", "contract", "minimal"],
        rounds=2,
        proposals_per_round=4,
        topn=2,
    )

    assert result.optimized_prompt != ""
    assert len(result.rounds) >= 1
    assert result.best_score >= 0
    assert result.final_delta
    assert any(round_result.merge_result is not None for round_result in result.rounds)

"""Stage 3 proposal generation for GEPA+."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Iterable


@dataclass(slots=True)
class PromptProposal:
    """A single prompt delta candidate."""

    module_name: str
    base_prompt: str
    edited_prompt: str
    delta: str
    origin: str
    temperature: float
    seed: int
    metadata: dict[str, object] = field(default_factory=dict)


class ProposalGenerator:
    """Generate multiple delta-style prompt candidates in parallel."""

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = seed

    def generate(
        self,
        module_name: str,
        base_prompt: str,
        *,
        k: int = 4,
        temperatures: Iterable[float] | None = None,
    ) -> list[PromptProposal]:
        temps = list(temperatures or (0.2, 0.4, 0.6, 0.8))
        proposals: list[PromptProposal] = []
        for index in range(k):
            temperature = temps[index % len(temps)]
            proposal_seed = self.seed + index
            rng = random.Random(proposal_seed)
            proposals.append(self._build_proposal(module_name, base_prompt, temperature=temperature, rng=rng, seed=proposal_seed))
        return proposals

    def _build_proposal(
        self,
        module_name: str,
        base_prompt: str,
        *,
        temperature: float,
        rng: random.Random,
        seed: int,
    ) -> PromptProposal:
        lines = [line.strip() for line in base_prompt.splitlines() if line.strip()]
        additions = self._candidate_additions(module_name)
        chosen = additions[rng.randrange(len(additions))]
        edited_lines = list(lines)
        edited_lines.append(chosen)
        edited_lines.append(f"Verification note: preserve {module_name} schema and output contract.")
        edited_prompt = "\n".join(edited_lines)
        delta = f"+ {chosen}"
        origin = f"seed={seed};temp={temperature:.2f};module={module_name}"
        return PromptProposal(
            module_name=module_name,
            base_prompt=base_prompt,
            edited_prompt=edited_prompt,
            delta=delta,
            origin=origin,
            temperature=temperature,
            seed=seed,
            metadata={"added_line": chosen},
        )

    def _candidate_additions(self, module_name: str) -> list[str]:
        if module_name.lower() == "write":
            return [
                "Always preserve the write interface contract and document the output structure.",
                "Prefer minimal edits that keep write behavior stable and the structure explicit.",
                "State the required output shape explicitly for write and keep the prose organized.",
                "Favor concise, MECE instructions for write while preserving structure.",
                "Reject edits that introduce ambiguity into write or weaken organization.",
            ]
        return [
            f"Always preserve the {module_name} interface contract.",
            f"Prefer minimal edits that keep {module_name} behavior stable.",
            f"State the required output shape explicitly for {module_name}.",
            f"Favor concise, MECE instructions for {module_name}.",
            f"Reject edits that introduce ambiguity into {module_name}.",
        ]

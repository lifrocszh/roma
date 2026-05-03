"""Stage 3 verifier subsystem for GEPA+."""

from __future__ import annotations

from dataclasses import dataclass

from src.optimization.proposer import PromptProposal


@dataclass(slots=True)
class VerificationResult:
    """Fast structural verification outcome."""

    proposal: PromptProposal
    ok: bool
    reason: str


class Verifier:
    """Fast checks for schema conformance and prompt safety."""

    REQUIRED_MARKERS = {
        "atomizer": ["EXECUTE", "PLAN", "atomic", "non-atomic"],
        "planner": ["MECE", "dependency", "subtask"],
        "retrieve": ["evidence", "search", "context"],
        "think": ["reason", "conclusion", "context"],
        "write": ["write", "prose", "structure"],
        "code": ["code", "sandbox", "programmatic"],
        "aggregator": ["merge", "compress", "conflict"],
    }

    def verify(self, proposal: PromptProposal) -> VerificationResult:
        text = proposal.edited_prompt.lower()
        markers = self.REQUIRED_MARKERS.get(proposal.module_name.lower(), [])
        missing = [marker for marker in markers if marker.lower() not in text]
        if proposal.module_name.lower() == "write" and "structure" not in text and "organize" not in text:
            missing.append("structure")
        if missing:
            return VerificationResult(
                proposal=proposal,
                ok=False,
                reason=f"missing required markers: {', '.join(missing)}",
            )

        if "schema" not in text and "contract" not in text:
            return VerificationResult(
                proposal=proposal,
                ok=False,
                reason="proposal does not explicitly preserve interface contract",
            )

        return VerificationResult(proposal=proposal, ok=True, reason="passed structural checks")

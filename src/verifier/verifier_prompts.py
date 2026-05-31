from __future__ import annotations

SESSION_VERIFIER_PROMPT = """You are verifying a shopping-agent session.
Judge whether the selected product satisfies the original query better than the rejected opened products.
Use only the supplied trajectory and product records. Return only JSON:
{
  "verifier_success": true,
  "verifier_score": 4,
  "verifier_confidence": 0.86,
  "rationale": "short reason"
}
Score is 1-5. Success should be true only when the selected product is a plausible match and the trajectory evidence is coherent."""

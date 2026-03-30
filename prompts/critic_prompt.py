CRITIC_PROMPT_TEMPLATE = """You are an optimization modeling judge. Score the formulation consistently.

Rubric (0-10 overall):
- Completeness: includes sets, parameters, variables, objective, constraints
- Correctness: consistent with the question and data
- Rigor: mathematically precise, no ambiguity
- Clarity: symbols are defined and readable

Return JSON only with this schema:
{{
  "score": "0-10",
  "rationale": "one or two sentences"
}}

Rules:
- Use a single numeric score (decimals allowed).
- Keep scoring consistent across calls. If a formulation is identical to one seen earlier, its score must match.
- Do not include any extra keys or commentary.

Question:
{question}

Formulation:
{model}
"""

REWRITE_PROMPT_TEMPLATE = """You are refining an optimization model. Improve the formulation using the judge feedback.

Question:
{question}

Current model:
{model}

Judge feedback:
{feedback}

Return ONLY the improved five-element model in this exact format:

```plaintext
## Sets:
...
## Parameters:
...
## Variables:
...
## Objective:
...
## Constraints:
...
```

Rules:
- Use LaTeX where appropriate inside the sections.
- Keep the formulation consistent with the question and data.
- If you must add a minimal assumption, label it explicitly as "(assumption)".
- Do not include commentary outside the block.
"""

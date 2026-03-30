ques_description = "The following is an optimization problem. "

five_description_complex = """In mathematics, any optimization problem can be modeled as the following expression $\\\\min_{\\\\boldsymbol{x} \\\\in \\\\mathcal{X}} f(\\\\boldsymbol{x}), {\\\\rm s.t.} g(\\\\boldsymbol{x}) \\\\leq b$, where $\\\\boldsymbol{x} = (x_1, x_2, \\\\ldots, x_d)^\\\\top$ is the $d$-dimensional decision variable, $\\\\mathcal{X} \\\\subset \\\\mathbb{R}^d$ is the feasible domain, $f: \\\\mathcal{X} \\\\rightarrow \\\\mathbb{R}$ is the objective function and the goal is to find the minima of $f$, $g(\\\\boldsymbol{x}) \\\\leq b$ is the constraint of $\\\\boldsymbol{x}$.
The above definition can be mapped to a five-element consisting of ``Variables, Objective, Constraints, Sets, Parameters\\'\\'. Variables indicates what $\\\\boldsymbol{x}$ is, Objective describes the form of the objective function $f(\\\\boldsymbol{x})$, and Constraints indicates the constraints $g(\\\\boldsymbol{x})$ and $\\\\mathcal{X}$. These three can abstract the optimization problem. Sets describes and explains the subscripts of the vectors or matrices in them. Parameters must supplement their specific values, like "c_s = 20 ".
"""

ques_description_five = """You need to write the corresponding five-element model based on the problem description and information provided.
The problem description is as follows:
"""

five_description_code = "The following is the five-element model of an optimization problem: "

five_suffix_with_source = """Please write the corresponding five-element model with sources. Please use LaTeX and ``` plain text environment to complete the following template to model the above optimization problem into five elements:

```
## Sets Content:
[You need to fill in]
## Sets Source:
[You need to fill in]

## Parameters Content:
[You need to fill in]
## Parameters Source:
[You need to fill in]

## Variables Content:
[You need to fill in]
## Variables Source:
[You need to fill in]

## Objective Content:
[You need to fill in]
## Objective Source:
[You need to fill in]

## Constraints Content:
[You need to fill in]
## Constraints Source:
[You need to fill in]

## Math Model Content:
[You need to fill in]
## Math Model Source:
[You need to fill in]
```

Rules:
- Do not output JSON.
- "Content" should be the five-element content (use LaTeX where appropriate).
- "Source" must point to where the content comes from.
  - If derived from documents, include document name, page number, and the specific supporting text.
  - If derived only from the natural language question, use "NL" as the source.
- If a component is not specified, use an empty string for both Content and Source.
- "Math Model Content" should summarize the mathematical model based on the five elements, mainly the Objective expression and explanations of variables appearing in the Objective.
"""


bound_symbol = """
```
"""

generate_system_info = "You are an expert in the field of operations and optimization. You need to complete some optimization problem modeling tasks."

id_gt = """Identify the optimal value corresponding to the solution of the problem in the following string. Your output only needs to be a numeric value. If you encounter an exception, please output "NAN".
Here is the problem:
"""

demo_structure_requirements = """

Additional Structure Requirements:
1. Parameters Content must contain only fixed values, bounds, limits, coefficients, capacities, budgets, demands, times, or other known constants.
2. Do not place full inequalities, equalities, or complete constraint expressions in Parameters Content.
3. If an expression is a modeling restriction with =, <=, or >=, place it in Constraints Content rather than Parameters Content.
4. Do not use abstract placeholder constraints such as g(x), h(x), g(x,u)=0, or h(x,u)<=0 unless no more explicit algebraic form can be written from the evidence.
5. Prefer explicit algebraic constraints written directly with the listed variables and parameters.
6. Every variable listed in Variables Content must appear in Objective Content or Constraints Content.
7. For each variable in Variables Content, indicate whether it is continuous, binary, or integer whenever possible.
For Content fields and Source Explanation, do not embed mathematical notation inside long prose sentences.
Prefer one item per line in one of the following forms only:
1. symbol: explanation
2. symbol = value: explanation
3. pure formula line
"""


def Q2F_with_source(ques):
    ques = bound_symbol + ques + bound_symbol
    return five_description_complex + ques_description_five + demo_structure_requirements + ques + demo_source_requirements + five_suffix_with_source



# =========================
# Demo-specific additions v2
# Keep the original schema/style, only add extra requirements
# =========================


demo_five_suffix_with_source = """Please write the corresponding five-element model with sources. Please use LaTeX and ``` plain text environment to complete the following template to model the above optimization problem into five elements:

```
## Sets Content:
[You need to fill in]
## Sets Source:
[You need to fill in]

## Parameters Content:
[You need to fill in]
## Parameters Source:
[You need to fill in]

## Variables Content:
[You need to fill in]
## Variables Source:
[You need to fill in]

## Objective Content:
[You need to fill in]
## Objective Source:
[You need to fill in]

## Constraints Content:
[You need to fill in]
## Constraints Source:
[You need to fill in]

## Math Model Content:
[You need to fill in]
## Math Model Source:
[You need to fill in]
```

Rules:
- Do not output JSON.
- "Content" should be the five-element content (use LaTeX only).
- "Source" must point to where the content comes from.
  - If derived from documents, include document name, page number, and the specific supporting text.
  - If derived only from the natural language question, use "NL" as the source.
- If a component is not specified, use an empty string for both Content and Source.
- "Math Model Content"  present only the final mathematical formulation based on the five elements.It must be a pure LaTeX mathematical model without explanatory prose, variable notes, or trailing comments.
"""

demo_five_description_executable = """You need to construct a source-aware five-element optimization model for a compact and executable demo instance.

The goal is not to produce a broad abstract optimization framework, but to produce a small, closed, and interpretable formulation that can be directly translated into optimization code.

Definitions:
- Variables should include only decision variables or state variables that actually appear in the final formulation.
- Objective should be an explicit mathematical expression using the chosen variables.
- Constraints should be explicit algebraic constraints, not unnamed abstract function families, whenever possible.
- Sets should include only the index sets that are actually used in the final formulation.
- Parameters should include concrete numerical data whenever possible; if the document does not provide enough numerical values, create a small reasonable demo instance and write those values explicitly.

Important:
- Do not introduce symbols that are not used later.
- Do not keep the model at a purely symbolic or theoretical level if a compact executable demo formulation can be produced.
"""

demo_source_requirements = """

Additional Source Requirements:
- Every component must have a Source field.
- Use exactly one of the following two source types.

Source Type A: natural-language-derived source
Source Type: NL
Explanation:
- <$symbol or expression$>: <how it is inferred from the user request>
- <$symbol or expression$>: <how it is inferred from the user request>

Source Type B: document-grounded source
Source Type: DOC
Document: <document name>
Page: <page number>
Locator: <section / subsection / equation / figure / table label if visible; otherwise a short locator>
Evidence: <brief quoted or paraphrased supporting content from the page>
Explanation:
- <line 1: symbol or expression>: <why this evidence supports it and how it is modeled>
- <line 2: symbol or expression>: <why this evidence supports it and how it is modeled>

Additional source rules:
1. Do not mix NL and DOC in the same Source block.
2. Do not use placeholder values such as N/A for Document, Page, or Locator.
3. If no direct document grounding is available, use NL instead of leaving the Source blank.
4. For Variables, Parameters, Sets, and Constraints, the Explanation must normally be written item by item rather than as one generic paragraph.
5. Each important symbol, variable, parameter, set, or constraint group appearing in Content should normally have its own explanation line.
6. If the Content has multiple lines, the Explanation should normally explain them in the same order.
7. Avoid high-level summary explanations when the Content already contains several distinct symbols or expressions.
"""

demo_special_requirements = """

Additional Modeling Requirements:
1. Keep the original five-element schema, but make the final formulation as clear and solvable as possible.
2. Prefer a compact and closed formulation over a broad but abstract framework.
3. Avoid unused symbols, unused sets, and unused auxiliary variables.
4. If a variable or set does not actually appear in the final objective or constraints, omit it.
5. Avoid abstract placeholder functions such as f(x), g(x), or h(x) whenever a more explicit algebraic form can be written.
6. Parameters Content should use explicit numerical values whenever they are visible in the retrieved evidence.
7. If the retrieved pages do not provide enough numerical values for a solvable formulation, you must create a small reasonable demo instance and write those numerical values explicitly in Parameters Content.
8. Any assumed demo-instance values must also be explained clearly in Parameters Source.
9. Do not leave Parameters Content at a purely symbolic level if the final model is intended to be executable.
10. For Variables, Parameters, Sets, and Constraints, prefer short line-by-line content in the format '<mathematical expression>: <short meaning>' whenever helpful.
11. The final Math Model Content should be as directly translatable into optimization code as possible.
12. The final Math Model Content should use only variables, sets, and parameters that are actually defined and used.
13. Prefer a small executable demo formulation with a few concrete values over a broad symbolic framework with no solvable instance.
14. If a quantity is numeric in the document, keep that numeric value instead of replacing it with a symbol.
15. Every decision variable appearing in Variables Content must appear in Objective Content or Constraints Content.
16. If a listed variable does not appear in the final objective or constraints, remove it.
"""

demo_parameter_closure_requirements = """

Parameter Closure Requirements:
1. Parameters should contain numerical data, limits, coefficients, budgets, capacities, demands, times, or known constants.
2. Constraint functions or equation groups such as g(·), h(·), balance equations, or feasibility mappings should not be listed as Parameters unless they are truly fixed coefficient matrices or explicit numeric data.
3. If a symbol mainly represents a constraint family or equation family, place it in Constraints rather than Parameters.
4. If the final formulation is intended to be solvable, Parameters Content should provide enough concrete data to instantiate the model.
5. Parameters must not contain abstract state vectors, unnamed constraint families, or symbolic function placeholders unless they are truly treated as fixed known numeric data.
6. Symbols such as g^(i)(·), h^(i)(·), x_base, x_n-1 should not be listed as Parameters unless their role as fixed known data is explicitly justified.
7. If a quantity represents a decision-dependent state or a constraint family, place it in Variables or Constraints instead of Parameters.
"""

demo_math_model_purity_requirements = """

Math Model Purity Requirements:
1. Math Model Content must contain only the mathematical formulation itself.
2. Do not include explanatory notes such as "where", "with", "here", "in which", or variable-definition sentences inside Math Model Content.
3. Do not append bullet points, textual comments, or symbol explanations after the formula block.
4. If variable meanings, parameter values, or demo-instance assumptions need to be stated, place them in Variables Content, Parameters Content, or their Source fields, not in Math Model Content.
5. Math Model Content should be renderable as a pure LaTeX mathematical block without trailing prose.
"""

def Q2F_with_source_demo_v2(ques: str, hit_meta_block: str = ""):
    ques = bound_symbol + ques + bound_symbol

    meta_part = ""
    if hit_meta_block.strip():
        meta_part = (
            "\nRetrieved image metadata:\n"
            f"{hit_meta_block}\n"
        )

    return (
        five_description_complex
        + ques_description_five
        + ques
        + meta_part
        + demo_source_requirements
        + five_suffix_with_source
    )
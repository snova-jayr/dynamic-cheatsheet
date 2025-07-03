generator_prompt = """You are a financial reasoning expert answering a US GAAP tagging question using a curated cheatsheet.

Your goal is to derive the correct tag for a given numerical entity based on the **cheatsheet's reasoning strategies, disambiguation rules, and contextual cues**. If the cheatsheet lacks a clear answer, you may reason independently, but **clearly mark that segment as [Assumed Knowledge]**.

---

📋 Inputs:

[Cheatsheet]
{}

[Reflection]
{}  

[Question]
{}

[Context (optional)]
{}

---

🎯 Instructions:
1. **Extract the relevant financial concept** described in the question.
2. **Use the cheatsheet’s disambiguation rules or heuristics** to map the concept to the correct tag.
3. **Justify your reasoning step by step**, referring to the cheatsheet wherever applicable.
4. **Keep reasoning minimal but rigorous**. Avoid pattern matching. Base answers on semantics and structure.
5. **Present the final answer in this format**: Finish[Your final tag].

You should place your answer within square brackets after the prefix "Finish". So let's say your final answer is "apple", you should present the answer as Finish[apple].

---

🧠 Reasoning:
1. [What concept does this number represent?]
2. [What context clues support this interpretation?]
3. [Which disambiguation rule or heuristic from the cheatsheet applies?]
4. [What is the final US GAAP tag?]

Finish[Concise answer here, follow the format for final answer]

----
"""


reflector_prompt="""You are a domain expert in US GAAP reasoning. Your job is to **diagnose a model’s error**, extract generalizable insights, and recommend a correction that can **improve future tagging accuracy** through the cheatsheet.

---

📥 Inputs:

[Question]
{}

[Model Reasoning Trace]
{}

[Predicted Answer]
{}

[Ground Truth Answer]
{}

[Cheatsheet Available]
{}

---

🔍 Diagnostic Analysis (use this structure):

**Error Type:** [Conceptual | Contextual | Disambiguation | Arithmetic | Omission | Overfit | Misreading]

**What Went Wrong:**
- Quote or summarize the faulty reasoning step.
- What misinterpretation or faulty assumption caused the mistake?

**Root Cause:**
- Did the model fail to distinguish between similar tags?
- Did it misinterpret context, structural cues, or underlying financial concepts?

**Correct Approach:**
- What reasoning path **should** the model have followed?
- Is there a missing or underemphasized rule in the cheatsheet?

**Key Insight for Cheatsheet:**
- Generalize this insight into a **rule, warning, or disambiguation strategy**.
- Focus on contextual clues, failure patterns, or structural logic.

Example format:
> “If a sentence mentions a total and a subset that affects effective tax rate, tag them distinctly using [X] and [Y].”

Only return the structured reflection. Do not explain the prompt format.

----
"""

curator_prompt="""You are the curator of a US GAAP cheatsheet used to help models answer financial tagging questions accurately.

Your job is to update the cheatsheet based on the latest **reflection**, **question**, and **correct answer**. The goal is to improve model performance by distilling **high-precision strategies, disambiguation rules, failure patterns, and context cues**.

---

📥 Inputs:

[Previous Cheatsheet]
{}

[Reflection]
{}

[Question]
{}

[Correct Answer]
{}

---

🎯 Curation Rules:
- Do **not repeat** existing entries; **merge or refactor** when possible.
- Abstract mistakes into **general rules**, not specific to this question.
- Write in **minimal, modular, and structured format**.
- Each section must be **useful in isolation** to a model with no memory.

---

🧠 Cheatsheet Format:
<cheatsheet>

## STRATEGIES & REASONING HEURISTICS
- (Reusable reasoning rules like “identify concept → check sentence role → disambiguate”)

## DISAMBIGUATION RULES
- (Precise logic for choosing between similar tags. Avoid quoting the tags in multiple sections.)

## COMMON MISTAKES & FAILURE PATTERNS
- Mistake: [Summarize misunderstanding]
  - Why: [Cognitive shortcut or contextual ambiguity]
  - Prevention: [New heuristic or verification step]

## CONTEXTUAL CUES & STRUCTURAL SIGNALS
- (Lexical or syntactic patterns that signal deeper meaning, e.g. conditionality, containment, comparisons)

</cheatsheet>

Only return the updated cheatsheet. Do not repeat the input or restate the prompt.

----
"""


generator_prompt = """You are a financial analysis expert tasked with answering questions using your knowledge, a curated cheatsheet of strategies and insights and a reflection that goes over the diagnosis of all previous mistakes made while answering the question.

**Instructions:**
- Read the cheatsheet carefully and apply relevant strategies, formulas, and insights
- Pay attention to common mistakes listed in the cheatsheet and avoid them
- Show your reasoning step-by-step
- Be concise but thorough in your analysis
- If the cheatsheet contains relevant code snippets or formulas, use them appropriately
- Double-check your calculations and logic before providing the final answer

You should place your answer within square brackets after the prefix "Finish". So let's say your final answer is "apple", you should present the answer as Finish[apple].

**Cheatsheet:**
{}

**Reflection**
{}

**Question:**
{}

**Context (if provided):**
{}

**Your Response:**
Please provide your step-by-step reasoning and final answer. Structure your response as:

**Reasoning:**
[Your detailed analysis and calculations]

**Final Answer:**
Finish[Your concise final answer]

---
"""

reflector_prompt="""You are an expert financial analyst and educator. Your job is to diagnose why a model's reasoning went wrong by comparing the predicted answer with the ground truth.

**Instructions:**
- Carefully analyze the model's reasoning trace to identify where it went wrong
- Compare the predicted answer with the ground truth to understand the gap
- Identify specific conceptual errors, calculation mistakes, or misapplied strategies
- Provide actionable insights that could help the model avoid this mistake in the future
- Focus on the root cause, not just surface-level errors
- Be specific about what the model should have done differently

**Question:**
{}

**Model's Reasoning Trace:**
{}

**Model's Predicted Answer:**
{}

**Ground Truth Answer:**
{}

**Your Diagnostic Analysis:**
Please structure your reflection as:

**Error Identification:**
[What specifically went wrong in the reasoning?]

**Root Cause Analysis:**
[Why did this error occur? What concept was misunderstood?]

**Correct Approach:**
[What should the model have done instead?]

**Key Insight:**
[What strategy, formula, or principle should be remembered to avoid this error?]

---
"""


curator_prompt="""You are a master curator of financial knowledge. Your job is to maintain and improve a comprehensive cheatsheet that helps models solve financial problems accurately.

**Instructions:**
- Review the previous cheatsheet, the reflection from the failed attempt, and the successful solution
- Add new insights, strategies, formulas, or common mistakes to the cheatsheet
- Organize information clearly with appropriate categories
- Avoid redundancy - if similar advice already exists, enhance it rather than duplicate
- Prioritize actionable, specific guidance over general statements
- Include code snippets when they would be helpful for calculations
- Balance positive strategies ("do this") with negative examples ("avoid this")
- Keep the cheatsheet concise but comprehensive
- Format your cheatsheet between <cheatsheet> and </cheatsheet> tags 

**Previous Cheatsheet:**
{}

**Recent Reflection:**
{}

**Question that was Solved:**
{}

**Successful Answer:**
{}

**Your Task:**
Update the cheatsheet by incorporating new learnings. Structure your updated cheatsheet with these sections:

## FINANCIAL STRATEGIES & INSIGHTS
[Key approaches, methodologies, and decision-making frameworks]

## FORMULAS & CALCULATIONS
[Important formulas, ratios, and computational methods]

## CODE SNIPPETS & TEMPLATES
[Useful Python code for common calculations]

## COMMON MISTAKES TO AVOID
[Specific errors and misconceptions to watch out for]

## PROBLEM-SOLVING HEURISTICS
[Rules of thumb and shortcuts for different problem types]

## CONTEXT CLUES & INDICATORS
[How to identify what approach to use based on question wording or data provided]

**Updated Cheatsheet:**
[Provide the enhanced cheatsheet following the structure above]

---
"""

compressor_prompt = """You are a master compressor of cheatsheets containing financial knowledge. You will be given a cheatsheet which has been curated from some samples. Your job is to create a clear, concise cheatsheet with minimum redundancy. 

**Instructions:**
- Perform a "consolidation pass" - merge similar patterns, remove redundant examples, and compress verbose explanations
- Keep the cheatsheet concise but comprehensive
- If there is no redundancy in the cheatsheet or you feel like nothing can be removed, then it is okay to retain the original cheatsheet 
- Format your cheatsheet between <cheatsheet> and </cheatsheet> tags 

**Previous Cheatsheet:**
{}

---
"""

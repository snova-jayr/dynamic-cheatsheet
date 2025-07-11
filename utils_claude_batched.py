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

'''
reflector_aggregator_prompt = """You are an expert financial aggregator. You will be given a list of reflections from a batch of financial questions. A reflection is a diagnosis when the final answer produced differs from the ground truth, which explains what went wrong and what could be done better when answering questions. Your job is to analyze these reflections one by one and the previous cheatsheet. Based on the reflections provided, you have to judge if you can extract valuable insights, clues, strategies, code snippets or common mistakes that can be used for answering future questions. Your job is critical, as whatever you extract will determine what the new cheatsheet would look like. A cheatsheet is a repository of dynamically evolving knowledge, that is accumulated from a set of examples to be used later for inference. Keep in mind, that the model is not going to be trained. It will only usewhatever you provide for solving more financial problems. You are a part of an inference-only agentic system.    

**Instructions:**
- Carefully analyze each reflection to understand if you can derive valuable insights and clues from it. 
- Identify specific conceptual errors, calculation mistakes, or misapplied strategies
- Provide actionable insights and code snippets that could help the model avoid this mistake in the future
- Focus on the root cause, not just surface-level errors
- It's okay to include specific rules or mapping pertaining to the training samples if you think that'll be useful.
- If you think there is nothing useful about a reflection provided, it's okay to skip that particular reflection. 
- Make sure to avoid redundancy or repetitive information in the aggregation you're producing. Please be concise. 
- After analysing each reflection, produce a summary which is like an aggregate reflection of useful strategies, learnings and common mistakes to avoid.
- You don't have to do anything to a cheatsheet, just give me a comprehensive summary from the reflections provided. 

**Reflection #1:**
{}

**Reflection #2:**
{}

**Reflection #3:**
{}

**Reflection #4:**
{}


**Your Diagnostic Analysis:**
Please structure your reflection as:

**Key Insight:**
[What strategy, formula, or principle should be remembered to avoid this error?]

---
"""
'''

reflector_aggregator_prompt="""You are an expert meta-analyst specializing in aggregating and synthesizing diagnostic insights from multiple model performance reflections. Your task is to analyze a collection of individual reflections and create a comprehensive, consolidated summary that captures the most important patterns, insights, and learnings.

**IMPORTANT CONTEXT:** This is a training-free approach. The aggregated reflection you produce will be used for test-time inference through in-context learning (ICL). Focus on actionable insights and patterns that can be provided as context to improve future model performance, rather than suggesting training data improvements or model retraining.

**CRITICAL REQUIREMENT:** Be extremely specific and detailed. Avoid generic statements like "model should understand context better" or "improve accuracy." Instead, provide concrete examples, specific decision rules, and detailed mapping patterns that can be directly applied during inference.

## Instructions:

**Input:** You will receive multiple individual reflections, each containing:
- Error Identification
- Root Cause Analysis  
- Correct Approach
- Key Insight

**Output:** Generate a consolidated reflection that synthesizes all inputs into a unified diagnostic summary with maximum specificity.

## Individual Reflections to Analyze:

# Reflection 1 
{}

# Reflection 2 
{}

# Reflection 3 
{}

# Reflection 4 
{}

## Your Consolidated Analysis:

Please structure your aggregated reflection as follows:

**Common Error Patterns:**
[List specific, concrete error types with exact examples. Instead of "misapplication of tags," specify "model incorrectly maps dollar amounts in acquisition contexts to 'DebtInstrumentCarryingAmount' when it should be 'BusinessCombinationConsiderationTransferred1'"]

**Root Cause Synthesis:**
[Identify specific reasoning failures with examples. What exact decision points or contextual cues is the model missing? Provide concrete instances of flawed reasoning patterns.]

**Consolidated Correct Approach:**
[Create a detailed, step-by-step decision framework with specific rules and conditions. Include exact phrases, keywords, or numerical patterns that should trigger specific responses. Make this actionable enough to follow like a checklist.]

**Key Contextual Clues:**
[Extract exact keywords, phrases, numerical patterns, or contextual markers from the reflections. Specify what these clues should trigger (e.g., "When sentence contains 'revolving loan facility' + 'repay' + dollar amount → use 'RepaymentsOfDebt' tag")]

**Critical Insights & Strategies:**
[Provide specific decision rules, disambiguation criteria, and exact mapping patterns. Include concrete examples of correct vs incorrect applications.]

**Specific Examples & Mappings:**
[Extract and organize all specific examples, correct answers, and mapping patterns from the individual reflections. Create a reference guide of concrete cases.]

**Priority Recommendations:**
[Rank specific contextual guidance with exact implementation details. What specific rules or patterns should be applied first during inference?]

## Analysis Guidelines:

1. **Maximum Specificity:** Every insight must be concrete and actionable
2. **Example-Driven:** Include specific examples, numbers, and exact phrases from reflections
3. **Decision Rules:** Create clear if-then rules that can be applied during inference
4. **Pattern Extraction:** Identify exact patterns, keywords, and contextual markers
5. **Avoid Generalities:** Replace vague advice with specific, implementable guidance
6. **Actionable Framework:** Create guidance that can be directly applied as context

## Output Format:
- Use specific examples throughout
- Include exact phrases, numbers, and patterns
- Create actionable decision rules
- Provide concrete mapping guidelines
- Focus on immediately applicable insights

---
"""

curator_prompt="""You are a master curator of financial knowledge. Your job is to maintain and improve a comprehensive cheatsheet that helps models solve financial problems accurately.

**Instructions:**
- Review the previous cheatsheet, the aggregated reflection (diagnosis) a batch of training samples, and the successful solution
- Add new insights, strategies, contextual clues, formulas, or common mistakes to the cheatsheet
- Code snippets or tag mapping or any contextual clues that draw a relation between context and US GAAP tags are important pieces of information. Try not to get rid of that. 
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

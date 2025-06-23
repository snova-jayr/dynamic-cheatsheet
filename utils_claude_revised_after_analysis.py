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


curator_prompt="""You are a master curator of US GAAP financial knowledge with deep expertise in contextual analysis and precision tagging. Your job is to maintain and improve a comprehensive cheatsheet that emphasizes **conceptual understanding over pattern matching** and **proactive error prevention over reactive error identification**.

### Core Philosophy
- **Context-First Approach**: Prioritize understanding financial concepts within their specific contexts rather than relying on keyword triggers
- **Granular Precision**: Provide highly specific tag distinctions with clear usage rules for each scenario
- **Comprehensive Coverage**: Address the full breadth of financial domains with deep expertise in each area
- **Proactive Prevention**: Focus on preventing errors through contextual warnings rather than just listing common mistakes

### Instructions:
1. **Review the Previous Cheatsheet**: Review the previous cheatsheet, the reflection from the failed attempt, and the successful solution
2. **Analyze for Conceptual Gaps**: Review the previous cheatsheet for missing financial domains, insufficient tag granularity, and oversimplified pattern matching
3. **Enhance Contextual Sophistication**: Add nuanced guidance that helps distinguish between similar concepts based on context
4. **Expand Domain Coverage**: Ensure comprehensive coverage across all major US GAAP areas
5. **Implement Prevention Strategies**: Include specific warnings about when and why certain errors occur
6. **Add Mathematical Rigor**: Include formulas and calculations where relevant to validate tag selections
7. **Create Hierarchical Decision Trees**: Organize guidance from broad concepts to specific tag selections
8. **Formatting Instructions**: Format your cheatsheet between <cheatsheet> and </cheatsheet> tags 

**Previous Cheatsheet:** 
{}

**Recent Reflection:**
{}

**Question that was Solved:**
{}

**Successful Answer:**
{}

### Your Task: 
Update the cheatsheet by incorporating new learnings with emphasis on the principles that make cheatsheets perform well. Structure your updated cheatsheet with these enhanced sections:

## UNDERSTANDING US GAAP CONCEPTS AND CONTEXTUAL ANALYSIS
[Deep conceptual frameworks for understanding financial concepts within their specific contexts. Include guidance on analyzing sentence context, distinguishing between similar concepts, and understanding the nuances that drive tag selection. Emphasize "why" over "what".]

## COMPREHENSIVE DOMAIN COVERAGE WITH GRANULAR TAG PRECISION
[Organize by major financial domains (Business Combinations, Debt Instruments, Share-Based Compensation, etc.) with highly specific tag distinctions within each domain. For each domain, provide:
- Multiple related tags with clear differentiation rules
- Contextual clues that determine tag selection
- Edge cases and exceptions
- Specific examples of when to use each tag]

## PROACTIVE ERROR PREVENTION STRATEGIES
[Focus on preventing errors before they occur through:
- Contextual warnings about commonly confused concepts
- Specific guidance on "Be cautious not to..." scenarios
- Disambiguation rules for similar tags
- Context analysis frameworks to avoid misclassification]

## FORMULAS, CALCULATIONS & MATHEMATICAL VALIDATION
[Include relevant formulas and mathematical relationships that can validate tag selections:
- Interest rate calculations (effective, stated, weighted average)
- Amortization and depreciation formulas
- Financial ratio calculations
- Code implementations for complex calculations]

## ADVANCED CODE SNIPPETS & DECISION TREES
[Provide sophisticated code that demonstrates:
- Hierarchical decision-making processes
- Context analysis beyond simple string matching
- Multi-factor tag selection logic
- Validation and confidence scoring mechanisms]

## DISAMBIGUATION FRAMEWORKS FOR COMPLEX SCENARIOS
[Detailed guidance for handling:
- Multiple concepts in single sentences
- Ambiguous contextual clues
- Edge cases and exceptions
- Conflicting indicators
- When to prioritize specific over general tags]

## COMPREHENSIVE MISTAKE PREVENTION GUIDE
[Organized by financial domain, include:
- Why specific mistakes occur (root cause analysis)
- Contextual clues that prevent misclassification
- Specific warning signs to watch for
- Validation questions to ask before tag selection]

## CONTEXTUAL PATTERN RECOGNITION BEYOND KEYWORDS
[Advanced guidance on:
- Understanding financial statement relationships
- Recognizing transaction types and their implications
- Identifying measurement bases (fair value, carrying amount, etc.)
- Understanding temporal aspects (recognition vs. measurement timing)]

### Quality Standards for Enhancement:
1. **Avoid Simple Pattern Matching**: Replace trigger word lists with contextual analysis frameworks
2. **Maximize Tag Granularity**: Provide specific tags for specific situations rather than general categories
3. **Include Edge Cases**: Address uncommon but important scenarios
4. **Provide Mathematical Support**: Include calculations that validate tag selections
5. **Focus on Prevention**: Emphasize avoiding errors rather than just identifying them
6. **Ensure Comprehensive Coverage**: Address all major US GAAP domains with appropriate depth

### Success Metrics:
- Each major financial domain should have 3+ related tags with clear distinctions
- Every disambiguation rule should explain "why" not just "when"
- Prevention strategies should outnumber reactive error identification
- Code examples should demonstrate sophisticated decision-making, not simple pattern matching
- Mathematical formulas should support and validate conceptual understanding

Updated Cheatsheet: [Provide the enhanced cheatsheet following the structure above, emphasizing the principles that distinguish high-performing cheatsheets from low-performing ones]
"""

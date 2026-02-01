import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from tools import search_eu_directive

# Load environment variables
load_dotenv()

# Initialize the LLM (GPT-4o-mini for cost efficiency)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,  # Deterministic outputs (no creativity needed)
    api_key=os.getenv("OPENAI_API_KEY")
)
# ============================================================
# AGENT 1: GREENWASHING ANALYZER
# ============================================================

analyzer_instructions = """You are an expert greenwashing analyst.

Your task:
1. Analyze the marketing text provided by the user
2. Use the search_eu_directive tool to find relevant regulations about vague claims, unsubstantiated statements, and misleading environmental claims
3. Determine if the text contains greenwashing
4. Provide a confidence score (0-100%)

IMPORTANT CONFIDENCE SCORING:
- 90-100%: Extremely obvious greenwashing (multiple absolute claims, no evidence at all)
- 70-89%: Clear greenwashing (vague terms, lack of substantiation)
- 50-69%: Likely greenwashing (some vague language, questionable claims)
- 30-49%: Borderline (some good practices, some concerns)
- 10-29%: Minimal greenwashing (mostly compliant with minor issues)
- 0-9%: No greenwashing (specific, substantiated, compliant)

Greenwashing indicators:
- Vague terms without specifics: "eco-friendly", "green", "sustainable" without proof
- Absolute claims: "100% sustainable", "completely carbon neutral" without evidence
- Misleading imagery or language that implies environmental benefits without substance
- Hidden trade-offs: highlighting one green aspect while ignoring harmful ones


COUNT the number of violations:
- 1 issue = lower confidence
- 2-3 issues = medium confidence  
- 4+ issues = high confidence

VARY your scores based on severity, don't default to 85%!

Output format (be concise):
{
  "is_greenwashing": true/false,
  "confidence": 0-100,
  "reasoning": "Brief explanation of why this is/isn't greenwashing",
  "flagged_phrases": ["specific phrases that are problematic"]
}

Be objective and cite the directive when relevant.
"""

analyzer_agent = create_react_agent(
    llm,
    tools=[search_eu_directive],
    state_modifier=analyzer_instructions
)

# ============================================================
# AGENT 2: ARTICLE VALIDATOR
# ============================================================

validator_instructions = """You are a legal compliance expert specializing in EU environmental regulations.

ARTICLE REFERENCE GUIDE (use this to know which articles to search for):

Article 3: Substantiation of explicit environmental claims (scientific evidence required)
Article 4: Assessment of environmental impacts (lifecycle, significant impacts)
Article 5: Presentation of environmental claims (avoid generic terms, be specific)
Article 6: Comparative environmental claims (comparison to other products/traders)
Article 7: Environmental claims related to future environmental performance (commitments, monitoring)
Article 8: Information requirements for consumers (clear, accessible information)
Article 9: Aggregated carbon footprint information (specific to carbon footprints)
Article 10: Environmental labelling schemes (certification, third-party verification)
Article 11: Update of explicit environmental claims (keeping claims current)
Article 12: Verification and conformity assessment (independent verification)

CRITICAL SEARCH STRATEGY - YOU MUST DO ALL THESE STEPS:

Step 1: Identify claim type from the analysis
- Future commitment (e.g., "will be neutral by 2030") → Article 7
- Comparison (e.g., "better than X") → Article 6
- Label/certification mention → Article 8 or 10
- Vague terms (e.g., "eco-friendly") → Article 5
- Unsubstantiated → Article 3

Step 2: Search for the PRIMARY article
- Use search_eu_directive with: "Article [NUMBER] [specific topic]"
- Example: "Article 7 future environmental performance"

Step 3: Search for SUBSTANTIATION requirements
- Use search_eu_directive with: "Article 3 substantiation scientific evidence"

Step 4: Search for one MORE relevant article
- Based on the claim, pick another relevant article
- Search for it specifically

YOU MUST USE THE SEARCH TOOL AT LEAST 3 TIMES WITH DIFFERENT QUERIES!

DO NOT just search "greenwashing" or "environmental claims" generally.
DO NOT stop after finding Articles 3 and 4.
SEARCH FOR SPECIFIC ARTICLE NUMBERS using the guide above.

Example workflow:
User claim: "We will be carbon neutral by 2030"
1. Identify: This is a FUTURE claim → Article 7
2. Search: "Article 7 future environmental performance"
3. Search: "Article 3 substantiation requirements"
4. Search: "Article 11 verification commitments"
Result: Articles 7, 3, 11

Your task:
1. READ the claim type from the analysis
2. LOOK UP which article matches in the guide above
3. SEARCH for that specific article
4. Then search for related articles




Output format (be concise):
{
  "violated_articles": ["Article X", "Article Y"],
  "explanations": {
    "Article X": "Brief explanation of what this article requires and how the claim violates it",
    "Article Y": "Brief explanation..."
  }
}

If no violations found, return empty lists.
Be precise with article numbers and cite the directive text.
"""

validator_agent = create_react_agent(
    llm,
    tools=[search_eu_directive],
    state_modifier=validator_instructions
)

# ============================================================
# AGENT 3: COMPLIANT REWRITER
# ============================================================

rewriter_instructions = """You are a marketing compliance advisor.

Your task:
1. You receive the original text and violation details
2. Use search_eu_directive to find examples of compliant language and requirements
3. Rewrite the marketing text to be compliant with EU directive
4. Remove vague terms, add specifics, include substantiation

Compliance principles:
- Replace vague terms with specific, measurable claims
- Add evidence/certification references where needed
- Remove absolute claims or qualify them with data
- Be honest about limitations and trade-offs
- Keep marketing appeal while being truthful

Output format (be concise):
{
  "suggested_text": "The rewritten compliant version",
  "changes_made": ["List of key changes: 'Removed vague term X', 'Added specific metric Y'"]
}

Make the text sound natural and professional, not overly legalistic.
"""

rewriter_agent = create_react_agent(
    llm,
    tools=[search_eu_directive],
    state_modifier=rewriter_instructions
)

# EXPORT ALL AGENTS
# Make agents available for import
__all__ = ['analyzer_agent', 'validator_agent', 'rewriter_agent']

# Optional: Test function to verify agents work
if __name__ == "__main__":
    print("✅ Agents defined successfully!")
    print("\nAvailable agents:")
    print("1. analyzer_agent - Detects greenwashing")
    print("2. validator_agent - Finds violated articles")
    print("3. rewriter_agent - Generates compliant text")


from typing import TypedDict, Annotated
from langgraph.graph import StateGraph

# Define constants
END = "__end__"
START = "__start__"
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from agents import analyzer_agent, validator_agent, rewriter_agent

# Define the state that flows between agents
class AgentState(TypedDict):
    """
    State object that gets passed between agents
    Each agent reads from and writes to this state
    """
    # Input
    input_text: str  # Original marketing text from user
    
    # Messages (conversation history with agents)
    messages: Annotated[list, add_messages]
    
    # Agent 1 outputs
    is_greenwashing: bool
    confidence: int
    reasoning: str
    flagged_phrases: list
    
    # Agent 2 outputs
    violated_articles: list
    article_explanations: dict
    
    # Agent 3 outputs
    suggested_text: str
    changes_made: list
# ============================================================
# AGENT NODE FUNCTIONS
# ============================================================

def analyze_node(state: AgentState) -> AgentState:
    """
    Node 1: Analyze text for greenwashing
    """
    print("\n🔍 Agent 1: Analyzing for greenwashing...")
    
    # Get input text
    input_text = state["input_text"]
    
    # Create message for agent
    message = HumanMessage(content=f"Analyze this marketing text for greenwashing: {input_text}")
    
    # Invoke agent
    result = analyzer_agent.invoke({"messages": [message]})
    
    # Extract agent's response
    agent_response = result["messages"][-1].content
    
    # Parse response (simplified - in production you'd parse JSON properly)
    # For now, store the full response
    state["messages"] = result["messages"]
    state["reasoning"] = agent_response
    
    print(f"✅ Analysis complete")
    
    return state


def validate_node(state: AgentState) -> AgentState:
    """
    Node 2: Find violated articles
    """
    print("\n📋 Agent 2: Identifying violated articles...")
    
    # Get previous analysis
    input_text = state["input_text"]
    previous_analysis = state["reasoning"]
    
    # Create message for agent
    
    message = HumanMessage(
        content=f"""Original text: {input_text}

    Previous analysis: {previous_analysis}

    Find which SPECIFIC EU directive articles are violated. 

    IMPORTANT: 
    - Look beyond Articles 3 and 5
    - If the claim involves comparisons, search for Article 6
    - If it's about future performance, search for Article 7
    - If it's about labels, search for Article 10
    - Search the directive multiple times with different queries if needed

    Be specific and find the EXACT articles that match this particular violation."""
    )
    
    # Invoke agent
    result = validator_agent.invoke({"messages": [message]})
    
    # Extract agent's response
    agent_response = result["messages"][-1].content
    
    # Update state
    state["messages"].extend(result["messages"])
    state["article_explanations"] = {"response": agent_response}
    
    print(f"✅ Validation complete")
    
    return state


def rewrite_node(state: AgentState) -> AgentState:
    """
    Node 3: Generate compliant alternative
    """
    print("\n✏️ Agent 3: Generating compliant alternative...")
    
    # Get all previous context
    input_text = state["input_text"]
    analysis = state["reasoning"]
    violations = state["article_explanations"]
    
    # Create message for agent
    message = HumanMessage(
        content=f"Original text: {input_text}\n\nAnalysis: {analysis}\n\nViolations: {violations}\n\nRewrite this to be compliant."
    )
    
    # Invoke agent
    result = rewriter_agent.invoke({"messages": [message]})
    
    # Extract agent's response
    agent_response = result["messages"][-1].content
    
    # Update state
    state["messages"].extend(result["messages"])
    state["suggested_text"] = agent_response
    
    print(f"✅ Rewrite complete")
    
    return state
# ============================================================
# AGENT NODE FUNCTIONS
# ============================================================

def analyze_node(state: AgentState) -> AgentState:
    """
    Node 1: Analyze text for greenwashing
    """
    print("\n🔍 Agent 1: Analyzing for greenwashing...")
    
    # Get input text
    input_text = state["input_text"]
    
    # Create message for agent
    message = HumanMessage(content=f"Analyze this marketing text for greenwashing: {input_text}")
    
    # Invoke agent
    result = analyzer_agent.invoke({"messages": [message]})
    
    # Extract agent's response
    agent_response = result["messages"][-1].content
    
    # Parse response (simplified - in production you'd parse JSON properly)
    # For now, store the full response
    state["messages"] = result["messages"]
    state["reasoning"] = agent_response
    
    print(f"✅ Analysis complete")
    
    return state


def validate_node(state: AgentState) -> AgentState:
    """
    Node 2: Find violated articles
    """
    print("\n📋 Agent 2: Identifying violated articles...")
    
    # Get previous analysis
    input_text = state["input_text"]
    previous_analysis = state["reasoning"]
    
    # Create message for agent
    message = HumanMessage(
        content=f"Original text: {input_text}\n\nPrevious analysis: {previous_analysis}\n\nFind which specific EU directive articles are violated."
    )
    
    # Invoke agent
    result = validator_agent.invoke({"messages": [message]})
    
    # Extract agent's response
    agent_response = result["messages"][-1].content
    
    # Update state
    state["messages"].extend(result["messages"])
    state["article_explanations"] = {"response": agent_response}
    
    print(f"✅ Validation complete")
    
    return state


def rewrite_node(state: AgentState) -> AgentState:
    """
    Node 3: Generate compliant alternative
    """
    print("\n✏️ Agent 3: Generating compliant alternative...")
    
    # Get all previous context
    input_text = state["input_text"]
    analysis = state["reasoning"]
    violations = state["article_explanations"]
    
    # Create message for agent
    message = HumanMessage(
        content=f"Original text: {input_text}\n\nAnalysis: {analysis}\n\nViolations: {violations}\n\nRewrite this to be compliant."
    )
    
    # Invoke agent
    result = rewriter_agent.invoke({"messages": [message]})
    
    # Extract agent's response
    agent_response = result["messages"][-1].content
    
    # Update state
    state["messages"].extend(result["messages"])
    state["suggested_text"] = agent_response
    
    print(f"✅ Rewrite complete")
    
    return state
# ============================================================
# BUILD THE WORKFLOW GRAPH
# ============================================================

def create_workflow():
    """
    Create and compile the LangGraph workflow
    """
    # Initialize graph
    workflow = StateGraph(AgentState)
    
    # Add nodes (agents)
    workflow.add_node("analyzer", analyze_node)
    workflow.add_node("validator", validate_node)
    workflow.add_node("rewriter", rewrite_node)
    
    # Define edges (flow between agents)
    workflow.add_edge(START, "analyzer")       # Start → Agent 1
    workflow.add_edge("analyzer", "validator")  # Agent 1 → Agent 2
    workflow.add_edge("validator", "rewriter")  # Agent 2 → Agent 3
    workflow.add_edge("rewriter", END)          # Agent 3 → End
    
    # Compile the graph
    app = workflow.compile()
    
    return app

# Create the app
app = create_workflow()
# ============================================================
# HELPER FUNCTION FOR EASY USE
# ============================================================

def analyze_greenwashing(text: str) -> dict:
    """
    Main function to analyze marketing text
    
    Args:
        text: Marketing text to analyze
        
    Returns:
        Dictionary with analysis, violations, and suggestions
    """
    print(f"\n{'='*60}")
    print(f"ANALYZING: {text}")
    print(f"{'='*60}")
    
    # Create initial state
    initial_state = {
        "input_text": text,
        "messages": [],
        "is_greenwashing": False,
        "confidence": 0,
        "reasoning": "",
        "flagged_phrases": [],
        "violated_articles": [],
        "article_explanations": {},
        "suggested_text": "",
        "changes_made": []
    }
    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    # Extract results
    result = {
        "original_text": text,
        "analysis": final_state.get("reasoning", ""),
        "violations": final_state.get("article_explanations", {}),
        "suggestion": final_state.get("suggested_text", "")
    }
    
    print(f"\n{'='*60}")
    print("✅ ANALYSIS COMPLETE")
    print(f"{'='*60}\n")
    
    return result


# ============================================================
# TEST THE WORKFLOW
# ============================================================

if __name__ == "__main__":
    print("🚀 Testing Greenwashing Detection Workflow\n")
    
    # Test case
    test_text = "Our 100% eco-friendly sustainable product is completely carbon neutral"
    
    result = analyze_greenwashing(test_text)
    
    print("\n📊 RESULTS:")
    print(f"\n1. ANALYSIS:\n{result['analysis']}")
    print(f"\n2. VIOLATIONS:\n{result['violations']}")
    print(f"\n3. SUGGESTION:\n{result['suggestion']}")

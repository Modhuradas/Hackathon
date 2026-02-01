"""
Streamlit UI for Greenwashing Detection System
"""

import streamlit as st
from graph import analyze_greenwashing
import json

# Page configuration
st.set_page_config(
    page_title="Greenwashing Detector",
    page_icon="🌱",
    layout="wide"
)

# Title and description
st.title("🌱 Greenwashing Detection System")
st.markdown("""
This AI-powered system analyzes marketing claims for greenwashing using:
- **RAG** (Retrieval-Augmented Generation) with EU Green Claims Directive
- **Multi-Agent System** (Analyzer → Validator → Rewriter)
- **LangGraph** for workflow orchestration
""")

st.divider()
# Input section
st.subheader("📝 Enter Marketing Text")

# Text area for user input
input_text = st.text_area(
    label="Marketing claim to analyze:",
    placeholder="Example: Our 100% eco-friendly product is completely sustainable...",
    height=150,
    help="Enter any marketing text that makes environmental claims"
)

# Analyze button
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("🗑️ Clear", use_container_width=True)

# Handle clear button
if clear_button:
    st.rerun()

st.divider()
# Main analysis logic
if analyze_button:
    if not input_text.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        # Show loading spinner
        with st.spinner("🤖 AI Agents are analyzing... This may take 30-60 seconds..."):
            try:
                # Call the workflow
                result = analyze_greenwashing(input_text)
                
                # Store in session state so it persists
                st.session_state['result'] = result
                st.session_state['analyzed'] = True
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.error("Please check your API key and try again.")
                st.session_state['analyzed'] = False
# Display results if analysis was performed
if st.session_state.get('analyzed', False):
    result = st.session_state.get('result', {})
    
    st.success("✅ Analysis Complete!")
    
    # Create three columns for results
    col1, col2, col3 = st.columns(3)
    
    # Column 1: Analysis
    with col1:
        st.subheader("🔍 Analysis")
        
        # Parse the analysis (it's a JSON string)
        try:
            analysis = json.loads(result.get('analysis', '{}'))
            
            # Display greenwashing status
            if analysis.get('is_greenwashing'):
                st.error("**Status:** Greenwashing Detected")
            else:
                st.success("**Status:** No Greenwashing Detected")
            
            # Display confidence
            confidence = analysis.get('confidence', 0)
            st.metric("Confidence", f"{confidence}%")
            
            # Display reasoning
            st.markdown("**Reasoning:**")
            st.info(analysis.get('reasoning', 'No reasoning provided'))
            
            # Display flagged phrases
            flagged = analysis.get('flagged_phrases', [])
            if flagged:
                st.markdown("**Flagged Phrases:**")
                for phrase in flagged:
                    st.markdown(f"- `{phrase}`")
        
        except json.JSONDecodeError:
            # If parsing fails, show raw text
            st.write(result.get('analysis', 'No analysis available'))
    
    # Column 2: Violations
    with col2:
        st.subheader("📋 Violations")
        
        # Parse violations
        violations_data = result.get('violations', {})
        
        try:
            if isinstance(violations_data, dict) and 'response' in violations_data:
                violations = json.loads(violations_data['response'])
            else:
                violations = violations_data
            
            # Display violated articles
            violated_articles = violations.get('violated_articles', [])
            
            if violated_articles:
                st.warning(f"**{len(violated_articles)} Article(s) Violated**")
                
                # Display each article with explanation
                explanations = violations.get('explanations', {})
                for article in violated_articles:
                    with st.expander(f"📄 {article}"):
                        st.write(explanations.get(article, 'No explanation available'))
            else:
                st.success("No violations found")
        
        except (json.JSONDecodeError, TypeError):
            st.write(violations_data)
    
    # Column 3: Suggestion
    with col3:
        st.subheader("✏️ Compliant Alternative")
        
        # Parse suggestion
        try:
            suggestion = json.loads(result.get('suggestion', '{}'))
            
            # Display suggested text
            suggested_text = suggestion.get('suggested_text', '')
            if suggested_text:
                st.success("**Suggested Text:**")
                st.write(suggested_text)
                
                # Display changes made
                changes = suggestion.get('changes_made', [])
                if changes:
                    st.markdown("**Changes Made:**")
                    for change in changes:
                        st.markdown(f"- {change}")
            else:
                st.info("No rewrite needed - text appears compliant")
        
        except json.JSONDecodeError:
            st.write(result.get('suggestion', 'No suggestion available'))
# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    
    st.markdown("""
    ### How it works:
    
    1. **Agent 1: Analyzer**
       - Detects greenwashing patterns
       - Provides confidence score
       - Flags problematic phrases
    
    2. **Agent 2: Validator**
       - Identifies violated EU articles
       - Provides regulatory context
       - Cites specific requirements
    
    3. **Agent 3: Rewriter**
       - Generates compliant alternative
       - Maintains marketing appeal
       - Lists all changes made
    
    ### Technologies:
    - OpenAI GPT-4o-mini
    - LangChain & LangGraph
    - ChromaDB (Vector Database)
    - EU Green Claims Directive
    """)
    
    st.divider()
    
    st.markdown("""
    ### Example Inputs:
    
    Try these examples:
    - "Our 100% eco-friendly product"
    - "Completely carbon neutral solution"
    - "Made from sustainable materials"
    - "Green and environmentally responsible"
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>Built with RAG, Multi-Agent AI, and LangGraph | Powered by OpenAI & EU Directive</p>
</div>
""", unsafe_allow_html=True)
                
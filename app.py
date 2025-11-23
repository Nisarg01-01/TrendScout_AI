import streamlit as st
import time
import pandas as pd
import re
from io import StringIO
from retrieval_service import TrendScoutBackend

# Page Config
st.set_page_config(page_title="TrendScout AI", page_icon="📈", layout="wide")

# Initialize Backend
@st.cache_resource
def get_backend():
    return TrendScoutBackend()

backend = get_backend()

def render_content(content):
    """
    I'm taking the raw answer from the LLM and making it look good.
    If there are any CSV tables hidden in the text, I'll turn them into nice interactive dataframes.
    """
    # I'll chop the text up wherever I see the <csv_table> tags.
    parts = re.split(r'(<csv_table>[\s\S]*?</csv_table>)', content)
    
    for part in parts:
        if "<csv_table>" in part:
            # Found a table! Let's clean up the tags and show it.
            csv_content = part.replace("<csv_table>", "").replace("</csv_table>", "").strip()
            try:
                df = pd.read_csv(StringIO(csv_content))
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"Could not render table: {e}")
                st.code(csv_content, language="csv")
        else:
            # Just regular text here.
            if part.strip():
                st.markdown(part)

# Title
st.title("📈 TrendScout AI")
st.markdown("### Market Intelligence & Trend Analysis Agent")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    st.info("Connected to Neo4j Aura & Local Vector Store")
    st.markdown("---")
    st.markdown("**Data Sources:**")
    st.markdown("- TechCrunch")
    st.markdown("- VentureBeat")
    st.markdown("- The Verge")
    st.markdown("- GitHub Jobs")

# I need to remember what we've talked about, so I'll initialize the chat history if it's empty.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Let's show the previous messages so the user doesn't feel like they're talking to a wall.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        render_content(message["content"])

# Waiting for the user to say something...
if prompt := st.chat_input("Ask about market trends, competitors, or SWOT analysis..."):
    # User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Now it's my turn. I'll think for a second (show a spinner) and then give the answer.
    with st.chat_message("assistant"):
        with st.spinner("Analyzing market data..."):
            try:
                # I'm calling the backend to do the heavy lifting (search, graph, LLM).
                result = backend.generate_answer(prompt, return_context=True)
                answer = result["answer"]
                
                # Time to show the result!
                render_content(answer)
                
                # I'll save what I said so it shows up next time.
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # For the curious users, I'll hide the raw data inside an expander.
                with st.expander("🔍 View Analysis Context"):
                    st.markdown(f"**Detected Entity:** `{result['entity_detected']}`")
                    st.markdown(f"**Intent:** `{result.get('intent', 'General')}`")
                    
                    st.markdown("### 🕸️ Knowledge Graph Insights")
                    if result['graph_context']:
                        st.info(result['graph_context'])
                    else:
                        st.warning("No direct graph connections found for this entity.")
                        
                    st.markdown("### 📄 Relevant Articles")
                    for doc in result['vector_context']:
                        st.markdown(f"- **{doc['source']}** ({doc.get('published', 'N/A')}): {doc['text'][:200]}...")
                        
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Footer
st.markdown("---")
st.caption("Powered by Llama 3.2, Neo4j, and Streamlit")

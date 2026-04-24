import streamlit as st
import os

# Set page config for aesthetics
st.set_page_config(page_title="Text-to-SQL Agent", page_icon="🤖", layout="centered")

# Custom CSS for styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 16px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)

# Securely grab the API key
api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))

if not api_key:
    st.error("Missing OpenAI API Key! Please set `OPENAI_API_KEY` in Streamlit secrets or environment variables.")
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

# We import the agent logic *after* ensuring the API key is in the environment
from agent_logic import run_text_to_sql_agent

st.title("🤖 Text-to-SQL AI Agent")
st.markdown("### Powered by LangGraph, DuckDB & Memento Strategy")
st.markdown("Convert natural language questions about HR Employee Attrition into executable SQL queries and actionable answers without writing code!")

st.divider()

user_query = st.text_input("Ask a business question:", placeholder="e.g. ¿Cuántos empleados son estrellas en ascenso?")

if st.button("Generate SQL & Run"):
    if user_query:
        with st.spinner("Analyzing request and building plan..."):
            try:
                result = run_text_to_sql_agent(user_query)
                st.success("Query processed successfully!")
                
                # Show results
                st.subheader("💡 Final Answer")
                st.markdown(result["answer"])
                
                with st.expander("🛠️ View Execution Plan"):
                    for step in result["steps"]:
                        st.write(f"- {step}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question first.")

st.divider()
st.caption("Developed by David Damian Arbeu.")

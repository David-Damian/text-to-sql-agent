"""
Text-to-SQL Agent — Streamlit MVP (Hybrid Mode)
  • Demo Mode: instant pre-computed results (no API key needed)
  • Live Mode: BYOK — visitor pastes their own OpenAI key
"""
import json
import os
import streamlit as st

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Text-to-SQL Agent · David Damian",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Header gradient */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0;
    }

    .sub-header {
        color: #8b8fa3;
        font-size: 1.05rem;
        margin-top: -8px;
        margin-bottom: 24px;
    }

    /* Mode badge */
    .mode-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .mode-demo {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
    }
    .mode-live {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
    }

    /* Result card */
    .result-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 12px;
        padding: 24px;
        margin-top: 16px;
    }

    /* Step pill */
    .step-pill {
        background: #313244;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.9rem;
        color: #cdd6f4;
        border-left: 3px solid #667eea;
    }

    /* Reward chip */
    .reward-chip {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .reward-good { background: rgba(16, 185, 129, 0.2); color: #10b981; }
    .reward-bad  { background: rgba(239, 68, 68, 0.2); color: #ef4444;  }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #181825;
    }

    /* Divider */
    .custom-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 24px 0;
    }

    /* Architecture diagram section */
    .arch-section {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 12px;
        padding: 20px;
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ── Load cached demos ────────────────────────────────────────
DEMOS_PATH = os.path.join(os.path.dirname(__file__), "..", "cached_demos", "demos.json")

@st.cache_data
def load_demos():
    if os.path.exists(DEMOS_PATH):
        with open(DEMOS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

demos = load_demos()


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    mode = st.radio(
        "Mode",
        ["🎬 Demo (pre-computed)", "⚡ Live (bring your own key)"],
        index=0,
        help="Demo mode uses cached results — no API key needed.",
    )
    is_live = "Live" in mode

    if is_live:
        user_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Your key is used only for this session and is never stored.",
        )
    else:
        user_api_key = None

    st.markdown("---")
    st.markdown("### 🏗️ Architecture")
    st.markdown("""
    ```
    ┌─────────────┐
    │  User Query  │
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │   Planner    │ ← Q-Network memory
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │  SQL Agent   │ ← search_schema()
    │  (Executor)  │ ← execute_sql()
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │  Evaluator   │ → reward → Q-Net update
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │   Answer     │
    └─────────────┘
    ```
    """)
    st.markdown("---")
    st.caption("Built by [David Damian](https://david-damian.github.io)")


# ── Header ───────────────────────────────────────────────────
st.markdown('<p class="main-header">Text-to-SQL Agent</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Ask business questions in natural language — '
    'get SQL-powered answers instantly.</p>',
    unsafe_allow_html=True,
)

badge_class = "mode-live" if is_live else "mode-demo"
badge_label = "LIVE MODE" if is_live else "DEMO MODE"
st.markdown(
    f'<span class="mode-badge {badge_class}">{badge_label}</span>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)


# ── Render a result ──────────────────────────────────────────
def render_result(result: dict):
    """Display answer, execution steps, and reward."""
    # Answer
    st.markdown("#### 💡 Answer")
    st.markdown(result["answer"])

    # Execution plan
    with st.expander("🛠️ Execution Plan", expanded=False):
        for step in result.get("steps", []):
            st.markdown(f'<div class="step-pill">{step}</div>', unsafe_allow_html=True)

    # Reward badge
    reward = result.get("reward", 0.0)
    chip_class = "reward-good" if reward >= 0.5 else "reward-bad"
    st.markdown(
        f'<span class="reward-chip {chip_class}">Evaluator Reward: {reward}</span>',
        unsafe_allow_html=True,
    )


# ── Demo Mode ────────────────────────────────────────────────
if not is_live:
    if not demos:
        st.warning(
            "No cached demos found. Run `python cache_demos.py` locally to generate them, "
            "or switch to Live mode."
        )
    else:
        query_options = [d["query"] for d in demos]
        selected_query = st.selectbox("Select a demo query:", query_options)

        if st.button("🚀 Show Result"):
            demo = next(d for d in demos if d["query"] == selected_query)
            render_result(demo)


# ── Live Mode ────────────────────────────────────────────────
else:
    user_query = st.text_input(
        "Enter your question:",
        placeholder="e.g. ¿Cuántos empleados son estrellas en ascenso?",
    )

    if st.button("🚀 Generate SQL & Run"):
        if not user_api_key:
            st.error("Please enter your OpenAI API Key in the sidebar.")
        elif not user_query:
            st.warning("Please type a question first.")
        else:
            with st.spinner("🧠 Planning → Executing SQL → Evaluating..."):
                try:
                    from agent_logic import initialize_agent, run_text_to_sql_agent

                    if "agent_ready" not in st.session_state:
                        initialize_agent(user_api_key)
                        st.session_state.agent_ready = True

                    result = run_text_to_sql_agent(user_query)
                    st.success("Query processed!")
                    render_result(result)
                except Exception as e:
                    st.error(f"Something went wrong: {e}")


# ── Footer ───────────────────────────────────────────────────
st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("📂 [Source Code](https://github.com/David-Damian/text-to-sql-agent)")
with col2:
    st.markdown("🌐 [Portfolio](https://david-damian.github.io)")
with col3:
    st.markdown("💼 [LinkedIn](https://linkedin.com/in/david-damian-arbeu)")

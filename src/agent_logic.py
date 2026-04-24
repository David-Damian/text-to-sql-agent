"""
Lazy-initialized Text-to-SQL Agent using LangGraph + Memento Strategy.
Imports safely without an API key; call initialize_agent(api_key) before running queries.
"""
import duckdb
import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import TypedDict, List

# Global handles — populated by initialize_agent()
con = None
llm_model = None
embeddings_model = None
schema_retriever = None
parametric_archivist = None
true_memento_app = None
_initialized = False


# ==========================================
# DB SETUP
# ==========================================
def _setup_db():
    global con
    con = duckdb.connect("rrhh_analytics.db")
    try:
        con.execute("INSTALL httpfs; LOAD httpfs;")
    except Exception:
        con.execute("LOAD httpfs;")
    con.execute("""
        CREATE OR REPLACE TABLE empleados AS
        SELECT * FROM read_csv_auto(
            'https://huggingface.co/spaces/Zhibekaa/female_male_salary_dep/raw/main/WA_Fn-UseC_-HR-Employee-Attrition.csv'
        );
    """)


# ==========================================
# NEURAL COMPONENTS
# ==========================================
class QNetwork(nn.Module):
    def __init__(self, embed_dim=1536):
        super().__init__()
        self.embed_dim = embed_dim
        self.fc1 = nn.Linear(embed_dim * 2, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, state_emb, case_emb):
        x = torch.cat([state_emb, case_emb], dim=-1)
        x = self.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))


class ParametricArchivist:
    """Experiential memory controller using Memento fine-tuning paradigm."""
    def __init__(self, embedder):
        self.embedder = embedder
        self.q_net = QNetwork()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=0.01)
        self.criterion = nn.BCELoss()
        self.case_bank = []

    def embed(self, text: str) -> torch.Tensor:
        vector = self.embedder.embed_query(text)
        return torch.tensor(vector, dtype=torch.float32)

    def retrieve_top_k(self, current_query: str, k: int = 2) -> str:
        if not self.case_bank:
            return "No hay casos previos en la memoria."
        state_emb = self.embed(current_query)
        scored_cases = []
        self.q_net.eval()
        with torch.no_grad():
            for case in self.case_bank:
                case_emb = self.embed(f"Task: {case['query']}\nPlan: {case['plan']}")
                q_value = self.q_net(state_emb, case_emb).item()
                scored_cases.append((q_value, case))
        scored_cases.sort(key=lambda x: x[0], reverse=True)
        top_k = scored_cases[:k]
        return "\n\n".join([
            f"[Q-Score: {s:.4f} | Reward Pasado: {c['reward']}]\n"
            f"Tarea: {c['query']}\nPlan: {c['plan']}\n"
            f"FEEDBACK: {c.get('feedback', 'N/A')}"
            for s, c in top_k
        ])

    def save_and_train(self, query: str, plan: str, reward: float, feedback: str):
        self.case_bank.append({
            'query': query, 'plan': plan, 'reward': reward, 'feedback': feedback
        })
        self.q_net.train()
        self.optimizer.zero_grad()
        state_emb = self.embed(query)
        case_emb = self.embed(f"Task: {query}\nPlan: {plan}")
        q_pred = self.q_net(state_emb, case_emb).squeeze()
        target_val = 1.0 if reward >= 0.5 else 0.0
        target = torch.tensor(target_val, dtype=torch.float32)
        loss = self.criterion(q_pred, target)
        loss.backward()
        self.optimizer.step()


# ==========================================
# STATE & GRAPH NODES
# ==========================================
class MementoState(TypedDict):
    user_query: str
    case_memory: str
    subtasks: List[str]
    final_answer: str
    reward: float


def _planner_node(state: MementoState):
    past_cases = parametric_archivist.retrieve_top_k(state['user_query'])
    prompt = f"""Eres el Planificador Memento. Descompón la tarea en pasos.
    === CASOS PREVIOS (Q-Valued) ===
    {past_cases}
    ================================
    INSTRUCCIÓN VITAL:
    1. Revisa el FEEDBACK (valor de los rewards) de los casos previos.
    2. El Ejecutor SQL NO tiene acceso a esta memoria. Por lo tanto, DEBES escribir las reglas matemáticas y filtros exactos en tu plan.
    Devuelve UNA LISTA NUMERADA de pasos explícitos."""

    from langchain_core.messages import HumanMessage, SystemMessage
    response = llm_model.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=state['user_query'])
    ])
    return {
        "subtasks": [step for step in response.content.split('\n') if step.strip()],
        "case_memory": past_cases
    }


def _executor_node(state: MementoState):
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent

    @tool
    def search_schema(query: str) -> str:
        """Searches the relevant schemas in SQL tables."""
        docs = schema_retriever.invoke(query)
        return "\n\n".join([
            f"Tabla: {d.metadata['table_name']}\n{d.page_content}" for d in docs
        ])

    @tool
    def execute_sql(sql_query: str) -> str:
        """Executes a SQL query in DuckDB."""
        try:
            result_df = con.execute(sql_query).df()
            return result_df.to_markdown(index=False) if not result_df.empty else "0 filas devueltas."
        except Exception as e:
            return f"Error SQL: {e}"

    plan_str = "\n".join(state['subtasks'])
    base_executor = create_react_agent(model=llm_model, tools=[search_schema, execute_sql])
    result = base_executor.invoke({
        "messages": [HumanMessage(
            content=f"Resuelve: '{state['user_query']}'.\nSIGUE ESTE PLAN ESTRICTAMENTE:\n{plan_str}"
        )]
    })
    return {"final_answer": result["messages"][-1].content}


def _evaluator_node(state: MementoState):
    from langchain_core.messages import HumanMessage
    prompt = f"""Eres un Evaluador Automático Estricto.
    Pregunta Original: {state['user_query']}
    Respuesta Generada: {state['final_answer']}
    Pasos Usados: {state['subtasks']}

    GLOSARIO ESTRICTO DE LA EMPRESA:
    - Flight Risk: PerformanceRating = 4 AND MonthlyIncome < 5000 AND TrainingTimesLastYear = 0
    - Rising Star: YearsAtCompany <= 2 AND PerformanceRating = 4 AND Attrition = 'No'
    - Stagnant Role: YearsInCurrentRole > 5 AND YearsSinceLastPromotion > 5

    REGLAS DE EVALUACIÓN:
    1. ASUME QUE LOS DATOS SON 100% CORRECTOS.
    2. EVALÚA SOLO LA LÓGICA de los pasos. Si incluyen los filtros correctos, REWARD = 1.0.
    3. MOTIVOS PARA FALLAR (REWARD 0.0): reglas distintas o glosario no aplicado.

    Responde EXACTAMENTE en dos líneas:
    REWARD: [0.0 o 1.0]
    FEEDBACK: [Explicación]"""

    texto = llm_model.invoke([HumanMessage(content=prompt)]).content.strip()
    reward, feedback = 0.0, "Sin feedback"
    for linea in texto.split('\n'):
        if linea.startswith("REWARD:"):
            try:
                reward = float(linea.replace("REWARD:", "").strip())
            except ValueError:
                pass
        elif linea.startswith("FEEDBACK:"):
            feedback = linea.replace("FEEDBACK:", "").strip()

    parametric_archivist.save_and_train(
        state['user_query'], "\n".join(state['subtasks']), reward, feedback
    )
    return {"reward": reward}


# ==========================================
# PUBLIC API
# ==========================================
def initialize_agent(api_key: str):
    """Set up all components. Must be called once before run_text_to_sql_agent()."""
    global llm_model, embeddings_model, schema_retriever
    global parametric_archivist, true_memento_app, _initialized

    os.environ["OPENAI_API_KEY"] = api_key

    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langgraph.graph import StateGraph, END

    _setup_db()

    llm_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    embeddings_model = OpenAIEmbeddings()

    schema_info = con.execute("DESCRIBE empleados").fetchall()
    column_descriptions = "\n".join([f"- {col[0]}: {col[1]}" for col in schema_info])
    table_schemas_docs = [
        Document(
            page_content=(
                f"Tabla: empleados\nEsquema Completo:\n{column_descriptions}\n"
                "Descripción: Datos de recursos humanos. Incluye salarios, "
                "satisfacción, rotación (Attrition) y métricas de desempeño."
            ),
            metadata={"table_name": "empleados", "source": "duckdb_local"}
        )
    ]
    schema_db = FAISS.from_documents(table_schemas_docs, embeddings_model)
    schema_retriever = schema_db.as_retriever(search_kwargs={"k": 3})

    parametric_archivist = ParametricArchivist(embedder=embeddings_model)

    workflow = StateGraph(MementoState)
    workflow.add_node("planner", _planner_node)
    workflow.add_node("executor", _executor_node)
    workflow.add_node("evaluator", _evaluator_node)
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "evaluator")
    workflow.add_edge("evaluator", END)
    true_memento_app = workflow.compile()

    _initialized = True
    print("✅ Agent initialized successfully.")


def run_text_to_sql_agent(query: str) -> dict:
    """Run the Memento agent on a query. Returns dict with 'answer' and 'steps'."""
    if not _initialized:
        raise RuntimeError("Agent not initialized. Call initialize_agent(api_key) first.")

    final_answer = ""
    subtasks = []
    reward = 0.0
    for event in true_memento_app.stream({"user_query": query}):
        if "planner" in event:
            subtasks = event["planner"]["subtasks"]
        if "executor" in event:
            final_answer = event["executor"]["final_answer"]
        if "evaluator" in event:
            reward = event["evaluator"]["reward"]
    return {"answer": final_answer, "steps": subtasks, "reward": reward}

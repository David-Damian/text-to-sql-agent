import duckdb
import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import TypedDict, List
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# Create global connection
con = duckdb.connect("rrhh_analytics.db")

def setup_db():
    try:
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute("""
            CREATE OR REPLACE TABLE empleados AS
            SELECT * FROM read_csv_auto('https://huggingface.co/spaces/Zhibekaa/female_male_salary_dep/raw/main/WA_Fn-UseC_-HR-Employee-Attrition.csv');
        """)
    except Exception as e:
        print(f"Error setting up DB: {e}")

setup_db()
schema_info = con.execute("DESCRIBE empleados").fetchall()

llm_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
embeddings_model = OpenAIEmbeddings()

# Vector DB for Schemas
column_descriptions = "\n".join([f"- {col[0]}: {col[1]}" for col in schema_info])
table_schemas_docs = [
    Document(
        page_content=f"Tabla: empleados\nEsquema Completo:\n{column_descriptions}\n"
                     "Descripción: Datos de recursos humanos. Incluye salarios, "
                     "satisfacción, rotación (Attrition) y métricas de desempeño.",
        metadata={"table_name": "empleados", "source": "duckdb_local"}
    )
]

schema_db = FAISS.from_documents(table_schemas_docs, embeddings_model)
schema_retriever = schema_db.as_retriever(search_kwargs={"k": 3})

class QNetwork(nn.Module):
    def __init__(self, embed_dim=1536):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim * 2, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, state_emb, case_emb):
        x = torch.cat([state_emb, case_emb], dim=-1)
        x = self.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))

class ParametricArchivist:
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
        if not self.case_bank: return "No hay casos previos en la memoria."
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
        return "\n\n".join([f"[Q-Score: {s:.4f} | Reward Pasado: {c['reward']}]\nTarea: {c['query']}\nPlan: {c['plan']}\nFEEDBACK: {c.get('feedback', 'N/A')}" for s, c in top_k])

    def save_and_train(self, query: str, plan: str, reward: float, feedback: str):
        self.case_bank.append({'query': query, 'plan': plan, 'reward': reward, 'feedback': feedback})
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

class MementoState(TypedDict):
    user_query: str; case_memory: str; subtasks: List[str]; final_answer: str; reward: float

parametric_archivist = ParametricArchivist(embedder=embeddings_model)

def true_planner_node(state: MementoState):
    past_cases = parametric_archivist.retrieve_top_k(state['user_query'])
    prompt = f"""Eres el Planificador Memento. Descompón la tarea en pasos.
    === CASOS PREVIOS (Q-Valued) ===
    {past_cases}
    ================================
    INSTRUCCIÓN VITAL:
    1. Revisa el FEEDBACK (valor de los rewards) de los casos previos.
    2. El Ejecutor SQL NO tiene acceso a esta memoria. Por lo tanto, DEBES escribir las reglas matemáticas y filtros exactos en tu plan.
    Devuelve UNA LISTA NUMERADA de pasos explícitos."""
    response = llm_model.invoke([SystemMessage(content=prompt), HumanMessage(content=state['user_query'])])
    return {"subtasks": [step for step in response.content.split('\n') if step.strip()], "case_memory": past_cases}

@tool
def search_schema(query: str) -> str:
    """Searches the relevant schemas in SQL tables."""
    docs = schema_retriever.invoke(query)
    return "\n\n".join([f"Tabla: {d.metadata['table_name']}\n{d.page_content}" for d in docs])

@tool
def execute_sql(sql_query: str) -> str:
    """Executes a SQL query in DuckDB."""
    try:
        result_df = con.execute(sql_query).df()
        return result_df.to_markdown(index=False) if not result_df.empty else "0 filas devueltas."
    except Exception as e:
        return f"Error SQL: {e}"

def memento_executor_node(state: MementoState):
    plan_str = "\n".join(state['subtasks'])
    base_executor = create_react_agent(model=llm_model, tools=[search_schema, execute_sql])
    result = base_executor.invoke({"messages": [HumanMessage(content=f"Resuelve: '{state['user_query']}'.\nSIGUE ESTE PLAN ESTRICTAMENTE:\n{plan_str}")]})
    return {"final_answer": result["messages"][-1].content}

def automated_evaluator_node(state: MementoState):
    prompt = f"""Eres un Evaluador Automático Estricto.
    Pregunta Original: {state['user_query']}
    Respuesta Generada: {state['final_answer']}
    Pasos Usados: {state['subtasks']}

    GLOSARIO ESTRICTO DE LA EMPRESA:
    - Flight Risk: PerformanceRating = 4 AND MonthlyIncome < 5000 AND TrainingTimesLastYear = 0
    - Rising Star: YearsAtCompany <= 2 AND PerformanceRating = 4 AND Attrition = 'No'
    - Stagnant Role: YearsInCurrentRole > 5 AND YearsSinceLastPromotion > 5

    REGLAS DE EVALUACIÓN VITALES (LEE ATENTAMENTE):
    1. ASUME QUE LOS DATOS SON 100% CORRECTOS: No tienes acceso a la base de datos.
    2. EVALÚA SOLO LA LÓGICA: Tu único trabajo es revisar los 'Pasos Usados'. Si los pasos incluyen los filtros matemáticas y lógicos exactos, el REWARD DEBE SER 1.0.
    3. MOTIVOS PARA FALLAR (REWARD 0.0): Si los pasos asumen reglas distintas, o si no aplicó el glosario.

    Responde EXACTAMENTE en dos líneas:
    REWARD: [0.0 o 1.0]
    FEEDBACK: [Explicación]"""
    texto = llm_model.invoke([HumanMessage(content=prompt)]).content.strip()
    reward, feedback = 0.0, "Sin feedback"
    for linea in texto.split('\n'):
        if linea.startswith("REWARD:"):
            try: reward = float(linea.replace("REWARD:", "").strip())
            except: pass
        elif linea.startswith("FEEDBACK:"): feedback = linea.replace("FEEDBACK:", "").strip()
    parametric_archivist.save_and_train(state['user_query'], "\n".join(state['subtasks']), reward, feedback)
    return {"reward": reward}

m_mdp_workflow = StateGraph(MementoState)
m_mdp_workflow.add_node("planner", true_planner_node)
m_mdp_workflow.add_node("executor", memento_executor_node)
m_mdp_workflow.add_node("evaluator", automated_evaluator_node)
m_mdp_workflow.set_entry_point("planner")
m_mdp_workflow.add_edge("planner", "executor")
m_mdp_workflow.add_edge("executor", "evaluator")
m_mdp_workflow.add_edge("evaluator", END)
true_memento_app = m_mdp_workflow.compile()

def run_text_to_sql_agent(query: str) -> dict:
    final_answer = ""
    subtasks = []
    for event in true_memento_app.stream({"user_query": query}):
        if "planner" in event:
            subtasks = event["planner"]["subtasks"]
        if "executor" in event:
            final_answer = event['executor']['final_answer']
    return {"answer": final_answer, "steps": subtasks}

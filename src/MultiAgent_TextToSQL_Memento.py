# !pip install -U langchain-openai langchain-community langgraph faiss-cpu openai duckdb pydantic torchview
"""
Module to build an agentic AI system that can answer questions about HR data
using Memento fine-tuning paradigm.

Tech stack: 
  - LangChain
  - LangGraph
  - FAISS
  - OpenAI
  - DuckDB
  - Pydantic
  - Torchview

Some test-questions to test the agent:

* "Cuantos empleados son estrellas en ascenso?" 

Sugerencia: Ejecuta el agente con la misma consulta dos veces para observar cómo la memoria 
y el aprendizaje reforzado mejoran el resultado.

DISCLAIMERS:
1. To run this script you will need an OpenAI API Key.
2. To run this script you will need to install the imported libraries
"""
import duckdb
import os
import getpass
import time
import torch
from torchview import draw_graph
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
# from google.colab import drive
# drive.mount('/content/drive')

def load_data_into_duckdb(db_path: str, repo_url: str, table_name: str, sanity_checks: bool = True):
    # 1. Connect to our local database
    con = duckdb.connect(db_path)

    # 2. Install and load the httpfs extension to handle URLs
    con.execute("INSTALL httpfs; LOAD httpfs;")

    # 3. Use the PERMANENT raw URL from the Space repository
    # This link points to the 'main' branch of the Space 'Zhibekaa/female_male_salary_dep'
    con.execute(f"""
        CREATE OR REPLACE TABLE {table_name} AS
    SELECT * FROM read_csv_auto('{repo_url}');
    """)
    if sanity_checks: 
        # 4. Sanity checks

        result = con.execute("SELECT count(*) FROM empleados").fetchone()
        print(f"Success! Loaded {result[0]} rows from the Space.")
        # 4.1 Verifyng if duplicates
        query = """
        SELECT EmployeeNumber, COUNT(*) as occurrence_count
        FROM empleados
        GROUP BY EmployeeNumber
        HAVING COUNT(*) = 1
        """
        con.execute(query).df()

load_data_into_duckdb("rrhh_analytics.db", 
                      "https://huggingface.co/spaces/Zhibekaa/female_male_salary_dep/raw/main/WA_Fn-UseC_-HR-Employee-Attrition.csv", 
                      "empleados")


# ==========================================
# 1. SETUP & API
# ==========================================
os.environ["OPENAI_API_KEY"] = getpass.getpass("Introduce tu OpenAI API Key: ")

llm_model = ChatOpenAI(model="gpt-4.1-mini", temperature=0.1)
embeddings_model = OpenAIEmbeddings()

# Vector DB for Schemas
# Get the schema of the SQL table we prevously built.
schema_info = con.execute("DESCRIBE empleados").fetchall()

# Creamos un string con el formato "Columna (Tipo)"
column_descriptions = "\n".join([f"- {col[0]}: {col[1]}" for col in schema_info])
table_schemas_docs = [
    Document(
    page_content=f"Tabla: empleados\nEsquema Completo:\n{column_descriptions}\n"
                 "Descripción: Datos de recursos humanos. Incluye salarios, "
                 "satisfacción, rotación (Attrition) y métricas de desempeño.",
    metadata={"table_name": "empleados", "source": "duckdb_local"}
)
]
# Get the schemas, get its OpenAI embeddings reprfesentations and save it in an optimized data structure for quick searches.
schema_db = FAISS.from_documents(table_schemas_docs, embeddings_model)
# The method as_retriever: Transforms the DB into a tool that helps answer questions
schema_retriever = schema_db.as_retriever(search_kwargs={"k": 3})

class QNetwork(nn.Module):
  """
  Q-network that will be used to estimate the value of a state-action pair
  and will help the AI Agent planner to enhance its behavior.
  """
  def __init__(self, embed_dim=1536):
      super().__init__()
      self.fc1 = nn.Linear(embed_dim * 2, 512)
      self.relu = nn.ReLU()
      self.fc2 = nn.Linear(512, 1)
      self.sigmoid = nn.Sigmoid()

  def forward(self, state_emb, case_emb):
    """
    Forward pass method to compute the Q-value for a given
    state (user query) - action (which experience retrive from case bank) pair.
    """
    x = torch.cat([state_emb, case_emb], dim=-1)
    x = self.relu(self.fc1(x))
    return self.sigmoid(self.fc2(x))

  def plot_nn_architecture(self, batch_size=1):
    """
    Generates and returns the visual graph of the network architecture.
    """
    # 1. Generate dummy inputs based on the expected embed_dim
    dummy_state = torch.randn(batch_size, self.embed_dim)
    dummy_case = torch.randn(batch_size, self.embed_dim)

    # 2. Draw the graph passing 'self' as the model
    model_graph = draw_graph(
        self,
        input_data=(dummy_state, dummy_case),
        expand_nested=True
    )

    # 3. Return the visual graph object so it renders in a Notebook
    return model_graph.visual_graph

class ParametricArchivist:
  """
    An experiential memory controller based on the Memento fine-tuning paradigm.

    This class implements a 'Neural Archivist' that manages a Case Bank $\mathcal{M}$
    using a Parametric Q-Network to optimize memory retrieval. Instead of relying
    solely on semantic similarity (e.g., Cosine Similarity), it learns a scoring
    function $Q(s, m) \to [0, 1]$ that predicts the utility of a past memory $m$
    for the current state $s$.

    ### Theoretical Framework:
    * **State ($s$):** The embedding of the current user query.
    * **Action ($a$):** The selection of a specific case $c_i$ from the bank $\mathcal{M}$.
    * **Reward ($R$):** A binary signal $\in \{0, 1\}$ derived from the success
        threshold of the agent's performance (e.g., $R=1$ if $reward \geq 0.5$).
    * **Optimization:** Minimizes the Binary Cross Entropy (BCE) loss between
        the predicted utility and the ground-truth outcome:
        $$\mathcal{L} = -[y \log(Q(s, m)) + (1 - y) \log(1 - Q(s, m))]$$

    Attributes:
        embedder: The encoder model used to map text into the vector space $\mathbb{R}^d$.
        q_net: A MLP that approximates the Q-value for state-memory pairs.
        case_bank: A non-parametric repository of historical trajectories
                   $\tau = (q, p, r, f)$.
    """
  def __init__(self, embedder):
    self.embedder = embedder
    self.q_net = QNetwork()
    self.optimizer = optim.Adam(self.q_net.parameters(), lr=0.01)
    self.criterion = nn.BCELoss()
    # Bank of past experiences M = {c_i: c_i = (s_i, a_i, r_i) = (user_query_i, executor_made_action_i, estimated_reward_for_experience_i)}
    self.case_bank = []

  def embed(self, text: str) -> torch.Tensor:
    """
    Embed a text using the provided embedder.
    """
    vector = self.embedder.embed_query(text)
    return torch.tensor(vector, dtype=torch.float32)

  def retrieve_top_k(self, current_query: str, k: int = 2) -> str:
    """
    Retrieve the top k cases with highest Q-value from the case bank
    that are most similar to the current query.
    """
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

    print("\n🔍 [Motor de Recuperación Q-Learning]:")
    for score, c in top_k:
        print(f"   -> Q-Value Estimado: {score:.4f} | Tarea: '{c['query']}'")

    return "\n\n".join([f"""[Q-Score: {s:.4f} | Reward Pasado: {c['reward']}]\n
                          Tarea: {c['query']}\n
                          Plan: {c['plan']}\n
                          FEEDBACK: {c.get('feedback', 'N/A')}""" for s, c in top_k])

  def save_and_train(self, query: str, plan: str, reward: float, feedback: str):
    """
    Append new case to the case bank and train the Q-network.
    """
    self.case_bank.append({'query': query, 'plan': plan, 'reward': reward, 'feedback': feedback})
    self.q_net.train()
    self.optimizer.zero_grad()

    state_emb = self.embed(query)
    case_emb = self.embed(f"Task: {query}\nPlan: {plan}")

    q_pred = self.q_net(state_emb, case_emb).squeeze()
    # Make decision if the computed probability, r, for the Q Network outputs r >= 0.5
    target_val = 1.0 if reward >= 0.5 else 0.0
    target = torch.tensor(target_val, dtype=torch.float32)

    loss = self.criterion(q_pred, target)
    loss.backward()
    self.optimizer.step()
    print(f"🧠 [Red Q Actualizada] Predicción Inicial: {q_pred.item():.4f} | Pérdida (Loss): {loss.item():.4f}")

class MementoState(TypedDict):
  """
  Global state of the Memento agent.
  """
  user_query: str; case_memory: str; subtasks: List[str]; final_answer: str; reward: float

def true_planner_node(state: MementoState):
  """
  AI Agent that builds a plan for the user query
  using the case bank, the Q-network values and the LLM model.
  """
  past_cases = parametric_archivist.retrieve_top_k(state['user_query'])

  # SYSTEM PROMPT PARA EL PLANNER: Forzamos al planificador a traducir el feedback en reglas SQL literales.
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

# ==========================================
# 3. GLOBAL TOOLS
# ==========================================
@tool
def search_schema(query: str) -> str:
    """Searches the relevant schemas in SQL tables."""
    # 1. Convertir query en un vector numérico usando los embeddings de la base de datos.
    # 2. Busca3 dentro de FAISS los esquemas de tablas que tengan el significado más cercano a esa pregunta.
    docs = schema_retriever.invoke(query)
    return "\n\n".join([f"Tabla: {d.metadata['table_name']}\n{d.page_content}" for d in docs])

@tool
def execute_sql(sql_query: str) -> str:
    """
    Executes a SQL query in DuckDB.
    """
    try:
        result_df = con.execute(sql_query).df()
        return result_df.to_markdown(index=False) if not result_df.empty else "0 filas devueltas."
    except Exception as e:
        return f"Error SQL: {e}"

def memento_executor_node(state: MementoState):
  """
  Given the planner outputs (subtaks), executes
  each subtask and returns the final answer.
  """
  plan_str = "\n".join(state['subtasks'])
  # Shared executor tool
  base_executor = create_react_agent(model=llm_model,
                                      tools=[search_schema, execute_sql]
                                      )
  result = base_executor.invoke({"messages": [HumanMessage(content=f"Resuelve: '{state['user_query']}'.\nSIGUE ESTE PLAN ESTRICTAMENTE:\n{plan_str}")]})
  return {"final_answer": result["messages"][-1].content}

def automated_evaluator_node(state: MementoState):
    # PROMPT CORREGIDO: Forzamos al evaluador a no verificar los datos, sino la lógica.
    prompt = f"""Eres un Evaluador Automático Estricto.
    Pregunta Original: {state['user_query']}
    Respuesta Generada: {state['final_answer']}
    Pasos Usados: {state['subtasks']}

    GLOSARIO ESTRICTO DE LA EMPRESA:
    - Flight Risk: PerformanceRating = 4 AND MonthlyIncome < 5000 AND TrainingTimesLastYear = 0
    - Rising Star: YearsAtCompany <= 2 AND PerformanceRating = 4 AND Attrition = 'No'
    - Stagnant Role: YearsInCurrentRole > 5 AND YearsSinceLastPromotion > 5

    REGLAS DE EVALUACIÓN VITALES (LEE ATENTAMENTE):
    1. ASUME QUE LOS DATOS SON 100% CORRECTOS: No tienes acceso a la base de datos. Confía en que los nombres (ej. Belén, Ana, Winter Coat) y los números devueltos en la 'Respuesta Generada' son reales. NO castigues a la respuesta por los datos.
    2. EVALÚA SOLO LA LÓGICA: Tu único trabajo es revisar los 'Pasos Usados'. Si los pasos incluyen los filtros matemáticos y lógicos exactos del glosario (ej. "stock_quantity > 0 AND is_active = FALSE" o "> 2000"), el REWARD DEBE SER 1.0.
    3. MOTIVOS PARA FALLAR (REWARD 0.0): Si los pasos asumen reglas distintas (ej. asumir que Star es PerformanceRating !=4), si la respuesta pide aclaraciones al usuario, o si no aplicó el glosario.

    Responde EXACTAMENTE en dos líneas:
    REWARD: [0.0 o 1.0]
    FEEDBACK: [Explicación concisa y directa citando la regla del glosario. Si es 1.0, felicita la lógica usada en los pasos."""

    texto = llm_model.invoke([HumanMessage(content=prompt)]).content.strip()
    reward, feedback = 0.0, "Sin feedback"

    for linea in texto.split('\n'):
        if linea.startswith("REWARD:"):
            try: reward = float(linea.replace("REWARD:", "").strip())
            except: pass
        elif linea.startswith("FEEDBACK:"): feedback = linea.replace("FEEDBACK:", "").strip()

    print(f"⚖️ [Auto-Evaluador] Reward: {reward} | Feedback: {feedback}")
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

# from IPython.display import Image, display
# Generate the graph and display it in your notebook
# display(Image(true_memento_app.get_graph().draw_mermaid_png()))

def run_true_memento(query: str):
    print(f"\n🗣️ [Usuario]: {query}")
    for event in true_memento_app.stream({"user_query": query}):
        if "planner" in event:
            print("\n📝 [Planificador] Plan Generado:")
            for step in event["planner"]["subtasks"]: print(f"   {step}")
        if "executor" in event:
            print(f"\n⚙️ [Ejecutor SQL] Resultado Final:\n{event['executor']['final_answer']}")

parametric_archivist = ParametricArchivist(embedder = embeddings_model)

# ==========================================
# SECUENCIA DE PRUEBA (4 ITERACIONES)
# ==========================================

print("\n--- INTENTO 1 Rising stars---")
run_true_memento("Cuantos empleados son estrellas en ascenso?")

time.sleep(2)
print("\n--- INTENTO 2 Rising stars ---")
run_true_memento("Quienes son los empleados que son estrellas en ascenso?")

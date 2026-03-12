import sys
import time

sys.path.append("../src/")
from auxiliar_functions import *
import pandas as pd 
import numpy as np
import os
import textwrap
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
# ==================
# NLP libraries
# ==================
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from collections import Counter
# Stemming (Spanish)
from nltk.stem.snowball import SnowballStemmer
import re
import unicodedata
from textwrap import shorten
# NER
import spacy
# ==================
# GenAI libraries
# ==================
from google import genai
from google.genai import types
import openai
from openai import OpenAI

""" 
Script to analyze via LDA, the treding topics in conversations with
AI assistant for technical help.
"""

# Get the Spanish stop words as a list
spanish_stopwords = stopwords.words('spanish')
english_stopwords = stopwords.words('english')

processed_data= pd.read_csv("../data/processed_conversations.csv")
exploded_data = pd.read_csv("../data/processed_exploded_conversations.csv")

# After loading CSVs
print("=" * 60)
print("📥 STAGE 1: DATA LOADING")
print("=" * 60)
print(f"  processed_data shape:  {processed_data.shape}")
print(f"  exploded_data shape:   {exploded_data.shape}")
print(f"  Columns processed:     {list(processed_data.columns)}")

# Filter
spanish_convs = processed_data[processed_data['language'] == 'es']
portuguese_convs = processed_data[processed_data['language'] == 'pt']
english_convs = processed_data[processed_data['language'] == 'en']


# After language filtering
print(f"\n🌐 Language distribution:")
print(f"  Spanish (es):     {len(spanish_convs):,d} conversations")
print(f"  Portuguese (pt):  {len(portuguese_convs):,d} conversations")
print(f"  English (en):     {len(english_convs):,d} conversations")
print(f"  Other:            {len(processed_data) - len(spanish_convs) - len(portuguese_convs) - len(english_convs):,d} conversations")

# After user message extraction
user_messages_es = exploded_data[exploded_data['message_role'] == 'user'].merge(spanish_convs[['CONVERSATION_ID']], how = 'inner', on = 'CONVERSATION_ID')
user_messages_es_, user_messages_es_ids = user_messages_es['message_text'], user_messages_es['CONVERSATION_ID']
print(f"\n👤 User messages (Spanish):")
print(f"  Total user messages:   {len(user_messages_es):,d}")
print(f"  Unique conversations:  {user_messages_es['CONVERSATION_ID'].nunique():,d}")
print(f"  Avg msgs/conversation: {user_messages_es.groupby('CONVERSATION_ID').size().mean():.1f}")
docs_raw = {
    "conv_ids": user_messages_es_ids.tolist(),
    "texts": user_messages_es_.astype(str).tolist()
}
print("Docs (raw):", len(docs_raw["texts"]))


# -----------------------
# 3) Preprocessing (standard NLP pipeline + stemming)
#    clean -> normalize -> tokenize -> stopwords -> stem -> re-join
# -----------------------
def strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

# Spanish stemmer (simple + no extra downloads)
stemmer = SnowballStemmer("spanish")
# Domain-specific stopwords for a tech-support chatbot
DOMAIN_STOPWORDS = {
    "hola", "gracias", "favor", "ayuda", "necesito", "quiero", 
    "puedes", "podrias", "tengo", "hacer", "buenos", "dias", 
    "buenas", "tardes", "noches", "por", "como", "puede",
    "seria", "posible", "quisiera", "gustaria"
}
STOPWORDS_ES = set(spanish_stopwords) | DOMAIN_STOPWORDS
def preprocess(text: str) -> str:
    # A) CLEANING (remove obvious noise)
    text = (text or "").lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # remove URLs

    # B) NORMALIZATION (accent folding)
    text = strip_accents(text)  # "qué" -> "que"

    # C) TOKENIZATION (letters only; after accent stripping)
    tokens = re.findall(r"[a-zñ]+", text)

    # D) FILTERING (stopwords + short tokens)
    tokens = [t for t in tokens if len(t) >= 3 and t not in STOPWORDS_ES]

    # E) STEMMING (reduce words to their root)
    tokens = [stemmer.stem(t) for t in tokens]

    # Return a string for CountVectorizer
    return " ".join(tokens)

# 1. Crear el diccionario de limpios (añadida la coma faltante)
docs_clean = {
    'conv_ids': docs_raw['conv_ids'], # Los IDs no cambian
    'texts': [preprocess(t) for t in docs_raw['texts']]
}
# 2. Alineación y Filtrado (Agrupamos ID, Raw y Clean en un solo iterable)
# Usamos docs_raw['texts'] para el Raw y docs_clean['texts'] para el Clean
triplets = [
    (conv_id, raw, clean) 
    for conv_id, raw, clean in zip(docs_raw['conv_ids'], docs_raw['texts'], docs_clean['texts'])
    if clean.strip() # Solo mantenemos si el texto limpio NO está vacío
]
# 3. Desempaquetar (Unzip) de vuelta a estructuras limpias
if triplets:
    ids_f, raw_f, clean_f = zip(*triplets)
    
    # Actualizamos docs_raw y docs_clean con los datos filtrados
    docs_raw = {'conv_ids': list(ids_f), 'texts': list(raw_f)}
    docs_clean = {'conv_ids': list(ids_f), 'texts': list(clean_f)}
else:
    docs_raw = {'conv_ids': [], 'texts': []}
    docs_clean = {'conv_ids': [], 'texts': []}
print("Docs (clean, non-empty):", len(docs_clean['texts']))
print("Top tokens after preprocessing:", Counter(" ".join(docs_clean['texts']).split()).most_common(20))

# After preprocessing
print("\n" + "=" * 60)
print("🧹 STAGE 2: PREPROCESSING")
print("=" * 60)
print(f"  Docs before cleaning: {len(docs_raw['texts']):,d}")
print(f"  Docs after cleaning:  {len(docs_clean['texts']):,d}")
print(f"  Docs removed (empty): {len(docs_raw['texts']) - len(docs_clean['texts']):,d} "
      f"({(len(docs_raw['texts']) - len(docs_clean['texts'])) / max(len(docs_raw['texts']), 1) * 100:.1f}%)")

# 2. Vectorize (removing common stopwords)
vectorizer = CountVectorizer(max_df=0.9, 
                            min_df=2, 
                            stop_words=spanish_stopwords + english_stopwords) # Add Spanish/Portuguese too
X = vectorizer.fit_transform(docs_clean['texts'])
vocab = np.array(vectorizer.get_feature_names_out())
print("Vocab size:", len(vocab))
n_topics = 6
# 3. Apply LDA
lda = LatentDirichletAllocation(n_components=n_topics, 
                                random_state=42)
lda.fit(X)

# -----------------------
# 4) Topics: print + DataFrame of top words per topic
# -----------------------
def show_topics(model, vocab, top_n=12):
    for k, weights in enumerate(model.components_):
        top_idx = np.argsort(weights)[::-1][:top_n]
        top_terms = [vocab[i] for i in top_idx]
        print(f"Topic {k}: {', '.join(top_terms)}")

def topics_dataframe(model, vocab, top_n=12):
    rows = []
    for topic_id, weights in enumerate(model.components_):
        top_idx = np.argsort(weights)[::-1][:top_n]
        for rank, i in enumerate(top_idx, start=1):
            rows.append({
                "topic_id": topic_id,
                "rank": rank,
                "term": vocab[i],
                "weight": float(weights[i]),
            })
    return pd.DataFrame(rows)

show_topics(lda, vocab, top_n=12)

df_topics = topics_dataframe(lda, vocab, top_n=12)

# -----------------------
# 7) Documents: DataFrame with topic assignment + probabilities
# -----------------------
doc_topic = lda.transform(X)
top_topic = doc_topic.argmax(axis=1)
top_conf = doc_topic.max(axis=1)

df_docs = pd.DataFrame({
    "conv_id": docs_raw['conv_ids'],
    "text_raw": [shorten(t, width=160, placeholder="…") for t in docs_raw['texts']],
    "text_clean": [shorten(t, width=160, placeholder="…") for t in docs_clean['texts']],
    "top_topic": top_topic,
    "top_topic_conf": np.round(top_conf, 3),
})
for k in range(n_topics):
    df_docs[f"topic_{k}_prob"] = np.round(doc_topic[:, k], 3)

client = genai.Client(api_key=os.environ.get("GOOGLE_GENAI_API_KEY", ""))
MODEL_ID = "gemini-2.5-flash-lite"  # much higher free-tier quota
# @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-3.1-flash-lite-preview", "gemini-3-flash-preview", "gemini-3.1-pro-preview"] {"allow-input":true, isTemplate: true}
model_info = client.models.get(model=MODEL_ID)

print("Context window:",model_info.input_token_limit, "tokens")
print("Max output window:",model_info.output_token_limit, "tokens")

topic_names = {}
# Configuración corregida para el nuevo SDK (google-genai)
safe_config = types.GenerateContentConfig(
    max_output_tokens=60,
    safety_settings=[
        types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH", 
            threshold="BLOCK_NONE"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT", 
            threshold="BLOCK_NONE"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT", 
            threshold="BLOCK_NONE"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT", 
            threshold="BLOCK_NONE"
        ),
    ]
)


def call_genai_with_retry(client, model_id, prompt, config, max_retries=3):
    """Llama a Gemini con reintentos y backoff exponencial para manejar rate limits."""
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=config
            )
            return resp
        except Exception as e:
            wait_time = 15 * (2 ** attempt)  # 15s, 30s, 60s
            print(f"    ⚠️ Error en intento {attempt + 1}/{max_retries}: {type(e).__name__}")
            if attempt < max_retries - 1:
                print(f"    ⏳ Reintentando en {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"    ❌ Todos los reintentos fallaron.")
                return None

for t in range(n_topics):
    print(f'Procesando tema {t}...')
    qs = (
        df_docs[df_docs["top_topic"] == t]
        .nlargest(10, "top_topic_conf")
        ["text_raw"]
        .tolist()
    )
    if not qs:
        topic_names[t] = f"Tema {t}"
        continue

    # Incluir las top words del tema para mejor contexto
    top_words = df_topics[df_topics['topic_id'] == t].nlargest(10, 'weight')['term'].tolist()

    prompt = (
        "Estas son las palabras clave y preguntas más representativas de un tema "
        "en un asistente conversacional de IA para consultas técnicas.\n\n"
        f"PALABRAS CLAVE: {', '.join(top_words)}\n\n"
        "PREGUNTAS REPRESENTATIVAS:\n"
        + "\n".join([f"- {q[:160]}" for q in qs]) +
        "\n\nDame un título descriptivo en español de 2 a 4 palabras. "
        "Incluye el lenguaje de programación si aplica.\n"
        "EJEMPLOS: \"Depuración de código Python\", \"Consultas de SQL\", \"Soporte de API\".\n"
        "RESPUESTA (SOLO EL TÍTULO):"
    )

    resp = call_genai_with_retry(client, MODEL_ID, prompt, safe_config)
    
    if resp is None:
        topic_names[t] = f"Tema {t} (sin etiquetar)"
        print(f"  Topic {t}: {topic_names[t]}")
        continue

    if resp.candidates[0].finish_reason == "SAFETY":
        print(f"  ⚠️ El Tema {t} fue bloqueado por seguridad.")
        print(f"  Calificaciones: {resp.candidates[0].safety_ratings}")
    topic_names[t] = (resp.text or "").strip().strip('"') or f"Tema {t}"
    print(f"  Topic {t}: {topic_names[t]}")
    
    # Pausa entre llamadas para respetar rate limits
    time.sleep(4)

# Final summary
print("\n" + "=" * 60)
print("🏷️ FINAL TOPIC NAMES")
print("=" * 60)
for t, name in topic_names.items():
    count = (top_topic == t).sum()
    print(f"  Topic {t}: {name:30s} ({count:,d} docs)")


# ================================================================
# 🔗 ANÁLISIS 6: CRUCE LDA × FLUJO DE CONVERSACIÓN
#    El análisis más accionable: ¿qué temas causan problemas de routing?
# ================================================================
print("\n" + "=" * 60)
print("🔗 STAGE 7: CRUCE LDA × FLUJO DE CONVERSACIÓN")
print("=" * 60)

# 1. Obtener el tema dominante por conversación (agrupando mensajes)
conv_topics = (df_docs.groupby('conv_id')
               .agg(
                   dominant_topic=('top_topic', lambda x: x.mode().iloc[0]),
                   avg_confidence=('top_topic_conf', 'mean'),
                   n_messages=('conv_id', 'size')
               )
               .reset_index())

print(f"\n  📊 Conversaciones con tema asignado: {len(conv_topics):,d}")
print(f"  Confianza promedio: {conv_topics['avg_confidence'].mean():.3f}")

# 2. Unir con metadata de flujo (processed_data)
analysis = processed_data.merge(
    conv_topics, 
    left_on='CONVERSATION_ID', 
    right_on='conv_id', 
    how='inner'
)
print(f"  Conversaciones cruzadas exitosamente: {len(analysis):,d}")

# ----- A) Duración promedio por tema LDA -----
print("\n  ⏱️ DURACIÓN PROMEDIO POR TEMA:")
print(f"  {'Tema':<6} {'Nombre':<32} {'Media(s)':>10} {'Mediana(s)':>10} {'N':>8}")
print(f"  {'-'*6} {'-'*32} {'-'*10} {'-'*10} {'-'*8}")

topic_duration = (analysis.groupby('dominant_topic')['CONV_SECONDS']
                  .agg(['mean', 'median', 'count'])
                  .sort_values('mean', ascending=False))

for topic_id, row in topic_duration.iterrows():
    name = topic_names.get(topic_id, f"Topic {topic_id}")
    print(f"  {topic_id:<6} {name:<32} {row['mean']:>10.0f} {row['median']:>10.0f} {int(row['count']):>8,d}")

# ----- B) Tasa de drop-off por tema -----
print("\n  🔚 TASA DE DROP-OFF POR TEMA (conversaciones con ≤1 mensaje del usuario):")
analysis['is_dropoff'] = analysis['user_message_count'] <= 1

topic_dropoff = (analysis.groupby('dominant_topic')
                 .agg(
                     dropoff_rate=('is_dropoff', 'mean'),
                     total=('is_dropoff', 'size'),
                     dropoffs=('is_dropoff', 'sum')
                 )
                 .sort_values('dropoff_rate', ascending=False))

print(f"  {'Tema':<6} {'Nombre':<32} {'Drop-off%':>10} {'Drops':>8} {'Total':>8}")
print(f"  {'-'*6} {'-'*32} {'-'*10} {'-'*8} {'-'*8}")
for topic_id, row in topic_dropoff.iterrows():
    name = topic_names.get(topic_id, f"Topic {topic_id}")
    bar = "█" * int(row['dropoff_rate'] * 20)
    print(f"  {topic_id:<6} {name:<32} {row['dropoff_rate']*100:>9.1f}% {int(row['dropoffs']):>8,d} {int(row['total']):>8,d}  {bar}")

# ----- C) Distribución de Skills por Tema -----
print("\n  🎯 DISTRIBUCIÓN DE SKILLS POR TEMA:")
topic_skill = pd.crosstab(
    analysis['dominant_topic'].map(lambda x: f"T{x}: {topic_names.get(x, '?')[:20]}"),
    analysis['skill_name'],
    normalize='index'
) * 100

# Mostrar solo skills con >5% en algún tema
significant_skills = topic_skill.columns[topic_skill.max() > 5]
print(topic_skill[significant_skills].round(1).to_string())

# ----- D) Nodos terminales por tema (¿dónde se quedan?) -----
print("\n  📍 TOP NODOS TERMINALES (LAST_NODE) POR TEMA:")
for topic_id in range(n_topics):
    topic_data = analysis[analysis['dominant_topic'] == topic_id]
    name = topic_names.get(topic_id, f"Topic {topic_id}")
    top_nodes = topic_data['LAST_NODE_ID'].value_counts().head(3)
    nodes_str = " | ".join([f"{node}: {count:,d}" for node, count in top_nodes.items()])
    print(f"  Tema {topic_id} ({name[:25]:25s}): {nodes_str}")

# ----- E) Conversaciones con baja confianza (ambiguas) -----
AMBIGUOUS_THRESHOLD = 0.4
ambiguous = analysis[analysis['avg_confidence'] < AMBIGUOUS_THRESHOLD]
print(f"\n  ⚠️ CONVERSACIONES AMBIGUAS (confianza < {AMBIGUOUS_THRESHOLD}):")
print(f"  Total: {len(ambiguous):,d} ({len(ambiguous)/len(analysis)*100:.1f}%)")
if len(ambiguous) > 0:
    print(f"  Skills en conversaciones ambiguas:")
    for skill, count in ambiguous['skill_name'].value_counts().head(5).items():
        print(f"    {skill}: {count:,d}")
    print(f"  Duración promedio (ambiguas): {ambiguous['CONV_SECONDS'].mean():.0f}s "
          f"vs. {analysis['CONV_SECONDS'].mean():.0f}s (general)")

# ----- F) Resumen accionable -----
print("\n" + "=" * 60)
print("📋 RESUMEN ACCIONABLE")
print("=" * 60)

# Tema con mayor drop-off
worst_dropoff = topic_dropoff['dropoff_rate'].idxmax()
worst_dropoff_name = topic_names.get(worst_dropoff, f"Topic {worst_dropoff}")
worst_dropoff_rate = topic_dropoff.loc[worst_dropoff, 'dropoff_rate']

# Tema con mayor duración
longest_topic = topic_duration['mean'].idxmax()
longest_name = topic_names.get(longest_topic, f"Topic {longest_topic}")
longest_duration = topic_duration.loc[longest_topic, 'mean']

print(f"\n  🔴 Tema con MAYOR DROP-OFF: Tema {worst_dropoff} ({worst_dropoff_name})")
print(f"     → {worst_dropoff_rate*100:.1f}% de usuarios abandonan con ≤1 mensaje")
print(f"     → Revisar la respuesta inicial del agente para este tipo de consultas")

print(f"\n  🟠 Tema con MAYOR DURACIÓN: Tema {longest_topic} ({longest_name})")
print(f"     → Duración promedio: {longest_duration:.0f}s ({longest_duration/60:.1f} min)")
print(f"     → Posible problema de routing: el agente no resuelve eficientemente")

# ================================================================
# 🏷️ STAGE 8: NER (Named Entity Recognition) × LDA
#    Extraer entidades específicas y cruzarlas con los temas
# ================================================================
print("\n" + "=" * 60)
print("🏷️ STAGE 8: NER × LDA (Named Entity Recognition)")
print("=" * 60)

# Cargar modelo de spaCy en español
print("\n  Cargando modelo spaCy es_core_news_sm...")
nlp_spacy = spacy.load("es_core_news_sm")
print("  ✅ Modelo cargado.")

# --- Diccionario de entidades de dominio ---
# Estas son tecnologías/herramientas/plataformas que spaCy no detectará
# pero que son CLAVE para el análisis de un agente conversacional técnico
TECH_ENTITIES = {
    # Lenguajes de programación
    "python", "javascript", "typescript", "java", "sql", "html", "css",
    "php", "ruby", "golang", "rust", "swift", "kotlin", "scala",
    # Plataformas / Herramientas
    "n8n", "zapier", "make", "docker", "kubernetes", "github", "gitlab",
    "jira", "slack", "notion", "figma", "postman",
    # APIs / Servicios
    "whatsapp", "telegram", "facebook", "instagram", "messenger",
    "openai", "chatgpt", "gpt", "gemini", "claude",
    "aws", "azure", "firebase", "supabase", "vercel", "heroku",
    # Frameworks
    "react", "angular", "vue", "nextjs", "django", "flask", "fastapi",
    "nodejs", "express", "laravel", "spring",
    # Bases de datos
    "mysql", "postgres", "postgresql", "mongodb", "redis", "sqlite",
    "snowflake", "bigquery",
    # Otros
    "api", "rest", "graphql", "webhook", "json", "csv", "xml",
}

def extract_entities(text: str, nlp_model) -> dict:
    """
    Extrae entidades nombradas de un texto usando spaCy + diccionario de dominio.
    
    Returns:
        dict con 'spacy_entities' (list de tuples) y 'tech_entities' (list de str)
    """
    if not isinstance(text, str) or len(text.strip()) < 3:
        return {"spacy_entities": [], "tech_entities": []}
    
    # A) Entidades de spaCy (PER, ORG, LOC, MISC)
    doc = nlp_model(text[:500])  # Limitar texto para rendimiento
    spacy_ents = [(ent.text, ent.label_) for ent in doc.ents]
    
    # B) Entidades de dominio (tecnologías, herramientas)
    text_lower = text.lower()
    # Buscar con word boundaries para evitar falsos positivos
    tech_found = []
    for tech in TECH_ENTITIES:
        # Usar regex para match de palabra completa
        if re.search(r'\b' + re.escape(tech) + r'\b', text_lower):
            tech_found.append(tech)
    
    return {
        "spacy_entities": spacy_ents,
        "tech_entities": tech_found
    }

# Aplicar NER a todos los mensajes de usuario (df_docs tiene los textos raw)
print(f"\n  🔍 Extrayendo entidades de {len(df_docs):,d} mensajes...")

# Usar nlp.pipe para procesamiento por lotes (mucho más rápido)
raw_texts = docs_raw['texts']

# Extraer entidades de dominio (rápido, regex)
tech_per_doc = []
for text in raw_texts:
    text_lower = (text or "").lower()
    found = [t for t in TECH_ENTITIES if re.search(r'\b' + re.escape(t) + r'\b', text_lower)]
    tech_per_doc.append(found)

df_docs['tech_entities'] = tech_per_doc
df_docs['n_tech_entities'] = df_docs['tech_entities'].apply(len)

# Extraer entidades de spaCy (más lento, usar pipe)
# Truncar textos a 5000 chars para evitar exceder el límite de spaCy (1M chars)
print("  Procesando con spaCy (puede tomar unos segundos)...")
truncated_texts = [str(t)[:5000] for t in raw_texts]
spacy_ents_per_doc = []
for doc in nlp_spacy.pipe(truncated_texts, batch_size=200, n_process=1):
    ents = [(ent.text, ent.label_) for ent in doc.ents]
    spacy_ents_per_doc.append(ents)

df_docs['spacy_entities'] = spacy_ents_per_doc
print(f"  ✅ NER completado.")

# ----- A) Estadísticas generales de entidades -----
all_tech = [t for techs in tech_per_doc for t in techs]
all_spacy = [ent for ents in spacy_ents_per_doc for ent in ents]

print(f"\n  📊 ESTADÍSTICAS DE ENTIDADES:")
print(f"  Mensajes con ≥1 tecnología: {sum(1 for t in tech_per_doc if t):,d} "
      f"({sum(1 for t in tech_per_doc if t)/len(tech_per_doc)*100:.1f}%)")
print(f"  Total menciones de tecnologías: {len(all_tech):,d}")
print(f"  Tecnologías únicas detectadas: {len(set(all_tech)):,d}")
print(f"  Entidades spaCy detectadas: {len(all_spacy):,d}")

# Top tecnologías mencionadas
tech_counts = Counter(all_tech).most_common(20)
print(f"\n  🔧 TOP 20 TECNOLOGÍAS MENCIONADAS:")
for tech, count in tech_counts:
    bar = "█" * min(int(count / max(c for _, c in tech_counts) * 30), 30)
    print(f"    {tech:15s} {count:>6,d}  {bar}")

# Top entidades de spaCy por tipo
spacy_by_type = {}
for text, label in all_spacy:
    spacy_by_type.setdefault(label, []).append(text)

print(f"\n  🏛️ ENTIDADES SPACY POR TIPO:")
for label, entities in sorted(spacy_by_type.items(), key=lambda x: -len(x[1])):
    top_ents = Counter(entities).most_common(5)
    top_str = ", ".join([f"{e}({c})" for e, c in top_ents])
    print(f"    {label:8s} ({len(entities):,d} total): {top_str}")

# ----- B) CRUCE: Tecnologías × Temas LDA -----
print(f"\n  🔗 CRUCE: TECNOLOGÍAS × TEMAS LDA:")
print(f"  {'Tema':<6} {'Nombre':<30} {'Top Tecnologías (% de docs del tema)'}")
print(f"  {'-'*6} {'-'*30} {'-'*50}")

for topic_id in range(n_topics):
    topic_mask = df_docs['top_topic'] == topic_id
    topic_docs = df_docs[topic_mask]
    n_topic_docs = len(topic_docs)
    
    if n_topic_docs == 0:
        continue
    
    # Contar tecnologías en este topic
    topic_techs = [t for techs in topic_docs['tech_entities'] for t in techs]
    topic_tech_counts = Counter(topic_techs).most_common(5)
    
    name = topic_names.get(topic_id, f"Topic {topic_id}")
    
    if topic_tech_counts:
        tech_str = ", ".join([
            f"{tech} ({count/n_topic_docs*100:.0f}%)" 
            for tech, count in topic_tech_counts
        ])
    else:
        tech_str = "(sin tecnologías detectadas)"
    
    print(f"  {topic_id:<6} {name:<30} {tech_str}")

# ----- C) Tecnologías problemáticas (alta duración + alta frecuencia) -----
print(f"\n  ⚠️ TECNOLOGÍAS PROBLEMÁTICAS (asociadas a conversaciones largas):")

# Expandir: una fila por (conversación, tecnología)
df_docs_with_convdata = df_docs.merge(
    analysis[['CONVERSATION_ID', 'CONV_SECONDS', 'user_message_count', 'skill_name']],
    left_on='conv_id', right_on='CONVERSATION_ID', how='inner'
)

tech_rows = []
for idx, row in df_docs_with_convdata.iterrows():
    for tech in row['tech_entities']:
        tech_rows.append({
            'tech': tech,
            'conv_seconds': row['CONV_SECONDS'],
            'user_msgs': row['user_message_count'],
            'topic': row['top_topic'],
            'skill': row['skill_name']
        })

if tech_rows:
    df_tech = pd.DataFrame(tech_rows)
    
    tech_stats = (df_tech.groupby('tech')
                  .agg(
                      mentions=('tech', 'size'),
                      avg_duration=('conv_seconds', 'mean'),
                      avg_user_msgs=('user_msgs', 'mean')
                  )
                  .sort_values('avg_duration', ascending=False))
    
    # Filtrar solo tecnologías con suficientes menciones
    tech_stats = tech_stats[tech_stats['mentions'] >= 10]
    
    print(f"  {'Tecnología':<15} {'Menciones':>10} {'Duración Avg(s)':>16} {'Msgs/Conv Avg':>14}")
    print(f"  {'-'*15} {'-'*10} {'-'*16} {'-'*14}")
    for tech, row in tech_stats.head(15).iterrows():
        flag = " 🔴" if row['avg_duration'] > analysis['CONV_SECONDS'].mean() * 1.5 else ""
        print(f"  {tech:<15} {int(row['mentions']):>10,d} {row['avg_duration']:>16.0f} {row['avg_user_msgs']:>14.1f}{flag}")
    
    # ----- D) Routing insights: ¿A qué skill se rutean las tecnologías? -----
    print(f"\n  🎯 ROUTING: ¿A QUÉ SKILL SE RUTEA CADA TECNOLOGÍA?")
    top_techs = tech_stats.head(10).index.tolist()
    
    for tech in top_techs:
        tech_data = df_tech[df_tech['tech'] == tech]
        skill_dist = tech_data['skill'].value_counts().head(3)
        skills_str = " | ".join([f"{skill}: {count:,d} ({count/len(tech_data)*100:.0f}%)" 
                                  for skill, count in skill_dist.items()])
        print(f"    {tech:15s} → {skills_str}")
else:
    print("  No se encontraron suficientes tecnologías para analizar.")


# ================================================================
# 💾 GUARDAR RESULTADOS
# ================================================================
print("\n" + "=" * 60)
print("💾 GUARDANDO RESULTADOS")
print("=" * 60)

# Preparar df_docs para guardado (convertir listas a strings para CSV)
df_docs_save = df_docs.copy()
df_docs_save['tech_entities'] = df_docs_save['tech_entities'].apply(lambda x: ", ".join(x) if x else "")
df_docs_save['spacy_entities'] = df_docs_save['spacy_entities'].apply(
    lambda x: "; ".join([f"{text}:{label}" for text, label in x]) if x else ""
)

analysis.to_csv("../data/lda_flow_cross_analysis.csv", index=False)
df_docs_save.to_csv("../data/lda_doc_topics_with_ner.csv", index=False)
df_topics.to_csv("../data/lda_topic_terms.csv", index=False)

import json
with open("../data/lda_topic_names.json", "w", encoding="utf-8") as f:
    json.dump(topic_names, f, ensure_ascii=False, indent=2)

# Guardar resumen de tecnologías por topic
ner_summary = {}
for topic_id in range(n_topics):
    topic_docs = df_docs[df_docs['top_topic'] == topic_id]
    topic_techs = [t for techs in topic_docs['tech_entities'] for t in techs]
    ner_summary[str(topic_id)] = {
        "topic_name": topic_names.get(topic_id, f"Topic {topic_id}"),
        "top_technologies": dict(Counter(topic_techs).most_common(10)),
        "n_docs": len(topic_docs)
    }

with open("../data/ner_topic_summary.json", "w", encoding="utf-8") as f:
    json.dump(ner_summary, f, ensure_ascii=False, indent=2)

print(f"\n  ✅ Resultados guardados en ../data/")
print(f"     - lda_flow_cross_analysis.csv")
print(f"     - lda_doc_topics_with_ner.csv  (incluye entidades)")
print(f"     - lda_topic_terms.csv")
print(f"     - lda_topic_names.json")
print(f"     - ner_topic_summary.json")
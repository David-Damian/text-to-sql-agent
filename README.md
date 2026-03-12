# 🤖 Conversational Agent Analytics: Topic Modeling & NER Pipeline

> **Análisis integral de un agente conversacional de IA** para identificar patrones de interacción problemáticos, evaluar la lógica de enrutamiento y mejorar la experiencia del usuario mediante NLP avanzado.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![spaCy](https://img.shields.io/badge/spaCy-3.8-09A3D5?logo=spacy)
![scikit-learn](https://img.shields.io/badge/scikit--learn-LDA-F7931E?logo=scikit-learn)
![Gemini](https://img.shields.io/badge/Gemini_API-Topic_Labeling-4285F4?logo=google)

---

## 📋 Tabla de Contenidos

- [Contexto de Negocio](#-contexto-de-negocio)
- [Objetivo](#-objetivo)
- [Metodología](#-metodología)
- [Arquitectura del Pipeline](#-arquitectura-del-pipeline)
- [Hallazgos Clave](#-hallazgos-clave)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Stack Tecnológico](#-stack-tecnológico)
- [Cómo Ejecutar](#-cómo-ejecutar)
- [Datos](#-datos)

---

## 💼 Contexto de Negocio

Una empresa opera un **agente conversacional de IA** que atiende consultas técnicas de usuarios. El agente tiene múltiples skills de enrutamiento y nodos de conversación diseñados para guiar al usuario hacia una solución.

### El Problema
La empresa sospecha que los usuarios **no están siguiendo el flujo de conversación deseado**, lo que resulta en:
- Conversaciones innecesariamente largas
- Usuarios que abandonan sin resolución
- Lógica de enrutamiento que no dirige correctamente según el tema

### El Impacto
Sin este análisis, el equipo de producto no tiene visibilidad sobre **dónde** y **por qué** la experiencia del usuario se deteriora, lo que lleva a pérdida de engagement y potencialmente de ingresos.

---

## 🎯 Objetivo

Proporcionar **hallazgos accionables** que permitan:

1. **Identificar los temas principales** de consulta de los usuarios
2. **Detectar patrones de abandono** (drop-off) por tema y nodo
3. **Evaluar la lógica de routing** cruzando temas con skills/nodos
4. **Extraer tecnologías específicas** mencionadas para mejorar el enrutamiento
5. **Recomendar mejoras concretas** al flujo conversacional

---

## 🔬 Metodología

El análisis emplea un pipeline de **3 técnicas complementarias de NLP**:

### 1. LDA (Latent Dirichlet Allocation) — Topic Modeling
- Modelo generativo probabilístico que descubre temas latentes en las conversaciones
- Preprocesamiento: limpieza → normalización → tokenización → stemming (SnowballStemmer español)
- Vectorización con Bag-of-Words (CountVectorizer)
- Etiquetado automático de temas vía Gemini API

### 2. NER (Named Entity Recognition) — Extracción de Entidades
- **spaCy `es_core_news_sm`**: modelo pre-entrenado (CNN) para detectar entidades estándar (PER, ORG, LOC, MISC)
- **Diccionario de dominio**: detección basada en regex para tecnologías específicas (Python, n8n, WhatsApp, etc.) que el modelo genérico no reconoce
- Cruce de entidades detectadas con los temas de LDA

### 3. Cross-Analysis (LDA × Flujo × NER)
- Cruce de temas con metadata de routing (`LAST_NODE_ID`, `NEXT_NODE_ID`, `skill_name`)
- Análisis de duración y drop-off por tema
- Identificación de tecnologías problemáticas (alta duración + alta frecuencia)
- Mapeo de tecnología → skill para validar routing

---

## 🏗️ Arquitectura del Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE COMPLETO                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  📥 Stage 1: Data Loading & Language Filtering               │
│       │                                                      │
│       ▼                                                      │
│  🧹 Stage 2: NLP Preprocessing                              │
│       │  (clean → normalize → tokenize → stem)               │
│       ▼                                                      │
│  📐 Stage 3: Vectorization (CountVectorizer)                 │
│       │                                                      │
│       ▼                                                      │
│  🧠 Stage 4: LDA Topic Modeling                             │
│       │                                                      │
│       ▼                                                      │
│  📄 Stage 5: Document-Topic Assignment                       │
│       │                                                      │
│       ▼                                                      │
│  🤖 Stage 6: GenAI Topic Labeling (Gemini API)              │
│       │                                                      │
│       ▼                                                      │
│  🔗 Stage 7: Cross-Analysis (LDA × Flujo)                   │
│       │  ├── Duración por tema                               │
│       │  ├── Drop-off rate por tema                          │
│       │  ├── Distribución tema → skill                       │
│       │  └── Nodos terminales por tema                       │
│       ▼                                                      │
│  🏷️ Stage 8: NER × LDA                                     │
│       │  ├── spaCy NER (PER, ORG, LOC)                      │
│       │  ├── Diccionario de tecnologías                      │
│       │  ├── Cruce: tecnologías × temas                     │
│       │  ├── Tecnologías problemáticas                       │
│       │  └── Routing: tecnología → skill                    │
│       ▼                                                      │
│  💾 Resultados (CSV + JSON)                                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Hallazgos Clave

> ⚠️ **Nota:** Los hallazgos a continuación son representativos de una ejecución del pipeline. Los resultados exactos pueden variar según los datos de entrada.

### Distribución de Temas
| Tema | Etiqueta | Documentos | Drop-off Rate |
|------|----------|------------|---------------|
| 0 | Soluciones de Código JavaScript | 811 | 44.9% |
| 1 | Soporte Técnico IA | 1,229 | 33.4% |
| 2 | Manejo de errores técnicos | 3,014 | 35.1% |
| 3 | Optimización de Skills IA | 16,045 | 40.6% |
| 4 | Análisis y generación de código | 500 | 72.8% 🔴 |
| 5 | Soporte IA Python | 2,072 | 57.2% |

### Hallazgo Principal
- **Tema 4 ("Análisis y generación de código")** tiene el **mayor drop-off (72.8%)** y la **mayor duración promedio** → El agente no está equipado para resolver este tipo de consultas
- **100% del tráfico** se rutea a un solo skill (`magic_help_skill`) → Oportunidad de crear skills especializados

---

## 📁 Estructura del Proyecto

```
.
├── README.md                          # Este archivo
├── data/                              # Datos (no incluidos en git por tamaño)
│   ├── magic_data_raw.csv             # Dataset original (~92MB)
│   ├── processed_conversations.csv    # Conversaciones procesadas
│   ├── processed_exploded_conversations.csv  # Mensajes individuales
│   ├── lda_doc_topics_with_ner.csv    # 🔧 Output: temas + entidades por mensaje
│   ├── lda_flow_cross_analysis.csv    # 🔧 Output: cruce LDA × flujo
│   ├── lda_topic_terms.csv            # 🔧 Output: términos por tema
│   ├── lda_topic_names.json           # 🔧 Output: nombres de temas (Gemini)
│   └── ner_topic_summary.json         # 🔧 Output: resumen NER por tema
├── notebooks/
│   ├── EDA.ipynb                      # Análisis exploratorio de datos
│   ├── LDA_analyze_trending_queries.ipynb  # Prototipo LDA
│   └── LDA_text.ipynb                 # Experimentación con textos
└── src/
    ├── auxiliar_functions.py           # Funciones auxiliares (parsing, lang detect)
    └── LDA_for ai_convs_analysis.py   # 🎯 Pipeline principal (LDA + NER + Cross)
```

---

## 🛠️ Stack Tecnológico

| Categoría | Tecnología | Uso |
|-----------|-----------|-----|
| **Topic Modeling** | scikit-learn (LDA) | Descubrimiento de temas latentes |
| **NER** | spaCy (es_core_news_sm) | Extracción de entidades nombradas |
| **NER (dominio)** | Regex + Diccionario | Detección de tecnologías específicas |
| **Preprocessing** | NLTK (SnowballStemmer) | Stemming en español |
| **Topic Labeling** | Google Gemini API | Etiquetado automático de temas con LLM |
| **Data** | pandas, numpy | Manipulación y análisis de datos |
| **Visualization** | matplotlib | Gráficas y visualizaciones |
| **Language Detection** | langdetect | Filtrado por idioma |

---

## 🚀 Cómo Ejecutar

### Prerrequisitos

```bash
# Crear entorno con mamba/conda
mamba create -n nlp_env python=3.10
mamba activate nlp_env

# Instalar dependencias
pip install pandas numpy scikit-learn nltk spacy matplotlib langdetect google-genai openai

# Descargar modelo de spaCy en español
python -m spacy download es_core_news_sm

# Descargar stopwords de NLTK
python -c "import nltk; nltk.download('stopwords')"
```

### Ejecución

```bash
cd src/
python "LDA_for ai_convs_analysis.py"
```

### Variables de Entorno (recomendado)

```bash
export GOOGLE_GENAI_API_KEY="tu-api-key-aqui"
```

---

## 📦 Datos

Los archivos de datos no están incluidos en el repositorio por su tamaño (~500MB total). Para reproducir los resultados:

1. Coloca `magic_data_raw.csv` en el directorio `data/`
2. Ejecuta el notebook `EDA.ipynb` para generar los archivos procesados
3. Ejecuta el pipeline principal desde `src/`

### Schema del Dataset Original

| Columna | Descripción |
|---------|-------------|
| `CONVERSATION_ID` | Identificador único de conversación |
| `CONV_LAST_UPDATED_DTTM` | Timestamp de última actualización |
| `CONVERSATION` | Metadata JSON (identity, skill, status, path) |
| `HISTORY_MESSAGES` | Lista de mensajes [{role, text, metadata}] |
| `CONV_SECONDS` | Duración total en segundos |
| `NEXT_NODE_ID` | Siguiente nodo en el flujo |
| `LAST_NODE_ID` | Último nodo visitado |
| `LAST_NODE_TYPE` | Tipo del último nodo |

---

"""
Run this script ONCE locally to pre-compute demo results.
Usage:
    export OPENAI_API_KEY="sk-..."
    python cache_demos.py
"""
import json
import os
import sys

DEMO_QUERIES = [
    "¿Cuántos empleados son estrellas en ascenso (Rising Stars)?",
    "¿Quiénes son los empleados en riesgo de fuga (Flight Risk)?",
    "¿Cuántos empleados tienen un rol estancado (Stagnant Role)?",
    "¿Cuál es el salario mensual promedio por departamento?",
    "¿Cuántos empleados han dejado la empresa (Attrition = 'Yes') agrupados por nivel de educación?",
]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "cached_demos")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "demos.json")


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ Set OPENAI_API_KEY environment variable first.")
        sys.exit(1)

    from agent_logic import initialize_agent, run_text_to_sql_agent

    print("🔧 Initializing agent...")
    initialize_agent(api_key)

    results = []
    for i, query in enumerate(DEMO_QUERIES, 1):
        print(f"\n{'='*60}")
        print(f"📝 [{i}/{len(DEMO_QUERIES)}] Running: {query}")
        print(f"{'='*60}")
        try:
            result = run_text_to_sql_agent(query)
            results.append({
                "query": query,
                "answer": result["answer"],
                "steps": result["steps"],
                "reward": result["reward"],
            })
            print(f"✅ Done — Reward: {result['reward']}")
        except Exception as e:
            print(f"❌ Failed: {e}")
            results.append({
                "query": query,
                "answer": f"Error during pre-computation: {e}",
                "steps": [],
                "reward": 0.0,
            })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 Cached {len(results)} demos to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

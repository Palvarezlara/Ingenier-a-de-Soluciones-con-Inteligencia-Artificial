"""
IL2.3: Planificación Básica con LangChain
========================================
Ejemplo de cómo un agente LangChain puede planificar y ejecutar pasos simples usando una herramienta.
"""

# Requiere: pip install langchain openai
from langchain_openai import ChatOpenAI
from langchain_classic.agents import initialize_agent, Tool, AgentType
import os

# Configura tu API key de OpenAI
os.environ["GITHUB_TOKEN"] = os.environ.get("GITHUB_TOKEN")

# Herramienta personalizada: pasos para preparar café
def pasos_cafe(_):
    return "1. Calentar agua\n2. Añadir café al filtro\n3. Verter agua caliente\n4. Servir en una taza"

herramienta_cafe = Tool(
    name="PasosCafé",
    func=pasos_cafe,
    description="Devuelve los pasos para preparar café."
)

# Inicializa el LLM y el agente
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ.get("GITHUB_TOKEN"),
    base_url="https://models.github.ai/inference",
    temperature=0
)
agente = initialize_agent(
    tools=[herramienta_cafe],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

if __name__ == "__main__":
    print("Planificación con LangChain:")
    print(agente.run("¿Cuáles son los pasos para preparar café?")) 
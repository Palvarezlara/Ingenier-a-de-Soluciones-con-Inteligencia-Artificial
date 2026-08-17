"""
IL2.3: Orquestación Multi-Agente con CrewAI
==========================================
Ejemplo de cómo dos agentes CrewAI colaboran para resolver una tarea.
"""

# Requiere: pip install crewai
from crewai import Agent, Task, Crew
from crewai import LLM
import os



# LLM compatible con GitHub Models
llm = LLM(
    model="openai/gpt-4o-mini",
    api_key=os.environ["GITHUB_TOKEN"],
    base_url="https://models.github.ai/inference",
    temperature=0
)

# Agente 1: Investigador
investigador = Agent(
    role="Investigador",
    goal="Buscar información completa sobre una capital del mundo",
    backstory="Eres experto en encontrar datos rápidos.",
    llm=llm
)

# Agente 2: Redactor
redactor = Agent(
    role="Redactor",
    goal="Redactar una respuesta clara y breve",
    backstory="Eres especialista en explicar conceptos de forma sencilla.",
    llm=llm
)

# Tareas
tarea_investigar = Task(
    description="Busca la informacion sobre la capital de Francia",
    expected_output="Toda la informacion encontrada sobre la capital de Francia",
    agent=investigador
)

tarea_redactar = Task(
    description="Redacta una respuesta usando la información encontrada",
    expected_output="Una oracion con la informacion bien redactada",
    agent=redactor,
    context=[tarea_investigar]
)

# Crew (equipo)
crew = Crew(
    agents=[investigador, redactor],
    tasks=[tarea_investigar, tarea_redactar],
    verbose=True
)

if __name__ == "__main__":
    print("Orquestación multi-agente con CrewAI:")
    print(crew.kickoff()) 
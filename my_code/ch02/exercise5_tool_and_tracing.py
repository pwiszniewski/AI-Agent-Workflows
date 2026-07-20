import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, ModelSettings
from agents import function_tool, trace

# Load environment variables from .env file
load_dotenv()

# Agent Instructions
# instructions = """
# You are a research planning assistant.

# **TASK INSTRUCTIONS**
# - You will be given a research topic.
# - Your task is to provide a plan on how to research this topic.
# - Output 5 concise tas        ks (5 words or less) to your plan.
# """

instructions = """
You are a research planning assistant. 
Use th given resources to find answer about social platforms
Follow the instuctions below to give best answers to the professor.

```
- You will be given a research topic.
- Begin by using the tool get_research_sources() to get a list of available research sources. 
 Constrain your research plan only to use the available research sources.
- Your task is to provide a plan on how to research this topic.
- Output 5 concise tasks and specify which of the available research sources will be used for each task.
- limit all points up to 4 words
- use academic language
"""


from pydantic import BaseModel, ConfigDict
from typing_extensions import TypedDict

class Task(TypedDict):
    id: int
    description: str

class ResearchPlanModel(BaseModel):
    tasks: list[Task]
    """Numbered tasks for research."""

    model_config = ConfigDict(extra='forbid')

@function_tool
def get_research_sources() -> list[str]:
    """Provides a list of research sources."""
    search_sources = [
        "Facebook",
        "Instagra,",
        "LinekIn",
    ]
    return search_sources   


client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

name="Research Planner"
agent = Agent(
        name=name,
        instructions=instructions,
        model=OpenAIChatCompletionsModel(
            # model="gemini-3.1-flash-lite",
            model="gemini-3.5-flash",
            openai_client=client,
        ),
        model_settings=ModelSettings(
            # temperature=0.0,
            temperature=1.0,
            # max_tokens=150,
            # top_p=1.0,
            # frequency_penalty=0.5,  
            # presence_penalty=0.5,  
        ),
        output_type=ResearchPlanModel,
        tools=[get_research_sources]
    )

input = "learn about AI agents"

with trace("Chapter 2 Tool Demo"):
    result = Runner.run_sync(
        agent,
        input=input,
    )

# print(f'{result=}')
print(result.final_output)

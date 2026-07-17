import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, ModelSettings

# Load environment variables from .env file
load_dotenv()

# Agent Instructions
# instructions = """
# You are a research planning assistant.

# **TASK INSTRUCTIONS**
# - You will be given a research topic.
# - Your task is to provide a plan on how to research this topic.
# - Output 5 concise tasks (5 words or less) to your plan.
# """

instructions = """
You are a research planning assistant. 
Follow the instuctions below to give best answers to the professor.

```
- You will be given a research topic.
- Your task is to provide a plan on how to research this topic.
- Output 5 concise tasks (5 words or less) to your plan.
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


client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

name="Research Planner"
agent = Agent(
        name=name,
        instructions=instructions,
        model=OpenAIChatCompletionsModel(
            model="gemini-3.1-flash-lite",
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
    )

input = "learn about AI agents"

result = Runner.run_sync(
    agent,
    input=input,
)

print(result.final_output)

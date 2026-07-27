
import asyncio
from pathlib import Path

from agents import Agent, Runner
from agents.mcp import MCPServerStdio, MCPServerStdioParams

from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel

import os
from dotenv import load_dotenv
load_dotenv()                       # konieczne żeby widział klucz API !!!

SANDBOX = os.path.dirname(os.path.abspath(__file__))

SCRIPT = Path(__file__).with_name(
    "01_research_tools_mcp_server.py").resolve()

client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

from pydantic import BaseModel
from typing import List
class ResearchSourcesModel(BaseModel):
        research_sources: List[str]
        """A list of research sources to use for research."""

servers = [
    MCPServerStdio(
        name="Research Tools",
        params=MCPServerStdioParams(
            command="mcp",
            args=["run", str(SCRIPT)],
        ),
    ),
    MCPServerStdio(
        name="sequential-thinking",
        params={
            "command": "npx",
            "args": [
                "-y", 
                "@modelcontextprotocol/server-sequential-thinking"
            ],
        },
    ),
    MCPServerStdio(
        name="filesystem",
        params={
            "command": "npx",
            "args": [
                "-y", 
                "@modelcontextprotocol/server-filesystem@latest", SANDBOX
            ],
        },
    ),
]

research_agent = Agent(
    name="Research Agent",
    instructions="""
        You are a research assistant.
        Your role is to find research sources.
        """,
    model=OpenAIChatCompletionsModel(
            # model="gemini-3.5-flash",
            model="gemini-3.1-flash-lite",
            openai_client=client,
        ),
    output_type=ResearchSourcesModel
)
thinking_agent = Agent(
    name="Thinking Agent",
    instructions="""
        You are a research assistant.
        Your role is to plan the research.
        """,
    model=OpenAIChatCompletionsModel(
            # model="gemini-3.5-flash",
            model="gemini-3.1-flash-lite",
            openai_client=client,
        ),
)

async def main():
    async with (
        servers[0] as research_srv,
        servers[1] as thinking_srv,
        servers[2] as fs_srv,
    ):
        goal = """
            Produce a research plan to find the book:
            'The Hitchhiker's Guide to the Galaxy'
            """
        print("Running...", goal)
        research_agent.mcp_servers = [research_srv]
        result = await Runner.run(research_agent, goal)
        research_sources = result.final_output.research_sources
        print('research agent:', research_sources)
        print('len of research plan:', len(research_sources))

        if research_sources and len(research_sources) > 0:
            thinking_agent.mcp_servers = [thinking_srv]
            agent_input = dict(
                 research_sources=research_sources,
                 goal=goal,
            )
            result = await Runner.run(thinking_agent, str(agent_input)) 
            print('################################################################################')
            print('thinking agent:', result.final_output)
        else:
             print('no research plan - aborted !')
             exit()

if __name__ == "__main__":
    asyncio.run(main())
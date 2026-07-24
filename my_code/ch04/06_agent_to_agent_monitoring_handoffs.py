import asyncio
import os
from pathlib import Path
from typing import List

from agents import Agent, Runner, handoff
from agents.mcp import MCPServerStdio, MCPServerStdioParams
from agents import Agent, RunContextWrapper
from pydantic import BaseModel

SANDBOX = os.path.dirname(os.path.abspath(__file__))
SCRIPT = Path(__file__).with_name("01_research_tools_mcp_server.py").resolve()

from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel
from dotenv import load_dotenv
load_dotenv()    
client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
model_name = 'gemini-3.1-flash-lite'
# model_name = 'gemini-3-flash-preview'
# model_name = 'gemini-3.6-flash'


async def main():
    class ResearchSourcesModel(BaseModel):
        research_sources: List[str]
        """A list of research sources to use for research."""

    # Instantiate the agents first…
    research_agent = Agent(
        name="Research Agent",
        output_type=ResearchSourcesModel,
        instructions="""
        You are a research assistant.
        Your role is to find research sources. 
        Do not make up or invent any research sources.
        Always hand off to the thinking agent.
        """,
        model=OpenAIChatCompletionsModel(
                    # model="gemini-3.5-flash",
                    # model="gemini-3.1-flash-lite",
                    model=model_name,
                    openai_client=client,
                ),
    )
    thinking_agent = Agent(
        name="Thinking Agent",
        instructions="""
        You are a research planning assistant.
        Your role is to plan the research.
        You will receive a list of research sources from the research agent.
        Use the sequentialThinking tool to create a research plan based on the sources.
        Always hand off to the filesystem agent.
        """,
        model=OpenAIChatCompletionsModel(
                    # model="gemini-3.5-flash",
                    # model="gemini-3.1-flash-lite",
                    model=model_name,
                    openai_client=client,
                ),
    )
    filesystem_agent = Agent(
        name="Filesystem Agent",
        instructions="""
        You are a filesystem assistant.
        Your role is to write the output as a text file.
        Never make up or invent any ouput.
        """,
        model=OpenAIChatCompletionsModel(
                    # model="gemini-3.5-flash",
                    # model="gemini-3.1-flash-lite",
                    model=model_name,
                    openai_client=client,
                ),
    )
    # Instantiate the servers next…
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
                "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
            },
        ),
        MCPServerStdio(
            name="filesystem",
            params={
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem@latest", SANDBOX],
            },
        ),
    ]

    # …then open them all at once
    async with (
        servers[0] as research_srv,
        servers[1] as thinking_srv,
        servers[2] as fs_srv,
    ):
    # ############################ najważdniejsze
    #     # updates to agent connections
    #     research_agent.mcp_servers = [research_srv]
    #     research_agent.handoffs = [thinking_agent]
    #     thinking_agent.mcp_servers = [thinking_srv]
    #     thinking_agent.handoffs = [filesystem_agent]
    #     filesystem_agent.mcp_servers = [fs_srv]

    #     from agents.extensions.visualization import draw_graph
    #     draw_graph(research_agent).view()    
    #     input("Press Enter to continue...")    

        goal = """
        Produce a research plan to find the book 'The Hitchhiker's Guide to the Galaxy'
        """

    #     print("Running...", goal)
    #     result = await Runner.run(
    #         research_agent,
    #         goal, 
    #         max_turns=25,
    #         )
    #     print(result.final_output)

        async def research_handoff(
            ctx: RunContextWrapper[None],
            sources: ResearchSourcesModel,
        ):
            print(f"Thinking agent called with sources: {sources.research_sources}")

        research_agent.mcp_servers = [research_srv]
        agent_handoff = handoff(
            agent=thinking_agent,
            on_handoff=research_handoff,
            input_type=ResearchSourcesModel,
        )
        research_agent.handoffs = [agent_handoff]
        thinking_agent.mcp_servers = [thinking_srv]
        thinking_agent.handoffs = [filesystem_agent]
        filesystem_agent.mcp_servers = [fs_srv]

        print("Running...", goal)
        result = await Runner.run(
            research_agent,
            goal,
            max_turns=25,
        )
        print(result.final_output)



if __name__ == "__main__":
    asyncio.run(main())

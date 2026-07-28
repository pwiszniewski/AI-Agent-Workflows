import asyncio
import os
from pathlib import Path
from typing import List

from agents import Agent, Runner
from agents.mcp import MCPServerStdio, MCPServerStdioParams
from pydantic import BaseModel

SANDBOX = os.path.dirname(os.path.abspath(__file__))
SCRIPT = Path(__file__).with_name("03_variable_research_tools.py").resolve()

from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel
from dotenv import load_dotenv
load_dotenv()    
client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
# model_name = 'gemini-3-flash-preview'
# model_name = 'gemini-3.6-flash'
# model_name = 'gemini-3.5-flash'
model_name = 'gemini-3.1-flash-lite'

from pydantic import BaseModel
class ResearchPlanModel(BaseModel):
        research_plan: str
        is_detailed: bool
        """A research plan"""

research_plan_guardrail_agent = Agent(
    name="Research Plan Guardrail Agent",
    instructions="""
        You are an output guardrail agent.
        Confirm the research plan is sufficiently detailed, atleast 1000 characters in length.
        If it is not sufficiently detailed, flag it.
        """,
    output_type=ResearchPlanModel,
        model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=client,
                ),
)

from agents import (
    GuardrailFunctionOutput,
    RunContextWrapper,
    output_guardrail,
    OutputGuardrailTripwireTriggered,
)
@output_guardrail
async def research_plan_guardrail(
    ctx: RunContextWrapper, agent: Agent, output: ResearchPlanModel
) -> GuardrailFunctionOutput:
    result = await Runner.run(
        research_plan_guardrail_agent, output.research_plan, context=ctx.context
    )
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_detailed,
    )

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
        """,
        output_type=ResearchPlanModel,
        output_guardrails=[research_plan_guardrail],
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
        goal = """
        Produce a research plan to find the book 'The Hitchhiker's Guide to the Galaxy'
        """
        print("Running...", goal)

        research_agent.mcp_servers = [research_srv]
        result = await Runner.run(research_agent, goal)
        thinking_agent.mcp_servers = [thinking_srv]
        final_output = result.final_output
        max_retries = 3
        for attempt in range(max_retries):
            try:
                agent_input = dict(
                    research_sources=final_output,
                    goal=goal,
                )
                result = await Runner.run(thinking_agent, str(agent_input))
                final_output = result.final_output.research_plan
                break
            except OutputGuardrailTripwireTriggered as output_tripped:
                final_output = output_tripped.guardrail_result.output.output_info
            if attempt == max_retries - 1:
                final_output = "A research plan didn't reach the required limit"
        filesystem_agent.mcp_servers = [fs_srv] 
        agent_input = dict(
            output=final_output,
            goal=goal,
        )
        result = await Runner.run(filesystem_agent, str(agent_input))
        print(result.final_output)


        
        # research_agent.mcp_servers = [research_srv]
        # result = await Runner.run(research_agent, goal)
        # # Extract the research sources from the result
        # research_sources = result.final_output.research_sources
        # if research_sources and len(research_sources) > 0:
        #     # if there are any research sources, use them to plan the research
        #     thinking_agent.mcp_servers = [thinking_srv]
        #     agent_input = dict(
        #         research_sources=research_sources,
        #         goal=goal,
        #     )
        #     result = await Runner.run(thinking_agent, str(agent_input))
        #     research_plan = result.final_output
        #     print(f'{research_plan=}')
        # else:
        #     research_plan = "No research sources found and no plan was created."
        # filesystem_agent.mcp_servers = [fs_srv]
        # agent_input = dict(
        #     output=research_plan,
        #     goal=goal,
        # )
        # result = await Runner.run(filesystem_agent, str(agent_input))
        # print(f'{result.final_output=}')


if __name__ == "__main__":
    asyncio.run(main())

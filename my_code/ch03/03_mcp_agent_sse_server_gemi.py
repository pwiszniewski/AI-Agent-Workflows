########
########

########## KOD NIE DZIAŁA


from mcp.server.fastmcp import FastMCP

from agents import Agent
from agents.mcp import MCPServerSse

 

from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel
from agents import Agent, Runner

import os
from dotenv import load_dotenv
load_dotenv()                       # konieczne żeby widział klucz API !!!

client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Create an MCP server
mcp = FastMCP("Research Tools") 


@mcp.tool()  
def get_research_sources() -> list[str]:
    """Provides a list of research sources."""
    search_sources = [
        "Wikipedia",
        "Google",
        "YouTube",
    ]
    return search_sources

import asyncio

# # if __name__ == "__main__":
# #     mcp.run()


# # only relevant section shown
# async def main():
#     async with MCPServerSse(
#         name="SSE Python Server",
#         params={
#             "url": "http://localhost:8000/sse",
#         },
#     ) as research_server:
#         agent = Agent(
#             name="Assistant",
#             instructions="Use the research tools to perform research.",
#             mcp_servers=[research_server],
#             model=OpenAIChatCompletionsModel(
#                     model="gemini-3.5-flash",
#                     openai_client=client,
#                 ),
#         )



# if __name__ == "__main__":
#     asyncio.run(main())

async def main():
    async with MCPServerSse(
        name="SSE Python Server",
        params={
            "url": "http://localhost:8000/sse",
        },
    ) as research_server:
        agent = Agent(
            name="Assistant",
            instructions="Use the research tools to perform research.",
            mcp_servers=[research_server],
        )

        print("Running: Get the available research sources")
        result = await Runner.run(agent, "Get the available research sources")
        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())

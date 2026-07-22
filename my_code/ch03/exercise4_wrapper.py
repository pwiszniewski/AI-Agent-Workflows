
from agents import Agent, Runner
from agents.mcp import MCPServerStdio, MCPServerStdioParams

import asyncio
import os
from dotenv import load_dotenv
load_dotenv()                       # konieczne żeby widział klucz API !!!

from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel
client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

async def main():    
    current_dir = os.path.dirname(os.path.abspath(__file__))

    async with MCPServerStdio(
        name="Mini FS.",
        params={
            "command": "npx",
            "args": ["-y", 
                     "@modelcontextprotocol/server-filesystem@latest",
                     current_dir],
        },
    ) as server:       
        agent = Agent(
            name="Filesystem Agent",
            instructions="Use the filesystem tools to help the user with their tasks.",
            mcp_servers=[server],
            model=OpenAIChatCompletionsModel(
                    model="gemini-3.5-flash",
                    openai_client=client,
                ),
            )

        print("Running: Get the available files")
        result = await Runner.run(
            starting_agent=agent, 
            # input="List the files in the current directory."
            input='List the files in the current directory and filter those containing "mcp_agent"'
            )
        print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())

# OUTPUT
# - **Files:**
#   - 01_claude_mcp_server.py
#   - 02_mcp_agent_stdio_server.py
#   - 03_mcp_agent_sse_server.py
#   - 04_mcp_agent_local_server_files.py
# … continued
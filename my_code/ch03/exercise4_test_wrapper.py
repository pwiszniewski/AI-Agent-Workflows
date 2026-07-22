
from agents import Agent, Runner
from agents.mcp import MCPServerStdio, MCPServerStdioParams

import asyncio
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()                       # konieczne żeby widział klucz API !!!

from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel
# client = AsyncOpenAI(
#     api_key=os.getenv("GEMINI_API_KEY"),
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
# )

SCRIPT = Path(__file__).with_name("exercise4_wrapper.py").resolve()

async def main():    
    current_dir = os.path.dirname(os.path.abspath(__file__))

    async with MCPServerStdio(
        name="Research Tools",
        params=MCPServerStdioParams(
            command="mcp",
            args=["run", str(SCRIPT)],
        ),
    ) as server:
        tools = await server.list_tools()
        print("Tools provided by the MCP server:")
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")
            print(f"  - Parameters: {tool.inputSchema}")
            print("-" * 40)

        tool_result = await server.call_tool(
            "list_directory_mod", 
            dict(path=current_dir))
        content = [text.text for text in tool_result.content]
        print(content[0])   

if __name__ == "__main__":
    asyncio.run(main())





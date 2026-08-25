# docker run -it --rm -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest

import os

from agents import Agent, Runner, trace

import os
from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()    
client_agent = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
# model_name = 'gemini-3-flash-preview'
# model_name = 'gemini-3.6-flash'
model_name = 'gemini-3.1-flash-lite'
# model_name = "gemini-2.5-pro"

os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"

from agents import set_trace_processors
from phoenix.otel import register

set_trace_processors([])  # Disable default trace processors
# configure the Phoenix tracer
tracer_provider = register(
    project_name="agents",  # Default is 'default'
    auto_instrument=True,  # Auto-instrument your app based on installed dependencies
)

agent = Agent(name="Assistant", instructions="You are a helpful assistant", model=OpenAIChatCompletionsModel(
                                    model=model_name,
                                    openai_client=client_agent,
                                ),)


async def main():
    with trace("Haiku Generator"):
        result = await Runner.run(
            agent, "Write a haiku about recursion in programming."
        )
        print(result.final_output)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

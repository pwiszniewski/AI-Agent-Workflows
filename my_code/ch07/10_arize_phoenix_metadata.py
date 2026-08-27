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
from openinference.instrumentation import (
    using_metadata,
    using_session,
)

# pip install openinference-instrumentation arize-phoenix-otel
from phoenix.otel import register

set_trace_processors([])  # Disable default trace processors
# configure the Phoenix tracer
tracer_provider = register(
    project_name="agents",  # Default is 'default'
    auto_instrument=True,  # Auto-instrument your app based on installed dependencies
)

model = "gpt-5-mini"
agent = Agent(name="Assistant", instructions="Always answer in a Haiku", model=OpenAIChatCompletionsModel(
                                    model=model_name,
                                    openai_client=client_agent,
                                ),)


async def main():
    agent_input = dict(question="why is the sky blue?")
    metadata = dict(run_id="abc123", env="dev", customer_tier="pro", model=model)
    with (
        using_session("sess-42"),
        using_metadata(metadata),
    ):
        with trace("Haiku Generator"):
            result = await Runner.run(agent, str(agent_input))
            print(result.final_output)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

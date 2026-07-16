import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel

load_dotenv()

# Tworzymy klienta tylko raz
client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


async def ask_agent(
    prompt: str,
    instructions: str = "You are a helpful assistant.",
    model: str = "gemini-3.5-flash",
    name:str = 'Agent'
) -> str:

    agent = Agent(
        name=name,
        instructions=instructions,
        model=OpenAIChatCompletionsModel(
            model=model,
            openai_client=client,
        ),
    )

    result = await Runner.run(
        agent,
        input=prompt,
    )

    return result.final_output
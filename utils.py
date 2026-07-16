import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, ModelSettings

load_dotenv()

# Tworzymy klienta tylko raz
client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


async def ask_agent(
    prompt: str,
    instructions: str = "You are a helpful assistant.",
    # model: str = "gemini-3.5-flash",
    model: str = "gemini-3.1-flash-lite",
    name:str = 'Agent',
    model_settings=None,
    output_type=None,
) -> str:
    # print(model_settings)
    agent = Agent(
        name=name,
        instructions=instructions,
        model=OpenAIChatCompletionsModel(
            model=model,
            openai_client=client,
        ),
        model_settings=ModelSettings(
            **model_settings if model_settings else {}
            # temperature=0.0,
            # max_tokens=150,
            # top_p=1.0,
            # frequency_penalty=0.5,  
            # presence_penalty=0.5,  
        ),
        output_type=output_type,
    )

    result = await Runner.run(
        agent,
        input=prompt,
    )

    return result.final_output
#docker run -it --rm -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest



import asyncio

from agents import Agent, Runner, function_tool
from agents import (
    Agent,
    GuardrailFunctionOutput,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    function_tool,
    output_guardrail,
)
from dotenv import load_dotenv
from pydantic import BaseModel

import os
from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel
from dotenv import load_dotenv
load_dotenv()    
client_agent = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
# model_name = 'gemini-3-flash-preview'
model_name = 'gemini-3.6-flash'
# model_name = 'gemini-3.1-flash-lite'
# model_name = "gemini-2.5-pro"

###### phoenix #######
from agents import trace
from agents import set_trace_processors
from phoenix.otel import register

os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"

set_trace_processors([])
tracer_provider = register(
    project_name="agents",
    auto_instrument=True,
)

from openinference.instrumentation import (
    using_metadata,
    using_session,
)
############################################################
agent = Agent(
    name="Assistant", 
    instructions="You are a helpful assistant")

# Simple in‑memory knowledge base
_special_knowledge_db = [
    "Nebula Forge engine spins antimatter rings for gravity control.",
    "Solaris Glacier absorbs heat, releasing luminescent icefire at dusk.",
    "Quantum Bark trees emit entangled photons guiding nocturnal insect migrations.",
    "Aether Silk fabric self-weaves repairs within fourteen-millisecond microtears.",
    "Chrono Coral reefs rewind water currents three minutes every solstice.",
]

# benchmarks for special knowledge
_benchmarks = [
    {"q": "Nebula Forge engine spins what?", "a": "antimatter", "wrong": "fusion"},
    {
        "q": "Solaris Glacier releases luminescent what?",
        "a": "icefire",
        "wrong": "lava",
    },
    {"q": "Quantum Bark trees emit what?", "a": "photons", "wrong": "spores"},
    {"q": "Aether Silk fabric repairs what?", "a": "microtears", "wrong": "threads"},
    {"q": "Chrono Coral reefs rewind what?", "a": "currents", "wrong": "tides"},
]

_last_context = ""  # Global variable to store the last context


@function_tool
def search_knowledge_by_keyword(query: str) -> dict:
    """
    Search the knowledge database for relevant facts.
    param query: The single keyword query string to search for.
    """
    global _last_context
    matches = [doc for doc in _special_knowledge_db if query.lower() in doc.lower()]
    print(f"Found {len(matches)} matches for '{query}'")
    _last_context = "\n".join(matches)
    return {"status": "ok", "context": _last_context}


agent = Agent(
    name="RAG Agent",
    instructions="""
You are a retrieval-augmented knowledge agent.
Break down the user's query into smaller parts if needed
to fetch relevant context for the user's query.
""",
    tools=[search_knowledge_by_keyword],
    model=OpenAIChatCompletionsModel(
                                    model=model_name,
                                    openai_client=client_agent,
                                ),
)


class EvaluationOutput(BaseModel):
    """Output model for evaluation agent."""
    is_correct: bool
    feedback: str

evaluation_agent = Agent(
    name="Evaluation Agent",
    instructions="""
You are an evaluation agent.
Your task is to evaluate the answers provided by the RAG Agent.
You will compare the answer against 
the expected answers key term to validate correctness.
""",
    output_type=EvaluationOutput,
)

class AnswerResult(BaseModel):
    """Output model for RAG agent."""

    question: str
    answer: str

@output_guardrail
async def ground_answer(
    context: RunContextWrapper, agent: Agent, output: AnswerResult
) -> GuardrailFunctionOutput:
    grounding_input = dict(
        question=output.question,
        answer=output.answer,
    )
    result = await Runner.run(
             grounding_agent, input=str(grounding_input))
    result = result.final_output

    return GuardrailFunctionOutput(
        output_info={
            "answer_is_grounded": result.is_answer_grounded,
            "feedback": result.feedback,
        },
        tripwire_triggered=result.is_answer_grounded is False,
    )

agent = Agent(
    name="RAG Agent",
    instructions="""
You are a retrieval-augmented knowledge agent.
Break down the user's query into smaller parts if needed
to fetch relevant context for the user's query.
""",
    output_type=AnswerResult,
    tools=[search_knowledge_by_keyword],
    model="gpt-4o",  # Specify the model to use
    output_guardrails=[ground_answer],
)

@function_tool
def get_last_context() -> str:
    """
    Retrieve the last context used by the grounding agent.
    """
    global _last_context
    if not _last_context:
        return "No context available."
    return _last_context

class GroundedAnswer(BaseModel):
    """Output model for grounding agent."""

    is_answer_grounded: bool
    feedback: str

grounding_agent = Agent(
    name="Grounding Agent",
    instructions="""
You are a grounding agent.
Your task is to evaluate the correctness of the answers
based on the provided question, the context used,
and output answer.
""",
    model="gpt-4o",  
    output_type=GroundedAnswer,
    tools=[get_last_context],
)


async def main():
    metadata = {"run_id": "ch7-ex5",
                "env": "dev",
                "model": "gpt-5"}
    with (
            using_session("sess-07"),
            using_metadata(metadata),
        ):
        with trace("RAG Benchmarks"):
            for benchmark in _benchmarks:
                question = benchmark["q"]
                try:
                    result = await Runner.run(agent, input=question)
                    result = result.final_output
                    answer = result.answer.strip()
                except OutputGuardrailTripwireTriggered as e:
                    print(f"Guardrail tripped. Info: {e.guardrail_result.output.output_info}")
                    answer = "No grounded answer available."

                print("" + "=" * 40)
                print(f"Question: {question}")
                print(f"Answer: {answer}")

if __name__ == "__main__":
    asyncio.run(main())

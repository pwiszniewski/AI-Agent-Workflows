import asyncio
import os
from pathlib import Path
from typing import List

from agents import Agent, Runner, RunContextWrapper
from agents.mcp import MCPServerStdio, MCPServerStdioParams
from pydantic import BaseModel

import os
from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel
from dotenv import load_dotenv
load_dotenv()    
client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
model_name = 'gemini-3-flash-preview'
# model_name = 'gemini-3.6-flash'
# model_name = 'gemini-3.5-flash'
# model_name = 'gemini-3.1-flash-lite'

def get_reflexion_solver_instructions(
    run_context: RunContextWrapper[str], agent: Agent[str]
) -> str:
    """Generate instructions for the reflexion solver agent."""
    instructions = (
        "You are a time-travel expert. Solve the problem step by step "
        "and be careful to avoid mistakes."
    )
    return instructions + "\nHINT:\n" + run_context.context


solver = Agent(
    name="TimeTravelerReflexion", 
    instructions=get_reflexion_solver_instructions,
    model=OpenAIChatCompletionsModel(
            model=model_name,
            openai_client=client,
        ),
)
critic = Agent(
    name="TimeTravelCritic",
    instructions=(
        "You are an expert tutor. If the solution is wrong, explain the error "
        "and give a concise hint for improvement."
    ),
    model=OpenAIChatCompletionsModel(
                model=model_name,
                openai_client=client,
            ),
)

# --- Problem spec ------------------------------------------------------------
problem = (
    "I left the year 2000 in a time machine, went forward 30 years, "
    "then back 40 years. I claim I'm now in 1990. Am I correct?"
)
problem = """
In a sci-fi film, Alex is a time traveler who decides to go back in time
to witness a famous historical event that took place 100 years ago,
which lasted for 10 days. He arrives three days before the event starts.
However, after spending six days in the past, he jumps forward in time
by 50 years and stays there for 20 days. Then, he travels back to
witness the end of the end. 
How many days does Alex spend in the past before he sees the end of the event?
"""
TARGET_DAYS = "26"  # expected final answer
MAX_ATTEMPTS = 5  # fail-safe cap on retries

async def main():
    #solver loop
    feedback_hint = ""
    for attempt_no in range(1, MAX_ATTEMPTS + 1):
        result = await Runner.run(
            solver, 
            input=problem, 
            context=feedback_hint)
        answer = result.final_output.strip()
        print(f"\nAttempt {attempt_no}:\n{answer}")

        has_correct_days = TARGET_DAYS in answer
        says_claim_correct = "yes" in answer.lower() or "correct" in answer.lower()
        solved = has_correct_days and says_claim_correct

        if solved:
            print(" Solution accepted.")
            break

        feedback_prompt = (
            f"Solution given:\n{answer}\n\n"
            f"Expected final days: {TARGET_DAYS}\n"
            "Explain the error briefly and give a helpful hint."
        )
        feedback_resp = await Runner.run(critic, input=feedback_prompt)
        hint = feedback_resp.final_output.strip()

        print(f"Feedback:\n{hint}")
        feedback_hint = hint
    else:
        print("\n  Max attempts reached without a correct solution.")

if __name__ == "__main__":
    asyncio.run(main())
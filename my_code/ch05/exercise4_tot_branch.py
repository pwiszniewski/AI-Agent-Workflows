import asyncio
from pathlib import Path

from agents import Agent, Runner
from agents.mcp import MCPServerStdio, MCPServerStdioParams

from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel

import os
from dotenv import load_dotenv
load_dotenv()                       # konieczne żeby widział klucz API !!!

client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
# model_name = 'gemini-3-flash-preview'
# model_name = 'gemini-3.6-flash'
model_name = 'gemini-3.1-flash-lite'

# agents and problem
generator = Agent(  
    name="ToT-Generator",
    instructions="Given the current situation, brainstorm a possible next step or action to reach the goal.",
    model=OpenAIChatCompletionsModel(
                model=model_name,
                openai_client=client,
            ),
    )

eval_instruct = """
Assess how likely the proposed plan will solve the problem. 

Output format:
score = N

Rules:
- output only one line
- output no explanation
- N must be an integer from 0 to 10
"""
# Agent for evaluating partial solutions
evaluator = Agent(
    name="ToT-Evaluator",
    instructions=eval_instruct,
    model=OpenAIChatCompletionsModel(
        model=model_name,
        openai_client=client,
    ),
)

problem = """
You need to reach the year 1800 from 2025 using a time machine
that can jump either -100 or -30 years.
"""

async def main():
    # running agents
    initial_thoughts = []
    n_candidates = 2
    for i in range(n_candidates):
        resp = await Runner.run(
            generator, input=f"Problem: {problem}\nThink of a first step."
        )
        initial_thoughts.append(resp.final_output.strip())

    promising_branches = []
    for thought in initial_thoughts:
        eval_resp = await Runner.run(
            evaluator, input=f"Plan: {thought}\nIs this promising (put a number)?"
        )
        print(eval_resp.final_output.lower())
        import re
        text = eval_resp.final_output
        match = re.search(r"(\d+)", text)
        if match:
            score = int(match.group(1))
        else:
            score = 0
        print('##################################################')
        print(f'{score=}')
        if score >= 7:
            next_step = await Runner.run(
                generator, input=f"Current idea: {thought}\nNext step?"
            )
            promising_branches.append(f"{thought} -> {next_step.final_output.strip()}")
        else:
            print(f'not promising')
    print("Initial thought candidates:", initial_thoughts)
    print(
        "Expanded promising branch:",
        promising_branches[0] if promising_branches else "None",
    )


if __name__ == "__main__":
    asyncio.run(main())
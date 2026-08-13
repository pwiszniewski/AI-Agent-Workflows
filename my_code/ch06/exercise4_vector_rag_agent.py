import uuid
from pathlib import Path

import chromadb
import tiktoken
from agents import Agent, Runner, function_tool  # OpenAI Agents SDK
from dotenv import load_dotenv

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
# model_name = 'gemini-3.6-flash'
# model_name = 'gemini-3.1-flash-lite'
model_name = 'gemini-3.5-flash-lite'
# model_name = "gemini-2.5-pro"

# ------------------------------------------------------------------
# 1. Load + chunk the script
# ------------------------------------------------------------------
script_text = Path("chapter_06/sample_documents/back_to_the_future.txt").read_text(
    encoding="utf-8"
)


def simple_chunk(text, max_tokens=150):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    words, chunk, chunks = text.split(), [], []
    for w in words:
        if len(tokenizer.encode(" ".join(chunk + [w]))) > max_tokens:
            chunks.append(" ".join(chunk))
            chunk = [w]
        else:
            chunk.append(w)
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks


docs = simple_chunk(script_text, max_tokens=150)

# ------------------------------------------------------------------
# 2. Create (or connect to) a Chroma collection with OpenAI embeddings
# ------------------------------------------------------------------
client = chromadb.PersistentClient(
    path="./chapter_06/chroma_script_store"  # on-disk so we reuse later
)
collection_name = "bttf_script"

# Try to get existing collection first
try:
    collection = client.get_collection(collection_name)
except Exception:
    # Collection doesn't exist, create it without embedding function
    collection = client.create_collection(name=collection_name)

# Populate once (skip if already populated)
if collection.count() == 0:
    collection.add(ids=[str(uuid.uuid4()) for _ in docs], documents=docs)


# ------------------------------------------------------------------
# 3. Define a semantic search tool that queries Chroma
# ------------------------------------------------------------------
@function_tool
def search_script(query: str, top_k: int = 3) -> str:
    res = collection.query(query_texts=[query], n_results=top_k)
    if res and "documents" in res and res["documents"] and res["documents"][0]:
        return "\n\n".join(res["documents"][0])  # combine best chunks
    return "No relevant documents found."


# ------------------------------------------------------------------
# 4. Build the agent
# ------------------------------------------------------------------
agent = Agent(
    name="Script Agent",
    instructions=(    
        "You answer questions about the movie *Back to the Future*.\n"  
        "When needed, call the `search_script` tool to fetch passages, "  
        "then cite or paraphrase them in your answer."  
        "Make sure your answers are grounded in the script text.\n"     
    ),
    tools=[search_script],
    model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=client_agent,
                ),
)

# ------------------------------------------------------------------
# 5. Ask a question
# ------------------------------------------------------------------
query = "Who is Pine City’s mayor in the script?"
result = Runner.run_sync(agent, query, max_turns=30)
print("\n--- ANSWER ---\n", result.final_output)

query = "What happens at 10:04 PM?"
result = Runner.run_sync(agent, query, max_turns=30)
print("\n--- ANSWER ---\n", result.final_output)


#########################################################
# Q: Who is Pine City’s mayor in the script?
# --- ANSWER ---
#  Based on the *Back to the Future* script, there is no mention of a "Pine City" or its mayor. (The town in the story is Hill Valley, and the famous mall near it is called Twin Pines Mall, later becoming Lone Pine Mall after one of the pine trees is run over).
# Q: What happens at 10:04 PM?
# --- ANSWER ---
#  Based on the original script for *Back to the Future* (which features a different plot involving a nuclear test site and a time machine powered by radiation rather than a DeLorean and a clock tower), 10:04 PM is not referenced. In this early draft, the key temporal moments center around the atomic bomb countdown in Nevada rather than a lightning strike at 10:04 PM.

## spodziewałem się innych wyników / halucynacji, ale odpowiedzi sa prawidlowe
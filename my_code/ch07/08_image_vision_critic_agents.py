# nie udało się nigdy uruchomić

import asyncio
import base64
import os
import subprocess
import sys

from dotenv import load_dotenv
from pydantic import BaseModel

from openai import OpenAI, AsyncOpenAI

from agents import (
    Agent,
    Runner,
    function_tool,
    trace,
    OpenAIChatCompletionsModel,
)

from google import genai
from google.genai import types

# ============================================================
# CONFIG
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client_agent = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

vision_client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

image_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# model_name = 'gemini-3-flash-preview'
# model_name = 'gemini-3.6-flash'
# model_name = 'gemini-3.1-flash-lite'
model_name = "gemini-3.5-flash"

# ============================================================
# HELPERS
# ============================================================


def open_file(path: str):
    if sys.platform.startswith("darwin"):
        subprocess.run(["open", path], check=False)

    elif os.name == "nt":
        os.startfile(path)

    elif os.name == "posix":
        subprocess.run(["xdg-open", path], check=False)

    else:
        print(f"Cannot open file on platform {sys.platform}")


def encode_image(image_path: str):

    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ============================================================
# TOOLS
# ============================================================


# @function_tool

def generate_image(prompt: str) -> str:
    """
    Generate image using Google Imagen.
    Returns image path.
    """

    os.makedirs("gen_images", exist_ok=True)

    image_path = "gen_images/generated.png"

    # response = imagen_client.models.generate_images(
    #     model="imagen-4.0-generate-001",
    #     prompt=prompt,
    #     config=types.GenerateImagesConfig(
    #         number_of_images=1
    #     ),
    # )

    response = image_client.models.generate_content(
        model="gemini-3.1-flash-image",
        contents=prompt,
    )

    image_bytes = response.generated_images[
        0
    ].image.image_bytes

    with open(image_path, "wb") as f:
        f.write(image_bytes)

    print(f"Saved image: {image_path}")

    return image_path


@function_tool
def describe_image(
    image_path: str,
    prompt: str,
) -> str:
    """
    Describe image using Gemini Vision.
    """

    image_b64 = encode_image(image_path)

    response = vision_client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        },
                    },
                ],
            }
        ],
    )

    return response.choices[0].message.content


# ============================================================
# STYLE
# ============================================================

style_guidelines = """
## Style Guidelines

- Consistency between all images.
- Professional photography quality.
- Cinematic lighting.
- Blue and purple color palette.
- Warm accents.
- Futuristic but approachable mood.
- Educational theme.
- AI robots should look friendly.
- High quality rendering.
"""

rubric = """
Score image from 1-5.

1 = Poor
2 = Fair
3 = Good
4 = Very Good
5 = Excellent

Pass if score >= 3.

Always provide detailed feedback.
"""

# ============================================================
# OUTPUT MODEL
# ============================================================


class CritiqueImage(BaseModel):
    image_pass: bool
    feedback: str


# ============================================================
# MAIN
# ============================================================


async def main():

    image_description = (
        "an agent generating an image"
    )

    image_name = "agent_image_generation"

    feedback = ""

    generator = Agent(
        name="Prompt Generator",
        instructions=f"""
You improve prompts for image generation.

Use feedback from previous iterations.

Return only the final optimized prompt.

{style_guidelines}
""",
        model=OpenAIChatCompletionsModel(
            model=model_name,
            openai_client=client_agent,
        ),
    )

    critic = Agent(
        name="Image Critic",
        instructions=f"""
You critique generated images.

Use the describe_image tool.

{style_guidelines}

{rubric}
""",
        model=OpenAIChatCompletionsModel(
            model=model_name,
            openai_client=client_agent,
        ),
        tools=[describe_image],
        output_type=CritiqueImage,
    )

    with trace("Gemini Image Loop"):

        iteration = 0

        while True:

            iteration += 1

            print("\n")
            print("=" * 60)
            print(f"ITERATION {iteration}")
            print("=" * 60)

            prompt_result = await Runner.run(
                generator,
                f"""
Description:
{image_description}

Feedback:
{feedback}
""",
            )

            final_prompt = prompt_result.final_output

            print("\nPROMPT:")
            print(final_prompt)

            image_path = generate_image(final_prompt)

            final_image_path = os.path.join(
                "gen_images",
                f"{image_name}.png",
            )

            if os.path.exists(final_image_path):
                os.remove(final_image_path)

            os.rename(
                image_path,
                final_image_path,
            )

            critique_result = await Runner.run(
                critic,
                f"""
Evaluate image:

image_path={final_image_path}

original_request=
{image_description}
""",
            )

            critique = critique_result.final_output

            print("\nPASS:")
            print(critique.image_pass)

            print("\nFEEDBACK:")
            print(critique.feedback)

            if critique.image_pass:
                print("\nSUCCESS")
                print(
                    f"Final image stored in: {final_image_path}"
                )

                open_file(final_image_path)
                break

            feedback = critique.feedback


if __name__ == "__main__":
    asyncio.run(main())
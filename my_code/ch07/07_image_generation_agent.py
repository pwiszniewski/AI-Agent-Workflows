# nie mam zasobów
import os
import subprocess
import sys

from google import genai
from google.genai import types

import os
from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()    

model_name = 'gemini-3.1-flash-image'



def open_file(path: str) -> None:
    if sys.platform.startswith("darwin"):
        subprocess.run(["open", path], check=False)
    elif os.name == "nt":
        os.startfile(path)
    elif os.name == "posix":
        subprocess.run(["xdg-open", path], check=False)
    else:
        print(f"Don't know how to open files on this platform: {sys.platform}")


def main():
    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"]
    )

    image_description = """
    an agent generating an image

    Style Guidelines for All Images:

    - Consistency: photographic quality with 3D-rendered elements
    - Color Palette: blues, purples, warm golds, greens
    - Lighting: professional photography lighting
    - Mood: optimistic, educational, futuristic
    """

    image_name = "agent_image_generation"

    response = client.models.generate_content(
        model=model_name,
        contents=image_description,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"]
        ),
    )

    os.makedirs("gen_images", exist_ok=True)

    for part in response.candidates[0].content.parts:
        if part.inline_data:
            image_path = os.path.join(
                "gen_images",
                f"{image_name}.png"
            )

            with open(image_path, "wb") as f:
                f.write(part.inline_data.data)

            print(f"Saved image to: {image_path}")
            open_file(image_path)

        elif part.text:
            print(part.text)


if __name__ == "__main__":
    main()
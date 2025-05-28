from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from .model_utils import predict_scene
from .schemas import StoryOut
from PIL import Image
import io, openai, os

openai.api_key = os.getenv("OPENAI_API_KEY")
app = FastAPI(title="Story Generator API")

# Permite peticiones desde la app móvil
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ⬅️ pon dominio específico en prod
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.post("/generate", response_model=StoryOut)
async def generate_story(file: UploadFile, prompt: str = Form(...)):
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    scene, conf = predict_scene(image)

    full_prompt = (
        f"The story takes place near a {scene}, where "
        f"{prompt.strip().replace(chr(10), ' ')}"
    )

    gpt = openai.ChatCompletion.create(
        model="gpt-4o-mini",          # usa tu modelo preferido
        messages=[
            {"role": "system", "content": "You are a creative storyteller."},
            {"role": "user", "content": full_prompt},
        ],
        temperature=0.85,
        max_tokens=300
    )

    return {"scene": scene,
            "confidence": conf,
            "story": gpt.choices[0].message.content}

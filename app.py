from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model + tokenizer (Phi-2)
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

# Request schema
class CaptionRequest(BaseModel):
    prompt: str
    style: str = "default"
    hashtags: bool = False

# Style prompts
styles = {
    "edgy": "Make it raw, punchy, and cool.",
    "emotional": "Add deep emotions and sentimental vibes.",
    "luxury": "Make it premium, stylish, and classy.",
    "professional": "Keep it clean, corporate, and sharp.",
    "funny": "Add humor and playfulness.",
    "motivational": "Make it inspiring and powerful.",
    "default": "Create a creative Instagram caption."
}

@app.post("/generate")
async def generate_caption(data: CaptionRequest):
    style_prompt = styles.get(data.style.lower(), styles["default"])
    hashtags_prompt = " Add relevant Instagram hashtags at the end." if data.hashtags else ""
    full_prompt = f"{style_prompt} Caption for: '{data.prompt}'.{hashtags_prompt}"

    input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_length=80, do_sample=True, temperature=0.9, top_k=50, num_return_sequences=2)

    captions = [tokenizer.decode(out, skip_special_tokens=True).replace(full_prompt, "").strip() for out in output]
    return {"captions": captions}


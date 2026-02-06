from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# -----------------------
# App initialization
# -----------------------
app = FastAPI(title="Text Summarization App")

# CORS (safe for local + deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Device setup
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Load model & tokenizer
# -----------------------
MODEL_PATH = "t5_samsum"

tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# -----------------------
# Request schema
# -----------------------
class TextRequest(BaseModel):
    text: str

# -----------------------
# API endpoint
# -----------------------
@app.post("/summarize")
def summarize_text(request: TextRequest):
    input_text = "summarize: " + request.text

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=80,
            num_beams=4,
            early_stopping=True
        )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return {"summary": summary}

# -----------------------
# Serve frontend
# -----------------------
app.mount("/", StaticFiles(directory="static", html=True), name="static")

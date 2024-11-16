from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = AutoTokenizer.from_pretrained("TroyDoesAI/BlackSheep-4B")
# Set padding token to be the same as the EOS token
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("TroyDoesAI/BlackSheep-4B")
# Make sure the model knows about the padding token
model.config.pad_token_id = tokenizer.eos_token_id
# Resize model embeddings to match
model.resize_token_embeddings(len(tokenizer))


class GenerateRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate(request: GenerateRequest):
    prompt = str(request.prompt)
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,  # Adjust this number to control response length
            do_sample=True,      # Enable sampling
            temperature=0.7,     # Control randomness (0.0 = deterministic, 1.0 = very random)
            top_p=0.9           # Nucleus sampling parameter
        )
        return {"response": tokenizer.decode(outputs[0], skip_special_tokens=True)}
    except Exception as e:
        return {"error": str(e)}

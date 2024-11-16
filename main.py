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

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")


class GenerateRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate(request: GenerateRequest):
    prompt = str(request.prompt)
    print("Type of prompt:", type(request.prompt))
    print("Value of prompt:", request.prompt)
    try:
        print("prompt: ", prompt)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        print("inputs: ", inputs)
        outputs = model.generate(**inputs)
        print("outputs: ", outputs)
        return {"response": tokenizer.decode(outputs[0], skip_special_tokens=True)}
    except Exception as e:
        return {"error": str(e)}

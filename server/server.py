import sys
import re
from typing import List, Literal, Optional

import torch
import uvicorn
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline


app = FastAPI()


class MessageInput(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    name: Optional[str] = None


class MessageOutput(BaseModel):
    role: Literal["assistant"]
    content: str = None
    name: Optional[str] = None


class Choice(BaseModel):
    message: MessageOutput


class Request(BaseModel):
    messages: List[MessageInput]


class Response(BaseModel):
    model: str
    choices: List[Choice]


@app.post("/v1/chat/completions", response_model=Response)
async def create_chat_completion(request: Request):
    global pipe

    print(datetime.now())
    print("\033[91m--received_request\033[0m", request)
    messages = [message.model_dump() for message in request.messages]
    outputs = pipe(
        messages,
        max_new_tokens=128000,
    )

    result = outputs[0]["generated_text"][-1].get('content')
    result = re.split(r'assistantfinal(?: [Rr]esponse| JSON| json)?', result, maxsplit=1)[-1]
    print(datetime.now())
    print("\033[91m--generated_text\033[0m", result)

    message = MessageOutput(
        role="assistant",
        content=result,
    )
    choice = Choice(
        message=message,
    )
    response = Response(model=sys.argv[1].split("/")[-1].lower(), choices=[choice])
    return response


torch.cuda.empty_cache()

if __name__ == "__main__":
    MODEL_PATH = sys.argv[1]

    pipe = pipeline(
        "text-generation",
        model=MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
    )

    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)

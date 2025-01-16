import os
import torch

from typing import Dict, Optional, List
import logging
from pydantic import BaseModel
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from ray import serve

logger = logging.getLogger("ray.serve")

app = FastAPI()


class Prompt(BaseModel):
    text: str
    temperature: float = 0.2
    max_new_tokens: int = 512
    top_p: float = 0.7
    top_k: int = 50
    no_repeat_ngram_size: int = 4
    repetition_penalty: float = 1
    streaming: bool = False
    do_sample: bool = False


@serve.deployment(
    num_replicas="auto",
    ray_actor_options={"num_gpus": 1},
    max_ongoing_requests=5,
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 5,
        "target_num_ongoing_requests_per_replica": 2,
    }
)
@serve.ingress(app)
class LlamaModel:
    def __init__(self):
        logger.info("Initializing LlamaModel")
        try:
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                raise ValueError("HF_TOKEN environment variable is not set")

            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.2-1B-Instruct",
                use_auth_token=hf_token,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-3.2-1B-Instruct",
                torch_dtype=torch.float16,
                load_in_8bit=False,
                device_map="auto",
                use_auth_token=hf_token,
            )

            self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

            if torch.cuda.is_available():
                self.model.to("cuda")
                logger.info("Model moved to CUDA")
            else:
                logger.warning("CUDA is not available. Using CPU.")

            logger.info("LlamaModel initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LlamaModel: {str(e)}")
            raise

    @app.post("/generate")
    async def generate(self, prompt: Prompt):
        try:
            logger.info(f"Prompt Received")
            output = self.pipe(
                prompt,
                return_full_text=False,
                do_sample=True,
                max_new_tokens=150,
                temperature=0.2,
                top_p=0.7,
                top_k=50,
                pad_token_id=self.tokenizer.pad_token_id,
                early_stopping=False,
            )
            return output[0]["generated_text"]
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            return str(e)


app = LlamaModel.bind()

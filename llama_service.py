import ray
from ray import serve
from transformers import AutoTokenizer, AutoModelForCausalLM
from starlette.requests import Request
import logging
import os
import torch

logger = logging.getLogger(__name__)


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

            if torch.cuda.is_available():
                self.model.to("cuda")
                logger.info("Model moved to CUDA")
            else:
                logger.warning("CUDA is not available. Using CPU.")

            logger.info("LlamaModel initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LlamaModel: {str(e)}")
            raise

    async def __call__(self, request: Request):
        # Check if the request is JSON or form data
        prompt = await request.json()
        logger.info(f"Received prompt: {prompt}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
        output = self.model.generate(input_ids, max_length=100)
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return decoded_output


app = LlamaModel.bind()

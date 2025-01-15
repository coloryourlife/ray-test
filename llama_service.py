import ray
from ray import serve
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os


@serve.deployment(
    num_replicas="auto",
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 5,
        "target_num_ongoing_requests_per_replica": 10,
    }
)
class LlamaModel:
    def __init__(self):
        hf_token = os.environ.get("HF_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct",
            use_auth_token=hf_token,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct",
            use_auth_token=hf_token,
        )
        self.model.to("cuda")

        async def __call__(self, request):
            prompt = request.query_params["prompt"]
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
            output = self.model.generate(input_ids, max_length=100)
            return self.tokenizer.decode(output[0], skip_special_tokens=True)


app = LlamaModel.bind()

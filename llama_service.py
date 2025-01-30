import os

from typing import Dict, Optional
import logging

from fastapi import FastAPI
from fastapi.responses import Response
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

from ray import serve

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_models import (BaseModelPath,
                                                    OpenAIServingModels)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.protocol import LoadLoraAdapterRequest, UnloadLoraAdapterRequest
from vllm.utils import FlexibleArgumentParser

logger = logging.getLogger("ray.serve")

app = FastAPI()


@serve.deployment(name="VLLMDeployment")
@serve.ingress(app)
class VLLMDeployment:
    def __init__(
        self,
        model: str,
        engine_args: AsyncEngineArgs,
        response_role: str,
        chat_template: Optional[str] = None,
    ):
        logger.info(f"Starting with engine args: {engine_args}")
        self.engine_args = engine_args
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.openai_serving_models = OpenAIServingModels(
            engine_client=self.engine,
            model_config=self.engine.get_model_config(),
            base_model_paths=[BaseModelPath(name=model, model_path=model)],            # BaseModelPath(name=name, model_path=args.model)
            lora_modules=None,              # If we are loading lora module when initiate app
        )
        # await self.openai_serving_models.init_static_loras() if we are attatching lora upfront
        self.openai_serving_chat = OpenAIServingChat(
            engine_client=self.engine,
            model_config=self.engine.get_model_config(),
            models=self.openai_serving_models,
            response_role=response_role,
            chat_template=chat_template,
        )
        self.response_role = response_role
        self.chat_template = chat_template

    @app.post("/v1/load_lora_adapter")
    async def load_lora_adapter(self,
                                request: LoadLoraAdapterRequest,
                                raw_request: Request):
        response = await self.openai_serving_models.load_lora_adapter(request)
        if isinstance(response, ErrorResponse):
            return JSONResponse(content=response.model_dump(),
                                status_code=response.code)

        return Response(status_code=200, content=response)

    @app.post("/v1/unload_lora_adapter")
    async def unload_lora_adapter(self,
                                  request: UnloadLoraAdapterRequest,
                                  raw_request: Request):

        response = await self.openai_serving_models.unload_lora_adapter(request)
        if isinstance(response, ErrorResponse):
            return JSONResponse(content=response.model_dump(),
                                status_code=response.code)

        return Response(status_code=200, content=response)

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        logger.info(f"Request: {request}")
        generator = await self.openai_serving_chat.create_chat_completion(
            request, raw_request
        )
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())


def parse_vllm_args(cli_args: Dict[str, str]):
    parser = FlexibleArgumentParser(description="vLLM CLI")
    parser = make_arg_parser(parser)
    arg_strings = []
    for key, value in cli_args.items():
        if value == "":  # Handle flag arguments
            arg_strings.append(f"--{key}")
        else:
            arg_strings.extend([f"--{key}", str(value)])
    logger.info(arg_strings)
    parsed_args = parser.parse_args(args=arg_strings)
    return parsed_args


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    parsed_args = parse_vllm_args(cli_args)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    engine_args.worker_use_ray = True

    return VLLMDeployment.bind(
        model=parsed_args.model,
        engine_args=engine_args,
        response_role=parsed_args.response_role,
        chat_template=parsed_args.chat_template,
    )


model = build_app({
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "max-lora-rank": "32",
    "enable-lora": "",
    # "lora-modules": {
    #     "name": os.environ['DEPLOYMENT_ID'],  # lora_integer_id (globally unique)
    #     "path": os.environ['LORA_PATH'],
    #     "base_model_name": "meta-llama/Llama-3.2-1B-Instruct",
    # },
    "tensor-parallel-size": os.environ['TENSOR_PARALLELISM'],
    "pipeline-parallel-size": os.environ['PIPELINE_PARALLELISM']
})

# vllm serve meta-llama/Llama-3.2-1B-Instruct --enable-lora --lora-modules '{"name": "tr-CQjJeCUnYtuTT8g3yp6zu2-3",
# "path": "/home/ubuntu/artifacts", "base_model_name": "meta-llama/Llama-3.2-1B-Instruct"}' --max-lora-rank 32

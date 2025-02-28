import os
import aioboto3
import aiofiles
import asyncio
import boto3
import zipfile
import math

from typing import Dict, Optional

import ray
from typing_extensions import assert_never
import logging

from fastapi import FastAPI
from fastapi.responses import Response
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

from pydantic import Field

from ray import serve

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    CompletionRequest,
    CompletionResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_models import (BaseModelPath,
                                                    OpenAIServingModels)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_pooling import OpenAIServingPooling
from vllm.entrypoints.openai.protocol import (LoadLoraAdapterRequest,
                                              UnloadLoraAdapterRequest,
                                              PoolingRequest, PoolingResponse)
from vllm.utils import FlexibleArgumentParser

from safetensors.torch import load_file, save_file

logger = logging.getLogger("ray.serve")
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID', '')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY', '')

app = FastAPI()


class S3LoadLoraAdapterRequest(LoadLoraAdapterRequest):
    bucket: str
    key: str
    region: str = Field(default="us-east-1")


@ray.remote
class DownloadWorker:
    def __init__(self):
        self.session = aioboto3.Session()

    async def download_file(self, bucket, key, destination):
        async with self.session.client('s3') as s3:
            await s3.download_file(bucket, key, destination)
        return destination

    async def download_and_extract(self, bucket, key, local_lora_path):
        temp_zip_path = os.path.join(local_lora_path, "artifacts.zip")
        try:
            await self.download_file(bucket, key, temp_zip_path)

            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                zip_ref.extractall(local_lora_path)

            await self._verify_unzip_and_update_model_file(local_lora_path)

            os.remove(temp_zip_path)
            return local_lora_path
        except Exception as e:
            if os.path.exists(temp_zip_path):
                os.remove(temp_zip_path)
            raise ValueError(f"Failed to download and extract LoRA from S3: {str(e)}")

    @staticmethod
    async def _verify_unzip_and_update_model_file(local_lora_dir):
        lora_tensor_path = os.path.join(local_lora_dir, "adapter_model.safetensors")
        logger.info(f"Checking for adapter file at: {lora_tensor_path}")
        if not os.path.exists(lora_tensor_path):
            logger.error("There would be problem while unzipping")
            logger.error(
                f"Directory contents: {os.listdir(local_lora_dir) if os.path.exists(local_lora_dir) else 'directory does not exist'}")
            raise ValueError("There would be problem while unzipping")

        async with aiofiles.open(lora_tensor_path, 'rb') as f:
            model_state_dict = load_file(await f.read())
        try:
            lora_state_dict = {k: v for k, v in model_state_dict.items() if 'lora_' in k or '.alpha' in k}
            keys_to_remove = [key for key in lora_state_dict.keys() if
                              'model.embed_tokens.weight' in key or "lm_head" in key]
            for extra_key in keys_to_remove:
                del lora_state_dict[extra_key]

            async with aiofiles.open(lora_tensor_path, 'wb') as f:
                await f.write(save_file(lora_state_dict))

            logger.info(f"LoRA adapter file updated successfully. Kept {len(lora_state_dict)} LoRA-related keys.")
        except Exception as e:
            raise ValueError(f"Error during model update for filtering LoRA state dict: {str(e)}")


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
        self.model = model
        self.response_role = response_role
        self.chat_template = chat_template
        self.model_config = None
        self.openai_serving_models = None
        self.openai_serving_chat = None
        self.openai_serving_completion = None
        self.openai_serving_pooling = None
        self._is_initialized = False
        self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
        self.local_lora = set('/models/Llama-3.2-1B-Instruct')
        self.download_worker = DownloadWorker.remote()
    
    async def _ensure_initialized(self):
        if not self._is_initialized:
            await self.initialize()
    
    async def initialize(self):
        if self._is_initialized:
            return
        self.model_config = await self.engine.get_model_config()
        logger.info(f"Model configs: {self.model_config}")

        self.openai_serving_models = OpenAIServingModels(
            engine_client=self.engine,
            model_config=self.model_config,
            base_model_paths=[BaseModelPath(name=self.model, model_path=self.model)],
            lora_modules=None,
        )
        # await self.openai_serving_models.init_static_loras()
        self.openai_serving_chat = OpenAIServingChat(
            engine_client=self.engine,
            model_config=self.model_config,
            models=self.openai_serving_models,
            response_role="assistant",
            request_logger=None,
            chat_template=self.chat_template,
            chat_template_content_format="auto"
        )

        self.openai_serving_completion = OpenAIServingCompletion(
            engine_client=self.engine,
            model_config=self.model_config,
            models=self.openai_serving_models,
            request_logger=None,
        )

        self._is_initialized = True

    @app.post("/v1/load_lora_adapter")
    async def load_lora_adapter(self, request: S3LoadLoraAdapterRequest, raw_request: Request):
        await self._ensure_initialized()
        try:
            local_lora_path = os.path.abspath(os.path.join("/models", request.lora_path))
            if not os.path.exists(local_lora_path):
                os.makedirs(local_lora_path, exist_ok=True)
                # Use the DownloadWorker to download, extract, and verify asynchronously
                local_lora_path = await self.download_worker.download_and_extract.remote(
                    request.bucket, request.key, local_lora_path
                )

            lora_request = LoadLoraAdapterRequest(
                lora_name=request.lora_name,
                lora_path=local_lora_path
            )
            response = await self.openai_serving_models.load_lora_adapter(lora_request)
            self.local_lora.add(lora_request.lora_name)

            if isinstance(response, ErrorResponse):
                return JSONResponse(content=response.model_dump(), status_code=response.code)

            return Response(status_code=200, content=response)
        except ValueError as e:
            return JSONResponse(content={"error": str(e)}, status_code=400)

    @app.post("/v1/unload_lora_adapter")
    async def unload_lora_adapter(self,
                                  request: UnloadLoraAdapterRequest,
                                  raw_request: Request):
        logger.info(f"Request: {request}")
        await self._ensure_initialized()
        response = await self.openai_serving_models.unload_lora_adapter(request)
        self.local_lora.remove(request.lora_name)
        if isinstance(response, ErrorResponse):
            return JSONResponse(content=response.model_dump(),
                                status_code=response.code)

        return Response(status_code=200, content=response)

    @app.post("/v1/completions")
    async def create_completion(self, request: CompletionRequest, raw_request: Request):
        logger.info(f"Request: {request}")
        await self._ensure_initialized()
        generator = await self.openai_serving_completion.create_completion(request, raw_request)
        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(),
                                status_code=generator.code)
        elif isinstance(generator, CompletionResponse):
            return JSONResponse(content=generator.model_dump())

        return StreamingResponse(content=generator, media_type="text/event-stream")

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        logger.info(f"Request: {request}")
        await self._ensure_initialized()
        if request.model not in self.local_lora:
            await self.openai_serving_models.load_lora_adapter(
                LoadLoraAdapterRequest(
                    lora_name=request.model,
                    lora_path=f"/models/{request.model}",
                )
            )
            self.local_lora.add(request.model)
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
        if value == "":  # Handle flag arguments (e.g. --lora-enable)
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
    "model": "/models/Llama-3.2-1B-Instruct",
    "max-lora-rank": "32",
    "enable-lora": "",
    "tensor-parallel-size": os.environ['TENSOR_PARALLELISM'],
    "pipeline-parallel-size": os.environ['PIPELINE_PARALLELISM'],
    "load-format": "safetensors",   # auto, pt, safetensors, npcache, dummy, etc
    # "task": "classify",
})

# vllm serve meta-llama/Llama-3.2-1B-Instruct --enable-lora --lora-modules '{"name": "tr-CQjJeCUnYtuTT8g3yp6zu2-3",
# "path": "/home/ubuntu/artifacts", "base_model_name": "meta-llama/Llama-3.2-1B-Instruct"}' --max-lora-rank 32

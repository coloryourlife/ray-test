import os
import boto3
import zipfile

from typing import Dict, Optional
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
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_models import (BaseModelPath,
                                                    OpenAIServingModels)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.protocol import LoadLoraAdapterRequest, UnloadLoraAdapterRequest
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

    async def ensure_local_lora(self) -> "LoadLoraAdapterRequest":
        local_lora_path = os.path.abspath(os.path.join("/tmp", self.lora_path))
        logger.info(f"Local lora path: {local_lora_path}")
        if os.path.exists(local_lora_path):
            logger.info(f"{self.lora_name} exists")
            return LoadLoraAdapterRequest(
                lora_name=self.lora_name,
                lora_path=local_lora_path
            )
        if not self.key:
            logger.error(f"LoRA not found at {self.lora_path} and no S3 config provided")
            raise ValueError(f"LoRA not found at {self.lora_path} and no S3 config provided")

        # Create the directory path if it doesn't exist
        os.makedirs(local_lora_path, exist_ok=True)

        # Download from S3 directly to the specified path e.g. tmp/lora_path/artifacts.zip
        temp_zip_path = os.path.join(local_lora_path, "artifacts.zip")
        try:
            logger.info("Downloading Lora Modules from S3")
            s3_client = boto3.client('s3', region_name=self.region)
            s3_client.download_file(
                self.bucket,
                self.key,
                temp_zip_path,
            )

            if not os.path.exists(temp_zip_path):
                logger.error(f"Failed to download {self.key} from S3")
                raise ValueError(f"Failed to download {self.key} from S3")

            # Unzip the contents
            logger.info("Unzip artifacts")
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                # DEBUGGING
                contents = zip_ref.namelist()
                logger.info(f"Zip Contents: {contents}")
                # DEBUGGING END
                zip_ref.extractall(local_lora_path)
                logger.info(f"Extracted to {local_lora_path}")

            extracted_files = os.listdir(local_lora_path)
            logger.info(f"Extracted Files: {extracted_files}")

            self._verify_unzip_and_update_model_file(local_lora_path)
            
            # Remove temporary zip file
            logger.info(f"Remove {temp_zip_path} from the local.")
            os.remove(temp_zip_path)

        except Exception as e:
            # Clean up the zip file if it exists
            if os.path.exists(temp_zip_path):
                os.remove(temp_zip_path)
            raise ValueError(f"Failed to download LoRA from S3: {str(e)}")
        
        return LoadLoraAdapterRequest(
            lora_name=self.lora_name,
            lora_path=local_lora_path,
        )

    # TODO: Only verify unzip
    @staticmethod
    def _verify_unzip_and_update_model_file(local_lora_dir):
        lora_tensor_path = os.path.join(local_lora_dir, "adapter_model.safetensors")
        logger.info(f"Checking for adapter file at: {lora_tensor_path}")
        if not os.path.exists(lora_tensor_path):
            logger.error(f"There would be problem while unzipping")
            logger.error(f"Directory contents: {os.listdir(local_lora_dir) if os.path.exists(local_lora_dir) else 'directory does not exist'}")
            raise ValueError(f"There would be problem while unzipping")
        model_state_dict = load_file(lora_tensor_path)
        try:
            keys = model_state_dict.keys()
            keys_to_remove = [key for key in keys if 'model.embed_tokens.weight' in key or "lm_head" in key]

            # Remove the keys
            for extra_key in keys_to_remove:
                del model_state_dict[extra_key]

            # Save the updated model if any keys were removed
            save_file(model_state_dict, lora_tensor_path)
        except Exception as e:
            raise ValueError("Error during model update for deleting lm.head from lora")
        return
        

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
        self._is_initialized = False
        self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
    
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
        self._is_initialized = True

    @app.post("/v1/load_lora_adapter")
    async def load_lora_adapter(self,
                                request: S3LoadLoraAdapterRequest,
                                raw_request: Request):
        logger.info(f"Request: {request}")
        await self._ensure_initialized()
        try:
            lora_request = await request.ensure_local_lora()
            # if lora_path not exist -> load_lora from S3
            # need to update the request -> need a path to S3
            response = await self.openai_serving_models.load_lora_adapter(lora_request)
            if isinstance(response, ErrorResponse):
                return JSONResponse(content=response.model_dump(),
                                    status_code=response.code)

            return Response(status_code=200, content=response)
        except ValueError as e:
            return JSONResponse(
                content={"error": str(e)},
                status_code=400,
            )

    @app.post("/v1/unload_lora_adapter")
    async def unload_lora_adapter(self,
                                  request: UnloadLoraAdapterRequest,
                                  raw_request: Request):
        logger.info(f"Request: {request}")
        await self._ensure_initialized()
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
        await self._ensure_initialized()
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
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "max-lora-rank": "32",
    "enable-lora": "",
    "tensor-parallel-size": os.environ['TENSOR_PARALLELISM'],
    "pipeline-parallel-size": os.environ['PIPELINE_PARALLELISM']
})

# vllm serve meta-llama/Llama-3.2-1B-Instruct --enable-lora --lora-modules '{"name": "tr-CQjJeCUnYtuTT8g3yp6zu2-3",
# "path": "/home/ubuntu/artifacts", "base_model_name": "meta-llama/Llama-3.2-1B-Instruct"}' --max-lora-rank 32

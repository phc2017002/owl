# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========

import os
from typing import Any, Dict, List, Optional, Type, Union

from openai import AsyncOpenAI, AsyncStream, OpenAI, Stream
from pydantic import BaseModel

from camel.configs import QWEN_API_PARAMS, QwenConfig
from camel.messages import OpenAIMessage
from camel.models import BaseModelBackend
from camel.models._utils import try_modify_message_with_format
from camel.types import (
    ChatCompletion,
    ChatCompletionChunk,
    ModelType,
)
from camel.utils import (
    BaseTokenCounter,
    OpenAITokenCounter,
    api_keys_required,
)


class QwenModel(BaseModelBackend):
    r"""Qwen API in a unified BaseModelBackend interface.

    Args:
        model_type (Union[ModelType, str]): Model for which a backend is
            created, one of Qwen series.
        model_config_dict (Optional[Dict[str, Any]], optional): A dictionary
            that will be fed into:obj:`openai.ChatCompletion.create()`. If
            :obj:`None`, :obj:`QwenConfig().as_dict()` will be used.
            (default: :obj:`None`)
        api_key (Optional[str], optional): The API key for authenticating with
            the Qwen service. (default: :obj:`None`)
        url (Optional[str], optional): The url to the Qwen service.
            (default: :obj:`https://dashscope-intl.aliyuncs.com/compatible-mode/v1`)
        token_counter (Optional[BaseTokenCounter], optional): Token counter to
            use for the model. If not provided, :obj:`OpenAITokenCounter(
            ModelType.GPT_4O_MINI)` will be used.
            (default: :obj:`None`)
    """

    @api_keys_required(
        [
            ("api_key", "QWEN_API_KEY"),
        ]
    )
    def __init__(
        self,
        model_type: Union[ModelType, str],
        model_config_dict: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        token_counter: Optional[BaseTokenCounter] = None,
    ) -> None:
        if model_config_dict is None:
            model_config_dict = QwenConfig().as_dict()
        api_key = api_key or os.environ.get("QWEN_API_KEY")
        url = url or os.environ.get(
            "QWEN_API_BASE_URL",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        super().__init__(
            model_type, model_config_dict, api_key, url, token_counter
        )
        self._client = OpenAI(
            timeout=180,
            max_retries=3,
            api_key=self._api_key,
            base_url=self._url,
        )
        self._async_client = AsyncOpenAI(
            timeout=180,
            max_retries=3,
            api_key=self._api_key,
            base_url=self._url,
        )

    async def _arun(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
        r"""Runs inference of Qwen chat completion.

        Args:
            messages (List[OpenAIMessage]): Message list with the chat history
                in OpenAI API format.

        Returns:
            Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
                `ChatCompletion` in the non-stream mode, or
                `AsyncStream[ChatCompletionChunk]` in the stream mode.
        """
        request_config = self._prepare_request(
            messages, response_format, tools
        )

        response = await self._async_client.chat.completions.create(
            messages=messages,
            model=self.model_type,
            **request_config,
        )
        return response

    def _run(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
        import json
        print("Original messages:")
        for msg in messages:
            print(json.dumps(msg, indent=2))

        try:
            request_config = self._prepare_request(
                messages, response_format, tools
            )
            print("Request config:", json.dumps(request_config, indent=2))
        
            # For VL models, we need to completely restructure the messages
            if "vl" in self.model_type.lower():
                # Create a new list for formatted messages
                formatted_messages = []
            
                # Process each message
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                
                    # For system messages
                    if role == "system":
                        formatted_messages.append({"role": role, "content": content})
                    else:
                        # For user/assistant messages in VL format
                        formatted_messages.append({
                            "role": role,
                            "content": [{"type": "text", "text": content}]
                        })
            
            # Use the formatted messages
                messages = formatted_messages
        
            print("Final messages:")
            for msg in messages:
                print(json.dumps(msg, indent=2))

            # Make the API call with the properly formatted messages
            response = self._client.chat.completions.create(
                messages=messages,
                model=self.model_type,
                **request_config,
            )
        
            return response
        except Exception as e:
            print(f"Error in _run: {e}")
            print(f"Model type: {self.model_type}")
            print(f"Response format: {response_format}")
            print(f"Tools: {tools}")
            raise

    def _prepare_request(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        request_config = self.model_config_dict.copy()
    
        if tools:
            request_config["tools"] = tools
        elif response_format:
            # Check if any message contains the word 'json'
            has_json_word = any("json" in str(msg.get("content", "")).lower() for msg in messages)
        
            # Only set response_format to json_object if 'json' is mentioned in messages
            if has_json_word:
                try_modify_message_with_format(messages[-1], response_format)
                request_config["response_format"] = {"type": "json_object"}

        return request_config

    @property
    def token_counter(self) -> BaseTokenCounter:
        r"""Initialize the token counter for the model backend.

        Returns:
            OpenAITokenCounter: The token counter following the model's
                tokenization style.
        """

        if not self._token_counter:
            self._token_counter = OpenAITokenCounter(ModelType.GPT_4O_MINI)
        return self._token_counter

    def check_model_config(self):
        r"""Check whether the model configuration contains any
        unexpected arguments to Qwen API.

        Raises:
            ValueError: If the model configuration dictionary contains any
                unexpected arguments to Qwen API.
        """
        for param in self.model_config_dict:
            if param not in QWEN_API_PARAMS:
                raise ValueError(
                    f"Unexpected argument `{param}` is "
                    "input into Qwen model backend."
                )

    @property
    def stream(self) -> bool:
        r"""Returns whether the model is in stream mode, which sends partial
        results each time.

        Returns:
            bool: Whether the model is in stream mode.
        """
        return self.model_config_dict.get('stream', False)

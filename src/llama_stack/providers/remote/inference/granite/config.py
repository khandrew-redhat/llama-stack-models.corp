# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import Field  # pyright: ignore[reportMissingImports]

from llama_stack.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from llama_stack_api import json_schema_type


@json_schema_type
class GraniteConfig(RemoteInferenceProviderConfig):
    """Configuration for the IBM Granite inference endpoint."""

    base_url: str = Field(
        default="https://granite-3-3-8b-instruct--apicast-production.apps.int.stc.ai.prod.us-east-1.aws.paas.redhat.com:443",
        description="Base URL for IBM Granite API",
    )

    @classmethod
    def sample_run_config(
        cls,
        base_url: str = "${env.MODEL_API:=https://granite-3-3-8b-instruct--apicast-production.apps.int.stc.ai.prod.us-east-1.aws.paas.redhat.com:443}",
        api_key: str = "${env.USER_KEY:=}",
        **kwargs,
    ) -> dict[str, Any]:
        return {
            "base_url": base_url,
            "api_key": api_key,
        }

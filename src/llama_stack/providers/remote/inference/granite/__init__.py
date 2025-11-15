# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import GraniteConfig


async def get_adapter_impl(config: GraniteConfig, _deps):
    from .granite import GraniteInferenceAdapter

    impl = GraniteInferenceAdapter(config=config)
    await impl.initialize()
    return impl

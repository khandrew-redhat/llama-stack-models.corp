# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.distributions.template import BuildProvider, DistributionTemplate

from ..starter.starter import get_distribution_template as get_starter_distribution_template


def get_distribution_template() -> DistributionTemplate:
    template = get_starter_distribution_template(name="granite")
    template.description = "IBM Granite distribution for running Llama Stack with IBM's Granite 3.3-8B-Instruct model. This distribution provides access to IBM's internal Granite language model through API endpoints."

    # Add granite provider to inference providers
    template.providers["inference"] = [
        BuildProvider(provider_type="remote::granite"),
    ]
    
    # Add httpx as additional dependency for granite provider
    template.additional_pip_packages = template.additional_pip_packages or []
    template.additional_pip_packages.append("httpx")
    
    return template

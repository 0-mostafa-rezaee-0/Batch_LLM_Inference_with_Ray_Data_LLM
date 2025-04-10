#!/usr/bin/env python3
# Ray Serve LLM Deployment Example
# This script demonstrates how to deploy a Qwen model with Ray Serve
# using the LLMServer and LLMRouter components

from ray import serve
from ray.serve.llm import LLMConfig, LLMServer, LLMRouter

def main():
    # Configure the LLM deployment
    llm_config = LLMConfig(
        model_loading_config=dict(
            model_id="qwen-0.5b",
            model_source="Qwen/Qwen2.5-0.5B-Instruct",
        ),
        deployment_config=dict(
            autoscaling_config=dict(
                min_replicas=1, max_replicas=2,
            )
        ),
        # Pass the desired accelerator type (e.g. A10G, L4, etc.)
        accelerator_type="A10G",
        # You can customize the engine arguments (e.g. vLLM engine kwargs)
        engine_kwargs=dict(
            tensor_parallel_size=2,
        ),
    )

    # Deploy the application
    deployment = LLMServer.as_deployment(
        llm_config.get_serve_options(name_prefix="vLLM:")).bind(llm_config)
    llm_app = LLMRouter.as_deployment().bind([deployment])
    serve.run(llm_app)
    
    print("Ray Serve LLM application is running at http://localhost:8000/v1")
    print("You can query it using the OpenAI client with the model ID 'qwen-0.5b'")

if __name__ == "__main__":
    main() 
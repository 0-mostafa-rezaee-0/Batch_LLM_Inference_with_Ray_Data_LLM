#!/usr/bin/env python3
# Ray Data LLM Haiku Generation Example
# This script demonstrates how to use Ray Data LLM for batch inference
# with a Llama model to generate haikus

import ray
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor
import numpy as np

def main():
    # Configure the vLLM processor
    config = vLLMEngineProcessorConfig(
        model="meta-llama/Llama-3.1-8B-Instruct",
        engine_kwargs={
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 4096,
            "max_model_len": 16384,
        },
        concurrency=1,
        batch_size=64,
    )
    
    # Build the LLM processor with preprocessing and postprocessing functions
    processor = build_llm_processor(
        config,
        preprocess=lambda row: dict(
            messages=[
                {"role": "system", "content": "You are a bot that responds with haikus."},
                {"role": "user", "content": row["item"]}
            ],
            sampling_params=dict(
                temperature=0.3,
                max_tokens=250,
            )
        ),
        postprocess=lambda row: dict(
            answer=row["generated_text"],
            **row  # This will return all the original columns in the dataset.
        ),
    )

    # Create a dataset with a single prompt
    ds = ray.data.from_items(["Start of the haiku is: Complete this for me..."])

    # Process the dataset with the LLM processor
    ds = processor(ds)
    
    # Display the result
    ds.show(limit=1)

if __name__ == "__main__":
    main() 
# Scripts Directory

This directory contains Python scripts demonstrating the use of Ray Data LLM and Ray Serve LLM for batch inference and model deployment.

## Contents

- `ray_data_llm_haiku_example.py` - Demonstrates how to use Ray Data LLM for batch inference with a Llama model to generate haikus
- `ray_serve_llm_deployment.py` - Shows how to deploy a Qwen model with Ray Serve using the LLMServer and LLMRouter components
- `openai_client_query.py` - Illustrates how to query a Ray Serve LLM deployment using the OpenAI Python client

## Usage

### Ray Data LLM Example

This script demonstrates batch inference with Ray Data LLM:

```bash
python scripts/ray_data_llm_haiku_example.py
```

The script configures a vLLM processor with a Llama model, processes a simple dataset with a haiku prompt, and displays the result.

### Ray Serve LLM Deployment

This script shows how to deploy an LLM with Ray Serve:

```bash
python scripts/ray_serve_llm_deployment.py
```

The script deploys a Qwen model using Ray Serve with automatic scaling and provides an OpenAI-compatible API endpoint.

### OpenAI Client Query

This script demonstrates how to query a deployed model:

```bash
python scripts/openai_client_query.py
```

Note: This assumes that the Ray Serve LLM deployment (from `ray_serve_llm_deployment.py`) is already running.

## Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install ray openai
```

For the vLLM-based examples, you'll also need:

```bash
pip install vllm
``` 
# Notebooks Directory

This directory contains Jupyter notebooks demonstrating how to use Ray Data LLM for batch inference with large language models.

## Contents

- `batch_inference_example.ipynb` - Original notebook example for batch inference
- `ray_data_llm_test.ipynb` - A comprehensive test notebook that demonstrates Ray Data LLM functionality from basic to advanced usage
- `llama_batch_inference.ipynb` - Specialized notebook for running Meta Llama 3.1 models with Ray Data LLM

## Usage

To run these notebooks:

1. Make sure the Docker containers are running:
   ```bash
   cd docker
   docker-compose up -d
   ```

2. Open Jupyter Lab in your browser at http://localhost:8888

3. Navigate to the notebooks directory

4. Open the desired notebook and execute the cells sequentially

## Notebook Descriptions

### ray_data_llm_test.ipynb

This notebook provides a complete walkthrough of Ray Data LLM functionality:

1. Initializing Ray
2. Creating datasets for batch processing
3. Setting up LLM processors
4. Running batch inference
5. Examining and analyzing the results

The notebook is designed to run in environments with or without GPU support, with appropriate fallbacks for systems with limited resources.

### llama_batch_inference.ipynb

This notebook demonstrates how to use Meta Llama 3.1 models with Ray Data LLM for batch inference:

1. Configuring vLLM for optimal memory usage with Llama models
2. Setting up Hugging Face authentication to access the models
3. Building a processor with appropriate parameters for Llama
4. Running batch inference on diverse prompts
5. Analyzing results and monitoring resource usage

## Hardware Considerations for Llama Models

The Llama 3.1 models require significant GPU resources. Here are important considerations:

### Minimum Requirements for Llama-3.1-8B

- **GPU**: NVIDIA GPU with at least 8GB VRAM (RTX 2080 or better)
- **RAM**: 32GB system RAM recommended
- **Storage**: 50GB+ free space for model weights and caching

### Performance Optimization Tips

1. **Memory Management**:
   - Reduce `batch_size` (16 or lower for 8GB GPUs)
   - Lower `max_num_batched_tokens` (2048 instead of 4096)
   - Set `gpu_memory_utilization` to 0.85 or lower
   - Enable `enable_chunked_prefill` for memory efficiency

2. **Authentication**:
   - A Hugging Face token with access to Llama models is required
   - Set your token as an environment variable: `export HUGGING_FACE_HUB_TOKEN='your_token_here'`

3. **Troubleshooting**:
   - If you encounter OOM (out of memory) errors, try reducing batch_size further
   - Close other GPU-intensive applications while running inference
   - Monitor GPU memory usage with `nvidia-smi`

## Additional Resources

For more information, see the [Ray Data LLM Documentation](https://docs.ray.io/en/latest/ray-data/llm.html). 
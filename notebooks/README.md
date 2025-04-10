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

### Note on Temporary Files

When running the notebooks, Ray will create a `tmp` directory at the project root that contains:
- Session information and logs
- Runtime resources for Ray processes
- Metrics data

This directory:
- Is automatically generated and can be safely ignored
- Is excluded from version control
- May grow in size during extended notebook sessions
- Can be safely deleted when not running Ray if you need to clean up disk space

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

## Model Acquisition and Storage

### Obtaining Llama Models

1. **Access Requirements**:
   - You need a Hugging Face account with access to Meta Llama models
   - Apply for access at [Llama 3.1 Models Page](https://huggingface.co/meta-llama)
   - Accept the license agreement on Hugging Face

2. **Authentication**:
   - Create a Hugging Face token at https://huggingface.co/settings/tokens
   - Set the token as an environment variable:
     ```bash
     export HUGGING_FACE_HUB_TOKEN='your_token_here'
     ```
   - For Docker environments, add this environment variable to your container configuration

### Storage Location

Models are cached in these default locations:
- Hugging Face models: `~/.cache/huggingface/hub/`
- vLLM-specific files: `~/.cache/vllm/`

For Docker environments, consider mounting these directories as volumes for persistence:
```yaml
volumes:
  - ~/.cache/huggingface:/root/.cache/huggingface
  - ~/.cache/vllm:/root/.cache/vllm
```

### Storage Requirements

- **Llama-3.1-8B**: ~5-6GB for model weights + additional space for tokenizer files (recommend 10GB+ free space)
- **Llama-3.1-70B**: ~40GB for model weights (recommend at least 50GB free space)
- First-time downloads may require double the space temporarily during download and extraction
- Additional cache space is used during inference for KV caches and intermediate states

## Data Formats and Examples

The notebooks in this directory demonstrate various input data formats for batch processing with Ray Data LLM.

### Supported Input Formats

- **Python Dictionaries/JSON**: Most flexible format, used in most examples
  ```python
  # From ray_data_llm_test.ipynb
  questions = [
      "What is the capital of France?",
      "How many planets are in our solar system?",
      "What is 2+2?",
      "Tell me a joke."
  ]
  ds = ray.data.from_items([{"question": q} for q in questions])
  ```

- **Simple Text Lists**: For straightforward prompts
  ```python
  # From llama_batch_inference.ipynb
  ds = ray.data.from_items(["Start of the haiku is: Complete this for me..."])
  ```

- **External Files**: You can load data from CSV, JSON, or Parquet files
  ```python
  ds = ray.data.read_csv("your_questions.csv")
  # or
  ds = ray.data.read_json("your_prompts.json")
  ```

### Data Format Requirements

When creating your own datasets for the notebooks, ensure your data has the following format requirements:

1. **Field Naming**: The processor expects specific field names that match your preprocessing function
   ```python
   # If your preprocessing function expects "prompt"
   ds = ray.data.from_items([{"prompt": "Write a story"}, ...])
   
   # If your preprocessing function expects "question"
   ds = ray.data.from_items([{"question": "What is Ray?"}, ...])
   ```

2. **Batch Size Considerations**: 
   - Smaller batches (16-32 items) for limited GPU memory
   - Larger batches (64+ items) for better throughput on powerful GPUs
   - Total token count is more important than number of items

### Example Datasets

The notebooks include several example datasets:

1. **Simple Questions** (ray_data_llm_test.ipynb):
   - Basic question-answer format
   - Low token count per item
   - Good for testing model performance

2. **Creative Prompts** (llama_batch_inference.ipynb):
   - Longer generation tasks
   - Higher token count for responses
   - Tests model creative capabilities

3. **Technical Queries** (llama_batch_inference.ipynb):
   - Domain-specific questions about Ray, LLMs, etc.
   - Tests model knowledge in technical domains

### Working with Custom Data

To use your own data in these notebooks:

1. Prepare your data in one of the supported formats
2. Replace the example dataset creation with your own data:
   ```python
   # Instead of the example prompts
   # prompts = ["Write a haiku...", ...]
   
   # Load your own data
   import pandas as pd
   your_data = pd.read_csv("your_data.csv")
   ds = ray.data.from_pandas(your_data)
   ```
3. Adjust the preprocessing function to match your data fields
4. Run the batch processing as shown in the notebook examples

For large datasets, consider using data streaming or chunking techniques to process data in manageable portions.

## Model Management and CI/CD Considerations

When working with these notebooks in collaborative environments or CI/CD pipelines, consider these best practices:

### Model Distribution

- **Document Access Requirements**: Don't store model weights in version control
- **Version Control**: Specify exact model versions in notebooks
  ```python
  model="meta-llama/Llama-3.1-8B-Instruct"  # Good - specific version
  model="meta-llama/Llama"                  # Bad - no version specified
  ```
- **Environment Variables**: Store tokens securely
  ```python
  # Use environment variables
  token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
  
  # Don't hardcode tokens in notebooks
  # token = "hf_AbCdEfGhIjKlMnOpQrStUv"  # Bad practice
  ```

### CI/CD Setup

- **Test with Smaller Models**: For CI pipelines, consider using smaller models
  ```python
  # For testing or CI
  model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  ```
- **Cache Persistence**: Configure persistent cache volumes
  ```yaml
  # In docker-compose.yml
  volumes:
    - huggingface-cache:/root/.cache/huggingface
  ```
- **Resource Limits**: Set appropriate resource limits for CI environments
  ```yaml
  resources:
    limits:
      memory: 16G
      nvidia.com/gpu: 1
  ```

### Team Collaboration

- **Share Cache Location**: Consider a shared model cache for teams
  ```python
  # Set custom cache location
  os.environ["TRANSFORMERS_CACHE"] = "/team/shared/model-cache"
  ```
- **Notebook Parameters**: Use notebook parameters for flexibility
  ```python
  # At the top of notebooks:
  # Parameters
  MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # @param
  BATCH_SIZE = 16  # @param
  ```

For more detailed guidelines, see the Model Management section in the main README.md.

## Additional Resources

For more information, see the [Ray Data LLM Documentation](https://docs.ray.io/en/latest/ray-data/llm.html). 
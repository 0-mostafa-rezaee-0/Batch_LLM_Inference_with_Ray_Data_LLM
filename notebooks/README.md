# Notebooks Directory

This directory contains Jupyter notebooks demonstrating how to use Ray Data LLM for batch inference with large language models.

## Contents

- `batch_inference_example.ipynb` - Original notebook example for batch inference
- `ray_data_llm_test.ipynb` - A comprehensive test notebook that demonstrates Ray Data LLM functionality from basic to advanced usage

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

## Additional Resources

For more information, see the [Ray Data LLM Documentation](https://docs.ray.io/en/latest/ray-data/llm.html). 
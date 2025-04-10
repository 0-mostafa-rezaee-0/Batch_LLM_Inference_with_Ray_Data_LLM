#!/usr/bin/env python3
# OpenAI Client Query Example
# This script demonstrates how to query a Ray Serve LLM deployment
# using the OpenAI Python client

from openai import OpenAI

def main():
    # Initialize client
    # Note: This assumes that a Ray Serve LLM application is already running
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="fake-key")

    # Basic chat completion with streaming
    print("Sending query to the LLM server...")
    response = client.chat.completions.create(
        model="qwen-0.5b",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True
    )

    print("Response from LLM server:")
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)
    
    print("\nQuery completed.")

if __name__ == "__main__":
    main() 
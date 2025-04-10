# Technical Jargon Glossary

## A

### Ad-hoc solutions
Approaches or fixes created specifically for a particular problem or situation without consideration of wider application or reusability. These solutions are typically created quickly to address immediate needs, often without following standardized procedures or design patterns. While they can solve urgent problems efficiently, ad-hoc solutions may lack scalability, documentation, and maintenance considerations, potentially creating technical debt over time.

### Author multi-model
A system or framework that can work with multiple different AI models simultaneously or interchangeably. In an authoring context, it refers to the ability to create, manage, or deploy content or applications that utilize different AI models based on specific needs, with each model potentially specializing in different tasks or domains. This approach allows for greater flexibility, as the system can leverage the strengths of various models rather than being limited to a single model's capabilities.

## B

### Boilerplate
Standardized, repetitive code or text that is used with minimal modification across multiple applications or projects. In programming, it's the routine code that needs to be included in many places with little or no alteration, often considered tedious to write and maintain. Reducing boilerplate code is usually seen as a way to improve developer productivity and code maintainability.

## F

### Fallback mechanisms
Contingency procedures or systems that activate when a primary system fails or encounters an error. In software development, fallback mechanisms provide alternative pathways to maintain functionality when preferred methods are unavailable. They often involve detecting failures, switching to backup systems or simpler versions of functionality, and handling degraded performance gracefully. Well-designed fallback mechanisms are essential for building resilient applications that can handle unpredictable failures while maintaining acceptable service levels.

## K

### Kubernetes
An open-source container orchestration platform for automating the deployment, scaling, and management of containerized applications. It provides a framework to run distributed systems efficiently, handling tasks like load balancing, storage orchestration, secret management, and automatic scaling. Kubernetes organizes containers into logical units called pods, manages their lifecycle, and ensures they run as expected across a cluster of machines. It has become the industry standard for container orchestration, enabling consistent application deployment across various environments.

### KubeRay
An open-source project that enables deploying Ray clusters on Kubernetes infrastructure. It simplifies running distributed Ray applications in Kubernetes environments by providing custom resource definitions (CRDs) and operators that automate the provisioning, scaling, and management of Ray clusters. KubeRay bridges the gap between Ray's distributed computing capabilities and Kubernetes' container orchestration, allowing data scientists and engineers to leverage both technologies efficiently without needing extensive knowledge of Kubernetes internals.

## L

### LoRA adapters
Low-Rank Adaptation adapters are lightweight, trainable modules that enable efficient fine-tuning of large language models. They work by decomposing weight updates into low-rank matrices that are applied to the model's existing weights, significantly reducing the number of parameters that need to be trained or stored. This approach allows for customizing model behavior for specific tasks or domains while maintaining most of the original model's parameters frozen, resulting in faster training, smaller storage requirements, and easier deployment compared to full fine-tuning.

## M

### Multi-LoRA
A technique that allows multiple LoRA (Low-Rank Adaptation) modules to be applied simultaneously to a language model. LoRA is a parameter-efficient fine-tuning method that reduces the number of trainable parameters by adding low-rank matrices to specific layers of the model. Multi-LoRA extends this by enabling the combination of multiple specialized LoRA adapters (each trained for different tasks or domains) at inference time, allowing a single base model to leverage multiple adaptations concurrently without interference.

## O

### OpenAI-compatible router
A system component that directs API requests to various LLM backends while maintaining the OpenAI API format. It allows applications built for OpenAI's API to seamlessly work with alternative model providers or self-hosted models without code changes. These routers typically handle authentication, load balancing, fallback mechanisms, and can route requests based on criteria like cost, performance needs, or specific model capabilities.

## Q

### Qwen model
A family of large language models developed by Alibaba Cloud (also known as Tongyi Qianwen). These models are designed to understand and generate human language, and have been trained on vast datasets of text in multiple languages including English and Chinese. Qwen models are capable of various natural language processing tasks like text generation, summarization, translation, and question answering. They are available in different sizes, ranging from more compact versions suitable for deployment in resource-constrained environments to larger models with enhanced capabilities.

## R

### Ray Data LLM (offline Ray)
An API within the Ray ecosystem that enables offline batch inference with Large Language Models (LLMs) as part of existing Ray Data pipelines. It provides optimized methods for processing large volumes of data through LLMs in a distributed and efficient manner.

### Ray Serve LLM (online Ray)
An API within the Ray ecosystem for deploying Large Language Models (LLMs) for online inference in Ray Serve applications. It facilitates real-time serving of LLMs with features designed for production deployment, scalability, and high-performance request handling.

## T

### Throughput
The amount of work or data that can be processed within a given time period. In computing and systems, it measures how many operations, transactions, or data units can be handled per unit of time. Higher throughput generally indicates better system performance.

## V

### vLLM
Versatile Large Language Model - an open-source library for efficient LLM inference and serving. It's designed to optimize memory usage and accelerate inference speed through techniques like PagedAttention, which enables more efficient memory management during model execution. vLLM is popular for deploying and scaling large language models in production environments due to its high throughput and optimized GPU utilization.

## Combinations & Integrations

### vLLM + Ray
The integration of vLLM (Versatile Large Language Model) with Ray's distributed computing framework. This combination leverages vLLM's optimized LLM inference capabilities with Ray's infrastructure for distributed computing. It enables highly scalable AI deployments for both batch processing (via Ray Data) and real-time serving (via Ray Serve) of large language models with improved resource utilization, higher throughput, and simplified deployment across clusters of machines.

# Regular Vocabulary

## C

### Contingency procedures or systems
Pre-planned methods and processes designed to respond to failures, emergencies, or unexpected events. These backup plans specify alternative actions and resources to maintain essential operations when primary systems or methods are unavailable. They typically include clearly defined steps for detecting problems, notification protocols, recovery procedures, and defined roles and responsibilities. Effective contingency systems include regular testing, frequent updates to reflect changing conditions, and integration with broader risk management strategies.

## E

### Entail
To involve or imply as a necessary consequence. In technical contexts, it often describes when one condition necessarily leads to or requires another condition or outcome.

## I

### Imperative
A style of programming or process that explicitly states the commands or steps to be performed in a specific sequence. In programming contexts, imperative code focuses on how to achieve a result through detailed step-by-step instructions, as opposed to declarative approaches that focus on what result is desired. Imperative programming is characterized by statements that change a program's state, with control flow being managed through conditionals, loops, and function calls.

### Instantiate
To create an instance or concrete occurrence of something. In programming, it refers to creating an actual object from a class or template, giving it specific values and allocating memory for it.

### Invoke
To call upon, activate, or put into effect something such as a function, method, or procedure. In programming, it refers to the action of executing or running a piece of code, function, or service. When you invoke a function, you're requesting that the system execute the instructions contained within that function, often with specific parameters or arguments.

## U

### Uptick
A small increase or upward movement in a measurement, value, or activity. In business and technical contexts, it often describes a modest but noticeable rise in metrics like usage statistics, performance indicators, or market values. For example, "There was an uptick in server response times after the update" means there was a small increase in how long the server took to respond.

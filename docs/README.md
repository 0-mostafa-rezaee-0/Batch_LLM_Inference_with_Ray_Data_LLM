# Documentation Directory

This directory contains supplementary documentation and reference materials for the Batch LLM Inference with Ray Data LLM project.

## Contents

- `reference_article.md` - Markdown version of the Anyscale article about Ray Data LLM and Ray Serve LLM APIs
- `reference_article.pdf` - PDF version of the Anyscale article about Ray Data LLM and Ray Serve LLM APIs
- `jargon.md` - A comprehensive glossary of technical terms and concepts used in this project, including Ray, Kubernetes, LLM-related terminology, and more.

## References and Important Links

1. [Linkedin announcement](https://www.linkedin.com/posts/robert-nishihara-b6465444_new-native-llm-apis-in-ray-data-and-ray-activity-7313348349699440643-R_aN/?utm_medium=ios_app&rcm=ACoAAAnEjvEBwXqQuBVVjxXuQ3cffPucbl2WqbM&utm_source=social_share_send&utm_campaign=copy_link)
2. [Announcing Native LLM APIs in Ray Data and Ray Serve](https://www.anyscale.com/blog/llm-apis-ray-data-serve)

## Reference Content

### LinkedIn Announcement

[Robert Nishihara](https://www.linkedin.com/in/robert-nishihara-b6465444?miniProfileUrn=urn%3Ali%3Afsd_profile%3AACoAAAlZvnwBU_2hc9u5bqpEN7IL4B2SvrM8SUA) - Co-founder at Anyscale

- If you're using vLLM + Ray for batch inference or online serving, check this out. We're investing heavily in making that combination work really well.

[Kourosh Hakhamaneshi](https://www.linkedin.com/feed/update/urn:li:activity:7313262118726705155/) - AI lead @Anyscale, PhD UC Berkeley

Announcing native LLM APIs in Ray Data and Ray Serve Libraries. These are experimental APIs we are announcing today that abstract two things:

1. **Serve LLM**: Simplifies the deployment of LLM engines (e.g. vLLM) through ray serve APIs. Enables features like auto-scaling, monitoring, LoRA management, resource allocation etc.
2. **Data LLM**: Helps you scale up offline inference horizontally for throughput sensitive applications (e.g. data curation, evaluation, etc). Ray data's lazy execution engine helps you pipeline complex heterogenous stages that involve LLMs.

### Key Features of Ray Data LLM and Ray Serve LLM

#### Ray Data LLM
- Integration with key LLM inference engines
- High-throughput batch inference
- Compatibility with OpenAI API endpoints
- Integration with existing Ray Data pipelines

#### Ray Serve LLM
- Automatic scaling and load balancing
- Unified multi-node multi-model deployment
- OpenAI compatibility
- Multi-LoRA support with shared base models
- Deep integration with inference engines (vLLM)
- Composable multi-model LLM pipelines

## Usage

The documentation in this directory provides valuable background information and technical details for implementing batch LLM inference with Ray Data LLM. The main examples and implementation details are in the repository's root directory.

## Additional Resources

- [Ray Data LLM Documentation](https://docs.ray.io/en/latest/data/working-with-llms.html)
- [Ray Serve LLM Documentation](https://docs.ray.io/en/master/serve/llm/serving-llms.html)
- [Ray Serve Production Guide](https://docs.ray.io/en/master/serve/production-guide/index.html)

# Multimodal Insurance Claims Agent Engine

An advanced, automated claims verification system designed to evaluate asset damage (cars, laptops, packages), cross-reference behavioral customer risk history, and output structured verification metrics.

### Key Architectural Features:
- **Resilient Fallback Engine:** Features localized deterministic string parsing to handle external API rate limits or missing dataset schemas gracefully.
- **Taxonomy Enforcement:** Guarantees strict structured alignment with target evaluation criteria.
- **Dynamic VRAM Scaling:** Implements automated image downsampling and VRAM clearing catches to process high-resolution assets smoothly without throwing CUDA Out-of-Memory (OOM) errors on T4 GPU hardware.

### Project Layout:
- **agent.py:** Core inference agent running an offline Qwen2-VL-2B-Instruct model with memory safety boundaries.
- **main.py:** Production execution pipeline parsing user claims, fault-tolerant historical datasets, and rule engines.
- **output.csv:** Dynamically compiled verification output metrics containing all 44 validated schema rows.

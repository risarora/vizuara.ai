# How the VLLM inference engine works?
https://youtu.be/QyHHbeXqgrQ?si=lH50by6yJ-UiDBph


### Intro
This summary distills a detailed lecture on how production-level inference engines for large language models (LLMs) operate, with a focus on the VLM inference system. It covers the journey of a prompt from arrival to token generation, highlighting core concepts that enable high-throughput, memory-efficient inference. The goal is to build a visual mental model of the internal steps, the role of KV caches, and the memory management strategies that power modern inference at scale.

### Center
- _Premise and motivation_: Inference engines must minimize GPU resource use to control cost. The lecturer examines what happens to prompt tokens during production-level inference, using VLM as a representative, open-source system.
- _Key terms you should know_: 
  - Inference engine, tokens, tokenization, KV cache (Key-Value cache), attention mechanism, context vector.
  - “Blocking” in memory: KV blocks store key/value matrices per transformer layer.
  - “Past work”: Pre-training from scratch is distinct from inference engineering innovations such as paged KV cache and continuous batching.
- _Overall flow: from prompt to completion_:
  - Arrival as a prompt: Three prompts are fed into VLM (P1: hi my name is; P2: today is a beautiful summer day; P3: hello there).
  - Tokenization: English prompts convert to token IDs via the model’s tokenizer (subword/BPE schemes common in modern LLMs).
  - Waiting queue and KV cache concept: Tokens enter a waiting queue where prefill or decoding decisions are made.
  - Prefill stage: Compute and cache the key/value matrices for the initial tokens; store in blocks on GPU VRAM.
  - Decoding stage: Generate new tokens iteratively, reusing cached KV states to avoid recomputing keys/values.
  - KV cache blocks: Each block stores up to 16 tokens per layer, across all transformer layers. Blocks are allocated per prompt to avoid cross-prompt attention leakage.
  - Continuous batching: Process multiple prompts together in a single batch when token budgets permit, enabling higher throughput.
  - Page KV cache (paged attention): A memory management scheme that uses a CPU-side “free block” queue and a GPU-side block pool to reuse blocks efficiently, reducing wasted GPU memory.
  - Token budgets and constraints: A typical token budget per iteration is 2048; prompts with total tokens less than this can be processed in one prefill/decode cycle.
  - CPU/GPU memory interplay: Model weights and optimizers require GPU memory; activations and occasional offloading to CPU memory are managed to optimize throughput and cost.
  - End conditions: A prompt ends when an end-of-sequence token is generated or a maximum token limit is reached; its blocks are freed and returned to the free pool.
- _Core mechanisms explained_:
  - KV cache: Inference through transformer layers requires repeated access to previously computed keys and values. The KV cache stores these values so that subsequent token predictions reuse prior computations, dramatically reducing compute.
  - Blocks: A KV block holds 16 token rows for keys and 16 for values per transformer layer. With many layers, many blocks are needed; total blocks are bounded by GPU memory and CPU swap space.
  - Blocks and memory layout: The GPU VRAM is conceptually divided into KV blocks (green tiles), model weights (blue), activations, and non-Torch memory (orange). Each block occupies a calculable amount of memory based on token count, number of heads, head dimension, and bytes per parameter.
  - Page table and free-block queue: The page table maps prompts to blocks; the free-block queue tracks available blocks per layer. This enables dynamic reassignment of blocks as prompts progress or finish.
- _Concrete numbers and relationships_:
  - Example prompts: P1 has 5 tokens, P2 has 7, P3 has 2. Each fits within a single 16-token block per layer in the simplified scenario.
  - Blocks per layer: For four transformer layers, 12 blocks are used (3 prompts × 4 layers). Across all layers, a total of 12 blocks are active for these prompts.
  - Maximum potential KV blocks: With a budget of about 10.86 GB for KV cache and ~34 MB per block, the theoretical maximum is roughly 32350 KV blocks; in practice, CPU-swap blocks (approx. 11,915) also factor into the resource accounting.
  - Continuous batching versus static batching: Continuous batching keeps memory and compute filled by streaming prompts, whereas static batching can leave memory idle between prompts. Continuous batching with paged KV cache minimizes idle memory and boosts throughput.
- _Practical implications for engineers_: 
  - Understanding these mechanisms helps reduce inference costs, optimize hardware usage, and contribute to innovations in latency and throughput for production LLMs.
  - The techniques discussed—paged KV cache, continuous batching, and the distinction between prefill and decoding—are foundational for more advanced topics like prefix caching, speculative decoding, and dynamic multi-model serving that will be explored in later lectures.

### Outro
- The lecture closes by inviting a mental exercise: close your eyes and trace the token’s journey through the inference engine, from arrival to completion. The presenter emphasizes that while the outward experience (prompt → output) may feel instantaneous to users, a sophisticated orchestration of tokenization, caching, memory management, and batched computation lies beneath.
- The speaker teases future topics, including prefix caching, speculative decoding, and multi-model dynamic serving, while reiterating the importance of mastering paged KV cache and continuous batching as the stepping stones to understanding modern inference systems.
- Final takeaway: Modern LLM inference hinges on intelligent memory layout and scheduling, where the KV cache and paging strategies unlock high throughput with constrained GPU memory, enabling scalable, cost-effective, production-grade AI services.

# Code Examples
```python
# 1. Install vLLM
!pip install -q vllm

# 2. Import vLLM
from vllm import LLM, SamplingParams

# 3. Define prompts
prompts = [
    "Hi, my name is ...",
    "Today is a beautiful summer day ...",
    "Hello there",
]

# 4. Define sampling parameters
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=50,  # number of tokens to generate
)

# 5. Run generation
llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

outputs = llm.generate(prompts, sampling_params)

# 6. Print results
for i, output in enumerate(outputs):
    print(f"\nPrompt {i+1}: {prompts[i]}")
    print(f"Completion: {output.outputs[0].text}")


```
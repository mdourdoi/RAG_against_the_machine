*This project has been created as part of the 42 curriculum by mdourdoi.*

# RAG Against the Machine

## Description

A Retrieval-Augmented Generation system that answers questions about the [vLLM](https://github.com/vllm-project/vllm) codebase. The system indexes the repository, searches it with BM25 to find relevant chunks for a given question, then asks Qwen3-0.6B to produce an answer grounded in the retrieved snippets.

## Instructions

You need Python 3.13+ and `uv`. A CUDA GPU is recommended (CPU works but is slow).

```bash
# Install dependencies
make install

# Index the repo (expects data/raw/vllm-0.10.1/)
uv run python -m src index --max_chunk_size 2000

# Search a dataset
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results

# Generate answers from search results
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer

# Evaluate retrieval against ground truth
uv run python -m src evaluate \
    --student_answer_path data/output/search_results/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

For single queries: `uv run python -m src search "..." --k 5` or `uv run python -m src answer "..." --k 10`.

`make lint` runs flake8 and mypy with the required flags. `make clean` removes caches and the virtual env.

## System Architecture

```
src/
├── __main__.py            # CLI (Python Fire)
├── data_ingestion/
│   ├── FilesGetter.py            # walks the repo, reads files
│   ├── ChunksCreator.py          # language-aware chunking
│   └── KnowledgeBaseBuilder.py   # builds & saves the BM25 index
├── search/
│   ├── SearchEngine.py           # loads the index, runs queries
│   └── Searcher.py               # batch search over a dataset
├── llm/
│   └── Generator.py              # Qwen3-0.6B answer generation
└── models/                # Pydantic models
```

The flow is straightforward: `FilesGetter` reads the repo → `ChunksCreator` splits files into chunks → `KnowledgeBaseBuilder` builds the BM25 index with an augmented version of the chunk (explained in the Retrieval Method section) and saves it to `data/processed/`. At query time, `SearchEngine` loads the index, returns the top-k chunks, and `Generator` feeds them to the model along with the question with a preprompt to produce a minimal but precise answer.

## Chunking Strategy

The chunker picks a splitter based on the file extension:

- `.py` files use LangChain's Python-aware splitter (breaks on `class`/`def`/blank lines).
- `.md` files use the Markdown-aware splitter (breaks on headings first).
- Everything else uses a generic recursive character splitter.

Max chunk size is configurable via `--max_chunk_size` (default 2000, minimum 800 to avoid bad chunking). Chunks overlap by 20% to reduce the risk of cutting an answer in half.

Each chunk stores its file path and the character offsets in the original file, which is what the evaluation needs to check overlap with ground truth.

## Retrieval Method

BM25 via the `bm25s` library, with `k1=2.5` (less penalty for greater word frequences) and `b=0.9` (little penalty for long chunks), an English stemmer and English stopword filtering.

Before indexing, each chunk's text is preprocessed to help BM25 match natural-language queries against code:

- The file path is included five times at the top of the indexed text, with `_`, `.`, `/`, `-` replaced by spaces, so a query mentioning a module or filename ranks the matching chunks higher.
- Inside the chunk content, `CamelCase` is split into separate tokens (e.g. `CamelCase` → `Camel Case`) and underscores are turned into spaces (e.g. `get_supported_mm_limits` → `get supported mm limits`). This way an English question can match identifiers without needing the exact symbol name.

The same CamelCase/separator normalization is applied to incoming questions before tokenization to match the indexed text.

The chunk content returned by the retriever is always the original text untouched, the preprocessing only affects what the BM25 tokenizer sees.

## Performance Analysis

The targets defined in the subject are:

- recall@5 ≥ 80% on docs questions, ≥ 50% on code questions : We are around 85% on docs and 80% on code, depending on the max_chunk_size.
- indexing ≤ 5 minutes : chromaDB and embeddings were less effective than just BM25 which take about a second to index everything with the recall@5 target. 
- cold-start retrieval ≤ 60 s, warm retrieval ≤ 90 s for 1000 questions : we didn't have a set of 1000 questions, but the retrieval for the 100 given as an example take at most 5 seconds.


Where the system works best: questions that share vocabulary with the code (method names, config keys, filenames). The path-boost in the indexed text was proven to be very effective on this matter.

Where it struggles: pure-natural-language questions with no identifier overlap. BM25 has no semantic understanding, so a question like "how does the scheduler decide what to run next" without any code-specific term retrieves less precisely than one that mentions `Scheduler`. The CamelCase splitting recovers some of this, but not all of it.

## Design Decisions

- **BM25 via `bm25s`.** Lightweight, fast, easy to serialize to disk, no heavy dependencies.
- **LangChain text splitters.** Their language-aware separators are well-tuned, no point reinventing them.
- **Pydantic on every data boundary.** Catches malformed JSON immediately instead of letting type bugs propagate.
- **`MinimalSource.content` is `Optional[str]`.** The subject defines `MinimalSource` with three fields. I extended it with a `content` field so the generator can read the chunk text directly from the search results, but the field is optional so files produced by the evaluation moulinette (without `content`) still load correctly.
- **`StudentSearchResults` and `StudentSearchResultsAndAnswer`** Modification from list to Sequence because it creates a mypy error because `List[MinimalAnswer]` is not a inherited type of `List[MinimalSearchResults]`.
- **Local inference via `transformers` rather than vLLM.** Qwen3-0.6B in bfloat16 on a single GPU is fast enough; no need to pull in vLLM as a dependency for the mandatory part.
- **`enable_thinking=False` on Qwen3.** The thinking mode generates a long internal monologue that I then strip out anyway. Disabling it makes generation much faster with no observable quality loss on this task at this model size.

## Challenges Faced

- **`mypy` and `transformers` stubs.** The type stubs around `AutoModelForCausalLM.from_pretrained` and `model.generate` are imprecise. Rather than scattering `# type: ignore`, I defined a small `Protocol` (`_CausalLM`) that declares the exact shape my code needs, and cast `self.model` to that Protocol at the call site. mypy is happy, and the code is explicit about what it actually depends on.
- **Pydantic list invariance.** As explain in the design decisions, `mypy` flags `List[MinimalAnswer]` as not a subtype of `List[MinimalSearchResults]` because `list` is invariant in Python's type system. Switched the relevant fields to `Sequence`, which is covariant.
- **Index accuracy.** At first, the index was terrible, with or without chromaDB. The augmenting described in the retrieval method is the result of a lot of trial and error.
- **LLM answering.** The way to retrieve the answer to a prompt differs completely from call_me_maybe and needed more work than I imagined but opened ways to be more accurate on what you 're prompting and how you do it.

## Resources

### References used

- [`bm25s` library](https://github.com/xhluca/bm25s) : the retrieval backend.
- [LangChain text splitters](https://python.langchain.com/docs/concepts/text_splitters/) : for the chunking strategies.
- [Qwen3-0.6B on Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B) : the model card and chat template documentation.
- [vLLM project](https://github.com/vllm-project/vllm) : the codebase being indexed.

### Use of AI

I used an AI assistant during the project for:

- **Documentation helper** : having the assistant explain how the libraries work with uncorrelated examples on top of the documentation.
- **Refining prompts** : create a system prompt for the LLM not to hallucinate, 
- **Debugging help** : investigating the chain of `mypy` errors around the `transformers` library.

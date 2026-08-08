FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Cap native thread pools — torch/spaCy default to one pool per CPU core,
    # which is wasted RAM for our single-message-at-a-time workload.
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    # Bound glibc per-thread malloc arenas. Without this, every worker thread
    # that runs torch inference gets its own arena that grows and is never
    # returned to the OS, ratcheting RSS up to the container limit.
    MALLOC_ARENA_MAX=2

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Presidio needs a spaCy model at runtime. en_core_web_lg matches the
# evaluator's detection behavior; swap to en_core_web_sm to cut ~700MB RSS.
RUN python -m spacy download en_core_web_lg

# Bake the sentence-transformer encoders into the image. These were previously
# fetched lazily on first use, which put a multi-hundred-MB download inside the
# request path — a cold task looked like a hang rather than a slow start.
#
#   all-MiniLM-L6-v2                     jailbreak scope   (~90MB)
#   paraphrase-multilingual-MiniLM-L12-v2 injection scope  (~470MB)
#
# Two encoders because the scopes want different things: roughly a third of real
# injection traffic is not English, while the jailbreak corpus is English-only
# and the monolingual model is stronger per-language there.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2'); \
    SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

COPY . .

CMD ["python", "-m", "app"]

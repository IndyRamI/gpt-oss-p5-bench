"""Build fixed-length (~15K token) prompts from your test-bed data or synthetically.

Input JSONL: one object per line, prompt text under --prompt-field (default
"prompt"). If the field holds a chat `messages` list, the message contents are
concatenated. Prompts are packed/truncated to exactly `target_len` tokens with
the gpt-oss tokenizer so every topology sees identical work.
"""

import json
import random
import uuid


class ApproxTokenizer:
    """Fallback for laptops / mock runs: ~1 token per whitespace word."""

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, toks):
        return " ".join(toks)


def load_tokenizer(name):
    if name in (None, "", "none"):
        return ApproxTokenizer()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def _record_text(rec, field):
    val = rec[field]
    if isinstance(val, list):  # chat messages
        return "\n\n".join(m.get("content", "") for m in val if isinstance(m, dict))
    return str(val)


def _fixed_len(tok, token_stream, target_len):
    out = []
    while len(out) < target_len:
        out.extend(next(token_stream))
    return tok.decode(out[:target_len])


def build_prompts(n, target_len, tokenizer, dataset=None, prompt_field="prompt",
                  seed=0, natural=False):
    rng = random.Random(seed)
    if dataset:
        with open(dataset) as f:
            texts = [_record_text(json.loads(l), prompt_field) for l in f if l.strip()]
        rng.shuffle(texts)
        token_lists = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
    else:
        # Synthetic: random words. Attention cost doesn't care about content,
        # but MoE routing does a little - prefer real data when you have it.
        words = ("alpha beta gamma delta kernel tensor replica router cache "
                 "prefill decode latency throughput batch token expert shard "
                 "matrix vector scheduler memory bandwidth compute").split()
        token_lists = [
            tokenizer.encode(" ".join(rng.choice(words) for _ in range(target_len)),
                             add_special_tokens=False)
            for _ in range(min(n, 64))
        ]

    if natural:  # use records as-is, only truncate overly long ones
        prompts = [tokenizer.decode(t[:target_len]) for t in token_lists[:n]]
    else:
        def stream():
            i = 0
            while True:
                yield token_lists[i % len(token_lists)]
                i += 1

        s = stream()
        prompts = [_fixed_len(tokenizer, s, target_len) for _ in range(n)]

    # Unique leading tag: guarantees no cross-request prefix reuse even if
    # someone forgets to disable prefix caching.
    return [f"[{uuid.uuid4().hex[:12]}] {p}" for p in prompts]

import argparse
import time

import tiktoken
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from omnitoken import Tokenizer as OmniTokenizer


def benchmark(tokenizer, text, tokenizer_name):
    start_time = time.time()
    tokenizer.encode(text)
    end_time = time.time()

    duration = end_time - start_time
    throughput = len(text) / duration / 1024 / 1024  # MB/s

    print(f"{tokenizer_name} Throughput: {throughput:.2f} MB/s")
    return throughput


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="bench/corpus.txt")
    args = parser.parse_args()

    with open(args.corpus, "r") as f:
        text = f.read()

    # 1. OmniToken
    print("--- OmniToken ---")
    omni_tokenizer = OmniTokenizer()
    omni_tokenizer.train_streaming(text.splitlines(), vocab_size=1000)
    benchmark(omni_tokenizer, text, "OmniToken")

    # 2. HuggingFace Tokenizer
    print("\n--- HuggingFace Tokenizer ---")
    hf_tokenizer = HFTokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]"], vocab_size=1000)
    hf_tokenizer.train_from_iterator(text.splitlines(), trainer=trainer)
    # The HF tokenizer needs to be wrapped for a fair comparison
    class HFWrapper:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def encode(self, text):
            return self.tokenizer.encode(text).ids

    benchmark(HFWrapper(hf_tokenizer), text, "HuggingFace")

    # 3. Tiktoken
    print("\n--- Tiktoken ---")
    tiktoken_enc = tiktoken.get_encoding("gpt2")
    benchmark(tiktoken_enc, text, "Tiktoken")


if __name__ == "__main__":
    main()

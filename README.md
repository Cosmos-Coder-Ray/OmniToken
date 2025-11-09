# OmniToken

OmniToken is a production-ready tokenization library for the modern enterprise.

## Quickstart

```python
from omnitoken import Tokenizer

# Load a tokenizer
tokenizer = Tokenizer() # This will be from_file later

# Encode text
ids = tokenizer.encode("Hello, world!")
print(ids)

# Decode ids
text = tokenizer.decode(ids)
print(text)
```

## Building from source

1.  Install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
2.  Install Python 3.7+
3.  Install maturin: `pip install maturin`
4.  Build and install: `cd python/omnitoken && maturin develop`

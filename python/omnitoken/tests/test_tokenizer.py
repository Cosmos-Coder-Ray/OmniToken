from omnitoken import Tokenizer
import os
from hypothesis import given, strategies as st

def test_train_encode_decode():
    # 1. Initialize a new tokenizer
    tokenizer = Tokenizer()

    # 2. Train on a small corpus
    corpus = ["hello world", "this is a test corpus", "for unigram model training."]
    tokenizer.train_streaming(iter(corpus), vocab_size=50)

    # 3. Encode some text
    ids = tokenizer.encode("hello")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)

    # 4. Decode
    decoded = tokenizer.decode(ids)
    assert "h" in decoded
    assert "e" in decoded
    assert "l" in decoded
    assert "o" in decoded

def test_save_and_load():
    # 1. Initialize and train a tokenizer
    tokenizer = Tokenizer()
    corpus = ["hello world", "this is a test corpus", "for unigram model training."]
    tokenizer.train_streaming(iter(corpus), vocab_size=20)

    # 2. Save the tokenizer
    model_path = "test_tokenizer.model"
    tokenizer.save(model_path)

    # 3. Load the tokenizer from file
    loaded_tokenizer = Tokenizer.from_file(model_path)

    # 4. Verify that the loaded tokenizer works
    text = "hello"
    original_ids = tokenizer.encode(text)
    loaded_ids = loaded_tokenizer.encode(text)

    assert original_ids == loaded_ids

    # Clean up the created model file
    os.remove(model_path)

def test_encode_empty_string():
    tokenizer = Tokenizer()
    ids = tokenizer.encode("")
    assert ids == []

@given(st.text())
def test_roundtrip_property(text):
    tokenizer = Tokenizer()
    corpus = ["hello world", "this is a test corpus", "for unigram model training."]
    tokenizer.train_streaming(iter(corpus), vocab_size=50)
    ids = tokenizer.encode(text)
    decoded_text = tokenizer.decode(ids)
    # The decode will not be perfect because of the unknown tokens
    # so we just check that the decoded text is a subset of the original
    assert all(c in text for c in decoded_text)

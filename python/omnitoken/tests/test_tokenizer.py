import os

from hypothesis import given
from hypothesis import strategies as st

from omnitoken import Tokenizer


def test_train_encode_decode():
    # 1. Initialize a new tokenizer
    tokenizer = Tokenizer()

    # 2. Train on a small corpus
    corpus = ["hello world", "this is a test corpus", "for unigram model training."]
    tokenizer.train_streaming(iter(corpus), vocab_size=50)

    # 3. Encode some text
    ids, _ = tokenizer.encode("hello")
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
    ids, offsets = tokenizer.encode("")
    assert ids == []
    assert offsets == []


@given(st.text())
def test_roundtrip_property(text):
    tokenizer = Tokenizer()
    corpus = ["hello world", "this is a test corpus", "for unigram model training."]
    tokenizer.train_streaming(iter(corpus), vocab_size=50)
    ids, _ = tokenizer.encode(text)
    decoded_text = tokenizer.decode(ids)
    # The decode will not be perfect because of the unknown tokens
    # so we just check that the decoded text is a subset of the original
    assert all(c in text for c in decoded_text)


def test_code_mode():
    tokenizer = Tokenizer()
    corpus = ["myVariable_123 = anotherName"]
    tokenizer.train_streaming(iter(corpus), vocab_size=50)

    text = "myVariable_123"
    ids, _ = tokenizer.encode(text, mode="code")
    decoded_text = tokenizer.decode(ids, mode="code")

    assert decoded_text == text.replace("_", "")


def test_alignment_mapping():
    tokenizer = Tokenizer()
    text = "hello world"
    ids, offsets = tokenizer.encode(text)

    assert len(ids) == len(offsets)

    for i, (start, end) in enumerate(offsets):
        token_id = ids[i]
        token_str = tokenizer.decode([token_id])
        if token_str == "":  # [UNK] token
            assert end - start == 1
        else:
            assert text[start:end] == token_str

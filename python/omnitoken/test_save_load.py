from omnitoken import Tokenizer
import os

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
original_tokens = tokenizer.encode(text)
loaded_tokens = loaded_tokenizer.encode(text)

print(f"Original tokens: {original_tokens}")
print(f"Loaded tokens:   {loaded_tokens}")

assert original_tokens == loaded_tokens

# Clean up the created model file
os.remove(model_path)

print("Tokenizer save and load test successful!")

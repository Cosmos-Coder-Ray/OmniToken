from omnitoken import Tokenizer
import collections

# This is a bit of a hack to create the merges dict with tuple keys
# In a real scenario, this would be loaded from a file
merges_list = [((0, 1), 8), ((2, 3), 9)]
merges = collections.OrderedDict(merges_list)


tokenizer = Tokenizer(
    {
        "h": 0, "e": 1, "l": 2, "o": 3, " ": 4, "w": 5, "r": 6, "d": 7,
        "he": 8, "lo": 9
    },
    merges
)

ids = tokenizer.encode("hello")
print(f"Encoded 'hello': {ids}")
assert ids == [8, 2, 9]

text = tokenizer.decode(ids)
print(f"Decoded {ids}: '{text}'")
assert text == "hello"

print("Python bindings test successful!")

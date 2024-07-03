from importlib.metadata import version
import tiktoken
import os

print(f"tiktoken version: {version('tiktoken')}")


# tokenizer = tiktoken.encoding_for_model("gpt-4o")
tokenizer = tiktoken.get_encoding("gpt2")


current_dir = os.path.dirname(os.path.abspath(__file__))
text_file_loc = os.path.join(current_dir, 'the-verdict.txt')
with open(text_file_loc, 'r') as file:
    raw_text = file.read()

enc_text = tokenizer.encode(raw_text)

print(len(enc_text))

enc_example = enc_text[50:]

context_size = 4

# print(f"x: {enc_example[: context_size]}")
# print(f"y: {enc_example[1: context_size + 1]}")
for i in range(1, context_size + 1):
    context = enc_example[: i]
    desired = enc_example[i]
    # print(f"{context} -> {desired}")
    print(f"{tokenizer.decode(context)} -> {tokenizer.decode([desired])}")

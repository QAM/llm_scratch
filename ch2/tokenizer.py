import re
import os

current_dir = os.path.dirname(os.path.abspath(__file__))


def clean_text(text: str):
    result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    return [item.strip() for item in result if item.strip()]


class SimpleTokenizer:

    def __init__(self, vocab: dict) -> None:
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = clean_text(text)
        print(f"preprocessed: {preprocessed}")
        ids = [self.str_to_int.get(s, self.str_to_int["<|unk|>"]) for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


# print(clean_text("Hello, world. Is this-- a test?"))

text_file_loc = os.path.join(current_dir, 'the-verdict.txt')


with open(text_file_loc, 'r') as file:
    raw_text = file.read()


all_words = sorted(set(clean_text(raw_text)))
all_words.extend(["<|endoftext|>", "<|unk|>"])
vocab_size = len(all_words)
vocab = {token: idx for idx, token in enumerate(all_words)}

simple_tokenizer = SimpleTokenizer(vocab)

text = """It's the last he painted, you know," Mrs. Gisburn said with pardonable pride. abc is good."""
ids = simple_tokenizer.encode(text)
print(f"ids: {ids}")

print(f"decoded: {simple_tokenizer.decode(ids)}")

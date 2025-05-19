import importlib
import tiktoken
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text=f.read()
tokenizer = tiktoken.get_encoding("gpt2")
enc_text = tokenizer.encode(raw_text)

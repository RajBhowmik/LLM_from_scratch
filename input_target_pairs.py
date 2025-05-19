import importlib
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text=f.read()
tokenizer = tiktoken.get_encoding("gpt2")
# enc_text = tokenizer.encode(raw_text)
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids=[]
        self.target_ids=[]
        
        token_ids=tokenizer.encode(txt,allowed_special={"<|endoftext|>"})

        for i in range(0,len(token_ids)- max_length,stride):
            input_chunk=token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
        def __len__(self):
            return len(self.input_ids)
        def __get__item(self,idx):
            return self.input_ids[idx],self.target_ids[idx]

 
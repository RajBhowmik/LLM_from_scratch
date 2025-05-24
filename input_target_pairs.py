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
    def __getitem__(self,idx): 
        '''
         returns that row of the input and output based on the index
        we are using a map style dataset
        '''
        return self.input_ids[idx],self.target_ids[idx]
    
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128,
                        shuffle=True, drop_last=True,num_workers=0 ):
    '''
    - batch_size determines how many parllel threads it runs
    - the DataLoader will directly look into the __getitem__() method in the GPTDatasetV1 class and collect the I/O values from the 
    return function
    '''
    dataset=GPTDatasetV1(txt,tokenizer,max_length,stride)

    #Create dataloader
    dataloader=DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader
dataloader = create_dataloader_v1(raw_text,batch_size=8,max_length=4,stride=4,shuffle=False)
data_iter = iter(dataloader)
inputs,targets = next(data_iter)
print("Inputs:\n",inputs)
print("\nTargets:\n",targets)

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self,d_in,d_out,context_length,dropout,num_heads,qkv_bias=False):
        '''
        if you use something only once in __init__ dont need self, but outside this if we use in any function use self
        '''
        super().__init__()
        assert(d_out%num_heads ==0),\
        "d_out must be divisible by zero"
        self.d_out = d_out
        self.num_heads=num_heads
        self.dropout=nn.Dropout(dropout)
        self.head_dim = d_out//num_heads
        self.W_query = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_key = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_value = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.out_proj = nn.Linear(d_out,d_out) # combine head outputs
        self.register_buffer("mask", torch.triu(torch.ones(context_length,context_length),diagonal=1))

    def forward(self,x):
        b,num_tokens,d_out = x.shape
        keys = self.W_key(x) #shape (b,num_tokens,d_out)
        queries = self.W_query(x)
        values = self. W_value(x)

        #Now we split the tensor by adding a 'num_heads' dimension. We unroll d_out --> num_heads,head_dim since head_dim = d_out//num_head
        keys = keys.view(b,num_tokens,self.num_heads,self.head_dim)
        values =values.view(b,num_tokens,self.num_heads,self.head_dim)
        queries = queries.view(b,num_tokens,self.num_heads,self.head_dim)
        #Transpose
        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        #attention scores
        attn_scores = queries@keys.transpose(2,3)

        # mask
        mask_bool =self.mask.bool()[:num_tokens,:num_tokens]

        # mask attention scores
        attn_scores.masked_fill(mask_bool,-torch.inf)

        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
        attn_weights=self.dropout(attn_weights)

        #context vector
        context_vec=(attn_weights@values).transpose(1,2) #bringing back the same dim we started with b,num_tokens,num_heads,head_dim

        #combine heads self.d_out = self.num_heads* self.head_dim

        context_vec = context_vec.contiguous().view(b,num_tokens,self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection, since GPT-4 does it, we did it'

        return context_vec
torch.manual_seed(123)

# Define the tensor with 3 rows and 6 columns
inputs = torch.tensor(
    [[0.43, 0.15, 0.89, 0.55, 0.87, 0.66],  # Row 1
     [0.57, 0.85, 0.64, 0.22, 0.58, 0.33],  # Row 2
     [0.77, 0.25, 0.10, 0.05, 0.80, 0.55]]  # Row 3
)

batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape) 

batch_size, context_length, d_in = batch.shape
d_out = 6
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
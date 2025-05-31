import torch
import torch.nn as nn

''''
REMEBER THE Wq,Wk,Wv ARE TRAINABLE MATRIX. THEY ARE TRAINED WHEN THE LLM RUNS. BELOW IS JUST A NORMAL VALUES WE SEE HOW THE INNER WORKINGS 
HAPPEN.

'''
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        
    
## STEP NUMBER 1 : HOW TO CONVERT INPUT EMBEDDINGS INTO QUERY, KEY AND VALUE VECTORS
        self.W_query = nn.Linear(d_in,d_out,bias= qkv_bias)
        self.W_key = nn.Linear(d_in,d_out,bias = qkv_bias)
        self.W_value = nn.Linear(d_in,d_out,bias = qkv_bias)

        '''
        The reason we use the nn.Linear becasue it has an optimized weight initilization scheme , contributing to stable 
          # self.W_query = nn.Parameter(torch.rand(d_in,d_out))
          # self.W_key = nn.Parameter(torch.rand(d_in,d_out))
          # self.W_value = nn.Parameter(torch.rand(d_in,d_out))
          '''

      
    def forward(self,x):
        queries= self.W_query(x)
        keys= self.W_key(x)
        values = self.W_value(x)
   
## STEP NUMBER 2: COMPUTE ATTENTION SCORES
        attn_scores = queries @ keys.T   
## STEP NUMBER 3: COMPUTE ATTENTION WEIGHTS - BASICALLY NORMALIZE IT. USE SOFTMAX

        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1) ## learn about why divide by square root ?  
# STEP NUMBER 4: CALCULATE CONTEXT VECTOR
        context_vector = attn_weights @ values
        return context_vector

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

d_in = inputs.shape[1]
d_out = 2

torch.manual_seed(789)
sa = SelfAttention_v1(d_in,d_out,)
print(sa(inputs))





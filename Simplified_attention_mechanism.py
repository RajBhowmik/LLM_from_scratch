# WITHOUT TRAINABLE WEIGHTS

import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

# UNCOMMENT AND SEE HOW FOR ONLY ONE QUERY "JOURNEY " EACH STEP WORKS. THEN WE WILL DO FOR THE ENTIRE INPUT

'''
query=inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])

#  calculating attention scores
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i,query)


#naive normalization
attn_weights_2_tmp = attn_scores_2/attn_scores_2.sum() #normalize, so that the attention weights sum upto 1
print(attn_weights_2_tmp)

# naive softmax normalization
def softmax_naive(x):
    return torch.exp(x)/torch.exp(x).sum(dim=0)
attn_weights_2_naive = softmax_naive(attn_scores_2)'


#PyTorch softmax calculating attention weights
attn_weights_2 = torch.softmax(attn_scores_2,dim=0)

# create the context vector
context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2+=attn_weights_2[i] *x_i'''

#Calculate attention scores
 
attn_scores = inputs @ inputs.T

# Calculate Attention weights .i.e. normalization
'''

By setting
dim=-1, we are instructing the softmax function to apply the normalization along the last
dimension of the attn_scores tensor. 

If attn_scores is a 2D tensor (for example, with a
shape of [rows, columns]), dim=-1 will normalize across the columns so that the values in
each row (summing over the column dimension) sum up to 1, creating a proper probability distribution
'''
attn_weights = torch.softmax(attn_scores,dim=-1) # 6X6 matrix

#Calculate context vector
all_conext_vecs = attn_weights @ inputs







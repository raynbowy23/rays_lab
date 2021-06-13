import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_ID = 0

class ScaledAttention(nn.Module):
    """
    Attributes:
        depth: the dimension of hidden layer and output
    """ 
    def __init__(self, in_feature: int, depth: int):
        """Inits scaled_attention"""
        super().__init__()
        self.depth = depth
        
        self.q_dense_layer = torch.nn.Linear(in_feature, depth, bias=False)
        self.k_dense_layer = torch.nn.Linear(in_feature, depth, bias=False)
        self.v_dense_layer = torch.nn.Linear(in_feature, depth, bias=False)
        self.output_dense_layer = torch.nn.Linear(in_feature, depth, bias=False)

    def forward(self, input: torch.tensor, memory: torch.tensor) -> torch.tensor:
        """Calls input and memory to calculate attention
        Args:
            input:  
                query tensor, input sequences
                length is the length of token in a sentence.
                depth is the dimension of embeddings of each word
                [batch_size, q_length, depth]
            memory:
                memory tensor, information sequences
                [batch_size, m_length, depth]

        Returns:
            attention_weight:
                A tensor
                weight whether which query can get info from each memory
        Raises:
        """
        q = self.q_dense_layer(input)
        k = self.k_dense_layer(memory)
        v = self.v_dense_layer(memory)
        # scale
        # For softmax func, if logit value is large, the value is saturated and its gradient becomes 0
        # then, the size of query gets smaller depending on depth
        # Calculates relation score between query and key by innner product of q and k
        q *= depth ** -0.5 # scale dot-product

        # calculate relative position by matrix inner product
        logit = torch.matmul(q, k)
        
        # mask
        # Turns attention weight to 0 for specific key
        # attention_mask : mask for <PAD>, and restriction of future info
        # logit += torch.float32(attention_mask) * input.dtype.min

        # softmax : [batch_size, q_length, k_length]
        attention_weight = torch.nn.Softmax(logit)

        # attention_output : get info from value depending on weight
        attention_output = torch.matmul(attention_weight, v)
        return self.output_dense_layer(attention_weight)


import pdb
import torch
from typing import Union, List
from torch import device
from torch.nn import Module, Linear, Softmax, Parameter
from torch import LongTensor, FloatTensor
from transformers import AutoModel, AutoTokenizer


class PlmMultiLabelEncoder(Module):
    def __init__(self, 
        label_num: int, 
        lm: Union[str, Module], lm_embd_dim: int, chunk_size: int=128, chunk_num: int=5, 
        first_attn_hidden_dim: int=512
    ):
        super().__init__()
        
        # Language model
        self._lm: Module = AutoModel.from_pretrained(lm) if isinstance(lm, str) else lm

        # Dimension info
        self._label_num: int = label_num
        self._lm_embd_dim: int = lm_embd_dim
        self._label_embd_dim: int = lm_embd_dim # Has to be same with `self._lm_embd_dim`
        self._chunk_size: int = chunk_size
        self._chunk_num: int = chunk_num

        self._label_attn_linear_weights_0: FloatTensor = Parameter(
            torch.rand(
                size=(first_attn_hidden_dim, lm_embd_dim), dtype=torch.float32
            ), requires_grad=True
        )
        self.register_parameter("label_attn_linear_weights_0", self._label_attn_linear_weights_0)

        self._label_attn_linear_weights_1: FloatTensor = Parameter(
            torch.rand(
                size=(label_num, first_attn_hidden_dim), dtype=torch.float32
            ), requires_grad=True
        )
        self.register_parameter("label_attn_linear_weights_1", self._label_attn_linear_weights_1)

        self._label_attn_linear_1_act: Softmax = Softmax(dim=2)

        # Label embedding matrix
        self._label_embeddings: FloatTensor = Parameter(
            torch.rand(
                size=(lm_embd_dim, label_num), dtype=torch.float32
            ), requires_grad=True
        )
        self.register_parameter("label_embeddings", self._label_embeddings)

    def forward(self, token_ids: LongTensor, attn_masks: LongTensor) -> FloatTensor:
        """
        Args:
            token_ids: Chunks of token IDs, the dimension should be:
                batch-size * chunk-number * chunk-size
        """
        assert(len(token_ids.shape) == 3)
        assert(token_ids.shape[1] == self._chunk_num)
        assert(token_ids.shape[2] == self._chunk_size)
        assert(token_ids.shape == attn_masks.shape)

        unnested_token_ids: LongTensor = token_ids.view(-1, self._chunk_size)
        unnested_attn_masks: LongTensor = attn_masks.view(-1, self._chunk_size)

        unnested_embeddings: FloatTensor = self._lm(
            input_ids=unnested_token_ids, attention_mask=unnested_attn_masks, 
            output_hidden_states=True
        ).hidden_states[-1][:, -1, :]

        chunk_embeddings: FloatTensor = unnested_embeddings.view(
            -1, self._chunk_num, self._lm_embd_dim
        )

        chunk_embeddings = chunk_embeddings.swapaxes(2, 1) 
  
        linear_trans_0: FloatTensor = self._label_attn_linear_weights_0.matmul(
            chunk_embeddings
        )
        linear_trans_0 = torch.tanh(linear_trans_0)

        linear_trans_1: FloatTensor = self._label_attn_linear_weights_1.matmul(
            linear_trans_0
        )
        linear_trans_1 = self._label_attn_linear_1_act(linear_trans_1)

        
        weighted_chunk_embeddings: FloatTensor = chunk_embeddings.matmul(
            # dim: (batch-size, chunk-num, label-num)
            linear_trans_1.transpose(2, 1)
        )

        # Label scores, dimension: (batch-size, label-num)
        logits: FloatTensor = torch.sum(
            # dimension: (batch-size, LM-hidden-dimension, label-num)
            torch.mul(weighted_chunk_embeddings, self._label_embeddings), dim=1
        )
        
        #logits = torch.sigmoid(logits)
        return logits

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from mamba_ssm.modules.mamba2 import Mamba2
from layers.Mamba_Family import Mamba_Layer, AM_Layer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
class BaseClass(torch.nn.Module):
    def __init__(self):
        super(BaseClass, self).__init__()
        self.cur_itr = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.best_mrr = torch.nn.Parameter(torch.tensor(0, dtype=torch.float64), requires_grad=False)
        self.best_itr = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.best_hit1 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float64), requires_grad=False)

class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0)
        return

    def forward(self, pred1, tar1):
        loss = self.loss_fn(pred1, tar1)
        return loss


class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        layer_num: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.layer_num = layer_num
        self.AM_layers = nn.ModuleList(
            [
                AM_Layer(
                    AttentionLayer(
                        FullAttention(True, factor=2, attention_dropout=0.1, output_attention=True),
                        d_model = d_model, n_heads=16),
                    Mamba2(d_model = d_model, d_state=16, d_conv=4, expand=2),
                    d_model,
                    0.1
                )
                for i in range(layer_num)
            ]
        )
    def forward(self, x,inference_params=None, **mixer_kwargs):
        for i in range(self.layer_num):
            x, attn = self.AM_layers[i](x)

        return x


class HMTE(BaseClass):

    def __init__(self, n_ent, n_rel, emb_dim, emb_dim1, max_arity, device, layer_num, ent_relnel, ary_list):
        super(HMTE, self).__init__()
        self.loss = MyLoss()
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.device = device
        self.emb_dim = emb_dim
        self.emb_dim1 = emb_dim1
        self.emb_dim2 = emb_dim // emb_dim1
        self.lmbda = 0.05
        self.max_arity = max_arity
        self.ent_embeddings = nn.Parameter(torch.Tensor(self.n_ent, self.emb_dim))
        self.propmt = nn.Parameter(torch.Tensor(1, self.emb_dim))
        if max_arity == 2:
            self.rel_embeddings = nn.Parameter(torch.Tensor(self.n_rel * 2, self.emb_dim))
        else:
            self.rel_embeddings = nn.Parameter(torch.Tensor(self.n_rel, self.emb_dim))
        self.pos_embeddings = nn.Embedding(self.max_arity, self.emb_dim)
        self.arylist = ary_list

        self.register_parameter('b', nn.Parameter(torch.zeros(n_ent)))
        self.mamba_preprocess = Mamba_Layer(Mamba2(d_model = emb_dim, d_state=16, d_conv=4, expand=2, headdim=64), emb_dim)
        self.decoder = nn.ModuleList([MixerModel(d_model=emb_dim, layer_num = layer_num) for _ in range(len(ary_list)+2)])


        # 初始化 embeddings 以及卷积层、全连接层的参数
        nn.init.xavier_uniform_(self.ent_embeddings.data)
        nn.init.xavier_uniform_(self.rel_embeddings.data)
        nn.init.xavier_uniform_(self.pos_embeddings.weight.data)
        nn.init.xavier_uniform_(self.propmt.data)




    def forward(self, rel_idx, ent_idx, miss_ent_domain):
        # 正版的
        r = self.rel_embeddings[rel_idx].unsqueeze(1)  # [128,1,400]
        ents = self.ent_embeddings[ent_idx]  # [128,1,400]
        concat_input = torch.cat((r, ents), dim=1)  # [128,2,400]
        pos = [i for i in range(ent_idx.shape[1] + 2) if i != miss_ent_domain]
        pos = torch.tensor(pos).to(self.device)
        pos = pos.unsqueeze(0).repeat(concat_input.shape[0], 1)
        concat_input = concat_input + self.pos_embeddings(pos)
        x = self.mamba_preprocess(concat_input)
        x = self.decoder[int(concat_input.shape[1]) - 2](x)[:,-1,:]
        miss_ent_domain = torch.LongTensor([miss_ent_domain]).to(self.device)
        mis_pos = self.pos_embeddings(miss_ent_domain)
        tar_emb = self.ent_embeddings + mis_pos
        scores = torch.mm(x, tar_emb.transpose(0, 1))
        scores += self.b.expand_as(scores)
        return scores, 0

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer,TransformerDecoderLayer,TransformerDecoder,LSTM
from typing import Dict
from .gnn_encoder import ReSaEEncoder, ReSaEBase
from utils.utils_gcn import get_param
from torch_scatter import scatter_mean, scatter_sum

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj=None):
        h = torch.mm(inp, self.W)  # [N, out_features]
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        # [N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -1e12 * torch.ones_like(e)
        if adj is None:
            adj = torch.ones((N,N),device=inp.device)
        attention = torch.where(adj > 0, e, zero_vec)  # [N, N]
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)  # [N, N].[N, out_features] => [N, out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime




class SIPREAD_mlp(nn.Module):
    def __init__(self, input_dim,sequence_length,device,type_num=5,sequence_type=None,hidden_dim=1):
        super(SIPREAD_mlp,self).__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.type_num = type_num
        self.device = device
        self.type_embed = nn.Embedding(self.type_num,hidden_dim) # 4,1
        self.single_type_length = (self.sequence_length-2)//2
        self.shrink = nn.Linear(hidden_dim,1)
        self.drop = nn.Dropout(0.2)
        self.act = nn.Tanh()
        self.scatter_shrink = nn.Linear(self.type_num*input_dim,2*input_dim)
        self.scatter_half_shrink = nn.Linear((self.type_num-1)*input_dim,2*input_dim) # no mask there are sub rel qual_ent, qual_rel
        self.no_quals_shrink = nn.Linear(self.type_num-2,2*input_dim) # no quals there are sub rel mask
        self.out_linear = nn.Linear(2*input_dim, input_dim)
        if not sequence_type:
            self.sequence_type = torch.tensor([0,1]+
                                              [2]*self.single_type_length+
                                              [3]*self.single_type_length,
                                              dtype=torch.long,
                                              device=self.device)

    def forward(self,sequence_hidden,attention_mask):
        # sequence_embedding (bs,seq,input_dim)
        sequence_type = self.sequence_type
        sequence_type = torch.where(attention_mask==1,sequence_type,torch.tensor(4,dtype=torch.long,device=self.device))
        type_embeddings = self.type_embed(sequence_type) # (bs,seq,hidden)
        x = sequence_hidden.permute(0,2,1) # (bs input_dim,seq)
        out = torch.bmm(x,type_embeddings) # (bs input_dim,1)
        if out.size(-1)!=1:
            out = self.drop(out)
            out = self.act(out)
            out = self.shrink(out)
        out = out.squeeze(-1)
        return out

    def forward_sum(self,sequence_hidden,attention_mask):
        # sequence_embedding (bs,seq,input_dim)
        sequence_type = self.sequence_type
        sequence_type = torch.where(attention_mask==1,sequence_type,torch.tensor(4,dtype=torch.long,device=self.device)) # bs,seq
        x = scatter_mean(sequence_hidden,sequence_type,dim=1)
        # x = self.drop(x)
        bs = x.size(0)        
        x = x.view(bs,-1)
        #print(x.size())
        if x.size(1) == self.type_num*self.input_dim:
            out = self.scatter_shrink(x)
        elif x.size(1) == (self.type_num-1)*self.input_dim:
            out = self.scatter_half_shrink(x)
        else:
            out = self.no_quals_shrink(x)
        out = self.act(out)
        out = self.out_linear(out)
       
        return out



class ReSaE_Transformer(ReSaEEncoder):
    model_name = 'ReSaE_Transformer_Statement'

    def __init__(self, kg_graph_repr: Dict[str, np.ndarray], config: dict, id2e: tuple = None):
        if id2e is not None:
            super(self.__class__, self).__init__(kg_graph_repr, config, id2e[1])
        else:
            super(self.__class__, self).__init__(kg_graph_repr, config)

        self.model_name = 'ReSaE_Transformer_Statement'
        self.config = config
        self.hid_drop2 = config['RESAEARGS']['HID_DROP2']
        self.feat_drop = config['RESAEARGS']['FEAT_DROP']
        self.num_transformer_layers = config['RESAEARGS']['T_LAYERS']
        self.num_heads = config['RESAEARGS']['T_N_HEADS']
        self.num_hidden = config['RESAEARGS']['T_HIDDEN']
        self.d_model = config['EMBEDDING_DIM']
        self.positional = config['RESAEARGS']['POSITIONAL']
        self.token_type = config['RESAEARGS']['TOKEN_TYPE']
        self.p_option = config['RESAEARGS']['POS_OPTION']
        self.pooling = config['RESAEARGS']['POOLING']  # min / avg / concat
        self.readout_dim = config['RESAEARGS']['READOUT_DIM'] #50
        self.hidden_drop = torch.nn.Dropout(self.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.feat_drop)
        self.device = config['DEVICE']

        if config['RESAEARGS']['DECODER'] == 'transformer':
            encoder_layers = TransformerEncoderLayer(self.d_model, self.num_heads, self.num_hidden,
                                                     config['RESAEARGS']['HID_DROP2'],activation='gelu')
            self.encoder = TransformerEncoder(encoder_layers, config['RESAEARGS']['T_LAYERS'])
        elif config['RESAEARGS']['DECODER'] == 'GAT':

            self.encoder = nn.ModuleList([GraphAttentionLayer(self.d_model,self.d_model,self.hid_drop2)
                                          for i in range(config['RESAEARGS']['T_LAYERS'])])
        else:
            raise NotImplementedError("for decoder, you can only go for transformer or GAT")
        # decoder_layers = TransformerDecoderLayer(self.d_model, self.num_heads, self.num_hidden, config['RESAEARGS']['HID_DROP2'])
        # self.decoder = TransformerDecoder(decoder_layers,1)
        self.position_embeddings = nn.Embedding(config['MAX_QPAIRS'] - 1, self.d_model)
        self.type_embeddings = nn.Embedding(config['MAX_QPAIRS'] - 1, self.d_model)
        self.layer_norm = torch.nn.LayerNorm(self.emb_dim)

        # if self.pooling == "concat":
        #     self.flat_sz = self.emb_dim * (config['MAX_QPAIRS'] - 1)
        #     self.fc = torch.nn.Linear(self.flat_sz, self.emb_dim)
        # else:
        #     self.fc = torch.nn.Linear(self.emb_dim, self.emb_dim)

        if config['RESAEARGS']['READOUT'] == 'mean':
            self.readout = torch.nn.Linear(self.emb_dim, self.emb_dim)
        elif config['RESAEARGS']['READOUT'] == 'typewise' :
            self.readout = SIPREAD_mlp(self.emb_dim,config['MAX_QPAIRS'] - 1,device=self.device,hidden_dim=self.readout_dim)
        else:
            raise NotImplementedError("for readout, you can only go for typewise or mean")

    def concat(self, e1_embed, rel_embed, qual_rel_embed, qual_obj_embed):
        e1_embed = e1_embed.unsqueeze(1)
        rel_embed = rel_embed.unsqueeze(1)
        qual_pair_num = qual_obj_embed.size(1)
        quals_emb = torch.cat((qual_obj_embed,qual_rel_embed), 1)
        quals_emb[:, 0::2, :], quals_emb[:, 1::2] = quals_emb[:, :qual_pair_num, :], quals_emb[:, qual_pair_num:, :]
        stack_inp = torch.cat([e1_embed, rel_embed, quals_emb], 1).transpose(1, 0)  # [2 + num_qual_pairs, bs, emb_dim]
        return stack_inp

    def making_mask(self,sub_emb, quals):
        mask = torch.zeros((sub_emb.shape[0], quals.shape[1] + 2)).bool().to(self.device)
        mask[:, 2::2] = quals[:,::2] == 0
        mask[:,3::2] = quals[:,1::2] == 0
        return mask


    def forward(self, sub, rel, quals):
        '''
        :param sub: bs
        :param rel: bs
        :param quals: bs*(sl-2) # bs*14
        :return:
        '''
        sub_emb, rel_emb, qual_obj_emb, qual_rel_emb, all_ent,all_rel, mask1 = \
            self.forward_ease(sub, rel, self.hidden_drop, self.feature_drop, quals, True, True)
        mask = self.making_mask(sub_emb,quals)

        stk_inp = self.concat(sub_emb, rel_emb, qual_rel_emb, qual_obj_emb)

        if self.config['RESAEARGS']['DECODER'] == 'transformer':
            x = self.encoder(stk_inp,src_key_padding_mask = mask)
            x = x.permute(1, 0, 2)
        else:
            out_list = []
            bs = stk_inp.size(1)
            for j in range(bs):
                x = stk_inp[:,j]
                for layer in self.encoder:
                    x = layer(x)
                out_list.append(x)
            x = torch.stack(out_list)

        if self.config['RESAEARGS']['READOUT'] == 'mean':
            x = torch.mean(x,0)
            x = self.readout(x)
        else:
            x = self.readout.forward_sum(x,mask)
        x = torch.mm(x , all_ent.transpose(1, 0))
        score = torch.sigmoid(x)
        return score






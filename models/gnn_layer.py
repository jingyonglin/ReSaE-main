import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch_scatter
from utils.utils_gcn import get_param, ccorr, rotate, softmax
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch_geometric.nn import MessagePassing


class ReSaEConvLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, num_rels, act=lambda x: x,
                 config=None):
        super(self.__class__, self).__init__(flow='target_to_source',
                                             aggr='add')
        self.p = config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.relation_weight_linear = torch.nn.Linear(out_channels, out_channels, bias=False)
        self.reset_parameters()
        self.drop_out_qr = torch.nn.Dropout(0.2)
        self.drop_out_qe = torch.nn.Dropout(0.2)

        self.num_rels = num_rels
        self.act = act
        self.device = None
        self.soft_max = torch.nn.Softmax(1)

        self.qual_weights = get_param((in_channels, in_channels))
        self.rel_weights = get_param((in_channels, in_channels))

        #self.w_loop = get_param((5 * in_channels, out_channels))  # (100,200)
        self.w_in = get_param((5 * in_channels, out_channels))  # (100,200)
        self.w_out = get_param((5 * in_channels, out_channels))  # (100,200)
        self.w_loop_usual = get_param((2 * in_channels, out_channels))  # (100,200)
        self.w_in_usual = get_param((2 * in_channels, out_channels))  # (100,200)
        self.w_out_usual = get_param((2 * in_channels, out_channels))  # (100,200)
        self.w_rel = get_param((in_channels, out_channels))  # (100,200)
        # self.w_rel_out = get_param((in_channels, out_channels))  # (100,200)

        self.concat_weight_qualifier = nn.Linear(2, 1, bias=False)
        self.concat_weight_regular = nn.Linear(2, 1, bias=False)
        self.concat_weight_loop = nn.Linear(2, 1, bias=False)
        self.concat_relation_update_in = nn.Linear(2, 1, bias=False)
        self.concat_relation_update_out = nn.Linear(2, 1, bias=False)
        self.init_linear_weight()
        self.drop_inside = nn.Dropout(0.2)
        if self.p['STATEMENT_LEN'] != 3:
            if self.p['RESAEARGS']['QUAL_AGGREGATE'] == 'sum' or self.p['RESAEARGS']['QUAL_AGGREGATE'] == 'mul':
                self.w_qr = get_param((in_channels, in_channels))  # new for quals setup
                self.w_qe = get_param((in_channels, in_channels))
            elif self.p['RESAEARGS']['QUAL_AGGREGATE'] == 'concat':
                self.w_q = get_param((2 * in_channels, in_channels))  # need 2x size due to the concat operation

        self.loop_rel = get_param((1, in_channels))  # (1,100)
        self.loop_ent = get_param((1, in_channels))  # new

        self.drop = torch.nn.Dropout(self.p['RESAEARGS']['GCN_DROP'])
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.bn_rel = torch.nn.BatchNorm1d(out_channels)
        self.layernorm_rel = torch.nn.LayerNorm(out_channels)
        self.layernorm_en = torch.nn.LayerNorm(out_channels)

        self.rel_Q = torch.nn.Linear(out_channels, out_channels, False)
        self.rel_K = torch.nn.Linear(out_channels, out_channels, False)

        if self.p['RESAEARGS']['ATTENTION']:
            assert self.p['RESAEARGS']['GCN_DIM'] == self.p[
                'EMBEDDING_DIM'], "Current attn implementation requires those tto be identical"
            assert self.p['EMBEDDING_DIM'] % self.p['RESAEARGS']['ATTENTION_HEADS'] == 0, "should be divisible"
            self.heads = self.p['RESAEARGS']['ATTENTION_HEADS']
            self.attn_dim = self.out_channels // self.heads
            self.negative_slope = self.p['RESAEARGS']['ATTENTION_SLOPE']
            self.attn_drop = self.p['RESAEARGS']['ATTENTION_DROP']
            self.att = get_param((1, self.heads, 2 * self.attn_dim))

        if self.p['RESAEARGS']['BIAS']: self.register_parameter('bias', torch.nn.Parameter(
            torch.zeros(out_channels)))

    def init_linear_weight(self):
        self.concat_weight_qualifier.weight = nn.Parameter(torch.tensor([0.8, 0.2]))
        self.concat_weight_regular.weight = nn.Parameter(torch.tensor([0.8, 0.2]))
        self.concat_weight_loop.weight = nn.Parameter(torch.tensor([0.8, 0.2]))
        self.concat_relation_update_in.weight = nn.Parameter(torch.tensor([0.8, 0.2]))
        self.concat_relation_update_out.weight = nn.Parameter(torch.tensor([0.8, 0.2]))

    def reset_parameters(self):
        self.relation_weight_linear.weight = torch.nn.init.uniform_(self.relation_weight_linear.weight, 0, 1)

    def qual_relation_attention(self, rel_embed, qualifier_rel):
        # 得到qualifiers的 weight   attention +residual
        relation_co_occ = torch.zeros(qualifier_rel.size(0), rel_embed.size(0), dtype=torch.long,
                                      device=rel_embed.device)  # 37433,1065

        relation_co_occ = relation_co_occ.scatter(1, qualifier_rel.unsqueeze(1), 1).to(dtype=torch.float)  # 37433,1065

        simi_matrix = torch.matmul(rel_embed, rel_embed.T) / np.sqrt(rel_embed.size(1))
        simi_matrix = self.soft_max(simi_matrix)
        relation_simi_matrix = simi_matrix[qualifier_rel]  # 37433,1065
        relation_co_occ = torch.mul(relation_simi_matrix, relation_co_occ)  # 37433,1065
        relation_with_weight = torch.matmul(relation_co_occ, rel_embed)  # 37433, 200
        out = self.drop_out_qr(self.layernorm_rel(rel_embed[qualifier_rel] + relation_with_weight))
        return out

    def qual_relation_attention_avg(self, rel_embed,qualifier_rel):
        # 得到qualifiers的 weight   attention +residual
        relation_co_occ = torch.zeros(qualifier_rel.size(0), rel_embed.size(0), dtype=torch.long,
                                      device=rel_embed.device)  # 37433,1065

        relation_co_occ = relation_co_occ.scatter(1, qualifier_rel.unsqueeze(1), 1).to(dtype=torch.float)  # 37433,1065

        simi_matrix = torch.matmul(rel_embed, rel_embed.T) / np.sqrt(rel_embed.size(1)) # 1065,1065
        simi_matrix = self.soft_max(simi_matrix)
        relation_simi_matrix = simi_matrix[qualifier_rel]  # 37433,1065
        relation_co_occ = torch.mul(relation_simi_matrix, relation_co_occ)  # 37433,1065
        relation_with_weight = torch.matmul(relation_co_occ, rel_embed)  # 37433, 200

        return relation_with_weight

    def relation_cooc(self, rel_embed, qualifier_rel, edge_type, qual_index):

        relation_co_occ = torch.zeros(qualifier_rel.size(0), rel_embed.size(0), dtype=torch.long,
                                      device=rel_embed.device)  # 450,300

        relation_co_occ = relation_co_occ.scatter(1, qualifier_rel.unsqueeze(1), 1).to(dtype=torch.float)  # 450,300

        cooc_num = scatter_add(relation_co_occ, qual_index, dim=0, dim_size=edge_type.size(0))  # 1000,300
        cooc_num_rel = scatter_add(cooc_num, edge_type, dim=0, dim_size=rel_embed.size(0))  # 300,030
        cooc_num_rel /= torch.mean(cooc_num_rel.sum(0))
        return cooc_num_rel

    def ent_cooc(self, ent_embed, qualifier_ent, ent_type, qual_index):
        ent_co_occ = torch.zeros(qualifier_ent.size(0), ent_embed.size(0), dtype=torch.long,
                                 device=ent_embed.device)  # 450,300

        ent_co_occ = ent_co_occ.scatter(1, qualifier_ent.unsqueeze(1), 1).to(dtype=torch.float)  # 450,300

        cooc_num = scatter_add(ent_co_occ, qual_index, dim=0, dim_size=ent_type.size(0))  # 1000,300
        cooc_num_ent = scatter_add(cooc_num, ent_type, dim=0, dim_size=ent_embed.size(0))  # 300,030
        cooc_num_ent /= torch.mean(cooc_num_ent.sum(0))
        return cooc_num_ent

    def update_rel_emb_with_qualifier(self, rel_embed, ent_embed, qualifier_rel, qualifier_ent, edge_type,
                                      qual_index=None):

        # qualifier_relation = self.qual_relation_attention(rel_embed,qualifier_rel) #
        # qualifier_relation = rel_embed[qualifier_rel]
        rel_part_emb = rel_embed[edge_type]  # 所有边
        qualifier_emb_rel = rel_embed[qualifier_rel]
        qualifier_emb_ent = ent_embed[qualifier_ent]

        # Step 2: pass it through qual_transform
        qualifier_emb = self.qual_transform(qualifier_ent=qualifier_emb_ent,
                                            qualifier_rel=qualifier_emb_rel)

        # qualifier_emb = torch.einsum('ij,jk -> ik',
        #                              self.coalesce_quals_rel(qualifier_emb, qual_index, rel_part_emb.shape[0]),
        #                              self.w_qr) #
        qualifier_emb = scatter_mean(qualifier_emb, qual_index, dim=0, dim_size=rel_part_emb.shape[0])
        qualifier_emb = self.drop_out_qr(qualifier_emb)
        # qualifier_emb = torch.tanh(qualifier_emb)

        add_index = torch.zeros(edge_type.size(0), device=qualifier_emb.device).index_fill(0, qual_index, 1)
        rel_qual_part = torch.stack([torch.mul(add_index.unsqueeze(1), rel_part_emb), qualifier_emb])
        out = torch.mul((1 - add_index).unsqueeze(1), rel_part_emb) + rel_qual_part

        return out

    def update_ent_emb_with_qualifier(self, ent_embs, ent_embed, qualifier_ent, edge_type_len, alpha=0.1,
                                      qual_index=None):
        # 先做加法
        qualifier_emb_ent = ent_embed[qualifier_ent]

        qualifier_emb_ent = torch.einsum('ij,jk -> ik',
                                         self.coalesce_quals_ent(qualifier_emb_ent, qual_index, edge_type_len),
                                         self.w_qe)

        qualifier_emb_ent = self.drop_out_qe(qualifier_emb_ent)
        # qualifier_emb_ent = torch.tanh(qualifier_emb_ent)
        add_index = torch.zeros(edge_type_len, device=qualifier_ent.device).index_fill(0, qual_index, 1)
        rel_qual_part = alpha * torch.mul(add_index.unsqueeze(1), ent_embs) + (1 - alpha) * qualifier_emb_ent
        out = torch.mul((1 - add_index).unsqueeze(1), ent_embs) + rel_qual_part

        return out

    def forward(self, x, edge_index, edge_type, rel_embed, qualifier_ent=None, qualifier_rel=None, quals=None):

        if self.device is None:
            self.device = edge_index.device

        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
        num_edges = edge_index.size(1) // 2
        num_ent = x.size(0)

        self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
        self.in_type, self.out_type = edge_type[:num_edges], edge_type[num_edges:]

        if self.p['STATEMENT_LEN'] != 3:
            num_quals = quals.size(1) // 2
            self.in_index_qual_ent, self.out_index_qual_ent = quals[1, :num_quals], quals[1, num_quals:]
            self.in_index_qual_rel, self.out_index_qual_rel = quals[0, :num_quals], quals[0, num_quals:]
            self.quals_index_in, self.quals_index_out = quals[2, :num_quals], quals[2, num_quals:]

            self.rel_co_occ_in = self.relation_cooc(rel_embed, self.in_index_qual_rel, edge_type, self.quals_index_in)
            self.rel_co_occ_out = self.relation_cooc(rel_embed, self.out_index_qual_rel, edge_type,
                                                     self.quals_index_out)
        '''
            Adding self loop by creating a COO matrix. Thus \
             loop index [1,2,3,4,5]
                        [1,2,3,4,5]
             loop type [10,10,10,10,10] --> assuming there are 9 relations


        '''
        # Self edges between all the nodes
        self.loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
        self.loop_type = torch.full((num_ent,), rel_embed.size(0) - 1,
                                    dtype=torch.long).to(self.device)  # if rel meb is 500, the index of the self emb is
        # 499 .. which is just added here

        self.in_norm = self.compute_norm(self.in_index, num_ent)
        self.out_norm = self.compute_norm(self.out_index, num_ent)

        if self.p['STATEMENT_LEN'] != 3:

            in_res = self.propagate(self.in_index, x=x, edge_type=self.in_type,
                                    rel_embed=rel_embed, edge_norm=self.in_norm, mode='in',
                                    ent_embed=x, qualifier_ent=self.in_index_qual_ent,
                                    qualifier_rel=self.in_index_qual_rel,
                                    qual_index=self.quals_index_in,
                                    source_index=self.in_index[0])

            loop_res = self.propagate(self.loop_index, x=x, edge_type=self.loop_type,
                                      rel_embed=rel_embed, edge_norm=None, mode='loop',
                                      ent_embed=None, qualifier_ent=None, qualifier_rel=None,
                                      qual_index=None, source_index=None)

            out_res = self.propagate(self.out_index, x=x, edge_type=self.out_type,
                                     rel_embed=rel_embed, edge_norm=self.out_norm, mode='out',
                                     ent_embed=x, qualifier_ent=self.out_index_qual_ent,
                                     qualifier_rel=self.out_index_qual_rel,
                                     qual_index=self.quals_index_out,
                                     source_index=self.out_index[0])

            rel_embed_in = self.concat_relation_update_in(torch.stack(
                [rel_embed, torch.matmul(torch.matmul(self.rel_co_occ_in, rel_embed), self.w_rel)]).permute(1, 2,
                                                                                                               0)).squeeze(
                -1)
            rel_embed_out = self.concat_relation_update_out(torch.stack(
                [rel_embed, torch.matmul(torch.matmul(self.rel_co_occ_out, rel_embed), self.w_rel)]).permute(1, 2,
                                                                                                                0)).squeeze(
                -1)
            rel_loop = rel_embed
            rel_embed = self.drop(rel_embed_in) * (1 / 3) + self.drop(rel_embed_out) * (1 / 3) + rel_loop * (1 / 3)

        else:
            in_res = self.propagate(self.in_index, x=x, edge_type=self.in_type,
                                    rel_embed=rel_embed, edge_norm=self.in_norm, mode='in',
                                    ent_embed=None, qualifier_ent=None, qualifier_rel=None,
                                    qual_index=None, source_index=None)

            loop_res = self.propagate(self.loop_index, x=x, edge_type=self.loop_type,
                                      rel_embed=rel_embed, edge_norm=None, mode='loop',
                                      ent_embed=None, qualifier_ent=None, qualifier_rel=None,
                                      qual_index=None, source_index=None)

            out_res = self.propagate(self.out_index, x=x, edge_type=self.out_type,
                                     rel_embed=rel_embed, edge_norm=self.out_norm, mode='out',
                                     ent_embed=None, qualifier_ent=None, qualifier_rel=None,
                                     qual_index=None, source_index=None)

        out = self.drop(in_res) * (1 / 3) + self.drop(out_res) * (1 / 3) + loop_res * (1 / 3)

        if self.p['RESAEARGS']['BIAS']:
            out = out + self.bias
        out = self.bn(out)
        rel_embed = self.bn_rel(rel_embed)
        # Ignoring the self loop inserted, return.
        return self.act(out), rel_embed[:-1]  # sub, rel

    def rel_transform(self, ent_embed, rel_embed):
        if self.p['RESAEARGS']['OPN'] == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.p['RESAEARGS']['OPN'] == 'sub':
            trans_embed = ent_embed - rel_embed
        elif self.p['RESAEARGS']['OPN'] == 'mult':
            trans_embed = ent_embed * rel_embed
        elif self.p['RESAEARGS']['OPN'] == 'rotate':
            trans_embed = rotate(ent_embed, rel_embed)
        else:
            raise NotImplementedError

        return trans_embed

    def qual_transform(self, qualifier_ent, qualifier_rel):
        """

        :return:
        """
        if self.p['RESAEARGS']['QUAL_OPN'] == 'corr':
            trans_embed = ccorr(qualifier_ent, qualifier_rel)
        elif self.p['RESAEARGS']['QUAL_OPN'] == 'sub':
            trans_embed = qualifier_ent - qualifier_rel
        elif self.p['RESAEARGS']['QUAL_OPN'] == 'mult':
            trans_embed = qualifier_ent * qualifier_rel
        elif self.p['RESAEARGS']['QUAL_OPN'] == 'rotate':
            trans_embed = rotate(qualifier_ent, qualifier_rel)
        else:
            raise NotImplementedError

        return trans_embed

    def qualifier_aggregate(self, qualifier_emb, rel_part_emb, alpha=0.5, qual_index=None):

        if self.p['RESAEARGS']['QUAL_AGGREGATE'] == 'sum':
            qualifier_emb = torch.einsum('ij,jk -> ik',
                                         self.coalesce_quals_rel(qualifier_emb, qual_index, rel_part_emb.shape[0]),
                                         self.w_qr)
            return alpha * rel_part_emb + (1 - alpha) * qualifier_emb  # [N_EDGES / 2 x EMB_DIM]
        elif self.p['RESAEARGS']['QUAL_AGGREGATE'] == 'concat':
            qualifier_emb = self.coalesce_quals_rel(qualifier_emb, qual_index, rel_part_emb.shape[0])
            agg_rel = torch.cat((rel_part_emb, qualifier_emb), dim=1)  # [N_EDGES / 2 x 2 * EMB_DIM]
            return torch.mm(agg_rel, self.w_qr)  # [N_EDGES / 2 x EMB_DIM]

        elif self.p['RESAEARGS']['QUAL_AGGREGATE'] == 'mul':
            qualifier_emb = torch.mm(self.coalesce_quals_rel(qualifier_emb, qual_index, rel_part_emb.shape[0], fill=1),
                                     self.w_qr)
            return rel_part_emb * qualifier_emb
        else:
            raise NotImplementedError

    def update_rel_emb_with_qualifier_att(self,rel_embed,ent_embed,qualifier_rel, qualifier_ent,edge_type, qual_index=None):

        # qualifier_relation = self.qual_relation_attention(rel_embed,qualifier_rel) # 边的加权
        # qualifier_relation = rel_embed[qualifier_rel]
        rel_part_emb = rel_embed[edge_type] # 所有边
        qualifier_emb_rel = rel_embed[qualifier_rel]
        qualifier_emb_ent = ent_embed[qualifier_ent]



        # Step 2: pass it through qual_transform
        qualifier_emb = self.qual_transform(qualifier_ent=qualifier_emb_ent,
                                            qualifier_rel=qualifier_emb_rel)
        qualifier_emb = torch.einsum('ij,jk -> ik',
                                     self.coalesce_quals_rel(qualifier_emb, qual_index, rel_part_emb.shape[0]),# 做加权和
                                     self.w_qr) #
        qualifier_emb = self.drop_out_qr(qualifier_emb)
        qualifier_emb = torch.tanh(qualifier_emb)
        add_index = torch.zeros(edge_type.size(0),device=qualifier_emb.device).index_fill(0,qual_index,1)
        rel_qual_part = 0.8 *torch.mul(add_index.unsqueeze(1),rel_part_emb) + (1 - 0.8) * qualifier_emb
        out = torch.mul((1-add_index).unsqueeze(1),rel_part_emb)+ rel_qual_part

        return out


    def update_ent_with_qual(self,x_j,rel_embed,edge_type,ent_embed,qual_index,qualifier_rel,qualifier_ent,mode):
        weight = getattr(self, 'w_{}'.format(mode))
        rel_part_emb = rel_embed[edge_type]  #
        qualifier_emb_rel = rel_embed[qualifier_rel]

        qualifier_emb_ent = ent_embed[qualifier_ent]

        qualifier_emb_rel = scatter_mean(qualifier_emb_rel, qual_index, dim=0, dim_size=rel_part_emb.shape[0])
        qualifier_emb_rel = self.drop_out_qr(qualifier_emb_rel)

        qualifier_emb_ent = scatter_mean(qualifier_emb_ent, qual_index, dim=0, dim_size=rel_part_emb.shape[0])
        qualifier_emb_ent = self.drop_out_qr(qualifier_emb_ent)
        qualifier_rel_weight = self.qual_relation_attention_avg(rel_embed, qualifier_rel)  # 边的加权
        qualifier_emb_rel_weight = scatter_mean(qualifier_rel_weight, qual_index, dim=0, dim_size=rel_part_emb.shape[0])
        qualifier_emb_rel_weight = self.drop_out_qr(qualifier_emb_rel_weight)

        out = self.concat_weight_qualifier(
            torch.stack(
                [x_j,
                 torch.matmul(
                     torch.cat([x_j, rel_part_emb, qualifier_emb_rel, qualifier_emb_ent, qualifier_emb_rel_weight], 1),
                     weight)]
            ).permute(1, 2, 0)
        ).squeeze(-1)
        return out


    def message(self, x_j, x_i, edge_type, rel_embed, edge_norm, mode, ent_embed=None, qualifier_ent=None,
                qualifier_rel=None, qual_index=None, source_index=None):

        if self.p['STATEMENT_LEN'] != 3:
            if mode != 'loop':
                out = self.update_ent_with_qual(x_j,rel_embed,edge_type,ent_embed,qual_index,qualifier_rel,qualifier_ent,mode)
            else:
                weight = getattr(self, 'w_{}_usual'.format(mode))
                rel_emb = torch.index_select(rel_embed, 0, edge_type)
                out = self.concat_weight_loop(
                    torch.stack([x_j, torch.matmul(torch.cat([x_j, rel_emb], 1), weight)]).permute(1, 2, 0)).squeeze(-1)
        else:
            weight = getattr(self, 'w_{}_usual'.format(mode))
            rel_emb = torch.index_select(rel_embed, 0, edge_type)

            out = self.concat_weight_regular(
                torch.stack([
                    x_j,
                    torch.matmul(torch.cat([x_j, rel_emb], 1), weight)]
                ).permute(1, 2, 0)).squeeze(-1)

        if self.p['RESAEARGS']['ATTENTION'] and mode != 'loop':
            out = out.view(-1, self.heads, self.attn_dim)
            x_i = x_i.view(-1, self.heads, self.attn_dim)

            alpha = torch.einsum('bij,kij -> bi', torch.cat([x_i, out], dim=-1), self.att)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, source_index, ent_embed.size(0))
            alpha = F.dropout(alpha, p=self.attn_drop)
            return out * alpha.view(-1, self.heads, 1)
        else:
            return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, mode):
        if self.p['RESAEARGS']['ATTENTION'] and mode != 'loop':
            aggr_out = aggr_out.view(-1, self.heads * self.attn_dim)

        return aggr_out

    @staticmethod
    def compute_norm(edge_index, num_ent):

        row, col = edge_index
        edge_weight = torch.ones_like(
            row).float()  # Identity matrix where we know all entities are there
        deg = scatter_add(edge_weight, row, dim=0,
                          dim_size=num_ent)  # Summing number of weights of
        # the edges, D = A + I
        deg_inv = deg.pow(-0.5)  # D^{-0.5}
        deg_inv[deg_inv == float('inf')] = 0  # for numerical stability
        norm = deg_inv[row] * edge_weight * deg_inv[col]  # Norm parameter D^{-0.5} *

        return norm

    def coalesce_quals_rel(self, qual_embeddings, qual_index, num_edges, fill=0):

        if self.p['RESAEARGS']['QUAL_N'] == 'sum':
            output = scatter_add(qual_embeddings, qual_index, dim=0, dim_size=num_edges)
            output = self.layernorm_rel(output)
            output = self.drop_out_qr(output)
        elif self.p['RESAEARGS']['QUAL_N'] == 'mean':
            output = scatter_mean(qual_embeddings, qual_index, dim=0, dim_size=num_edges)

        if fill != 0:
            # by default scatter_ functions assign zeros to the output, so we assign them 1's for correct mult
            mask = output.sum(dim=-1) == 0
            output[mask] = fill

        return output

    def coalesce_quals_ent(self, qual_embeddings, qual_index, num_edges, fill=0):

        if self.p['RESAEARGS']['QUAL_N'] == 'sum':
            output = scatter_add(qual_embeddings, qual_index, dim=0, dim_size=num_edges)
            output = self.layernorm_en(output)
            output = self.drop_out_qe(output)
        elif self.p['RESAEARGS']['QUAL_N'] == 'mean':
            output = scatter_mean(qual_embeddings, qual_index, dim=0, dim_size=num_edges)

        if fill != 0:
            # by default scatter_ functions assign zeros to the output, so we assign them 1's for correct mult
            mask = output.sum(dim=-1) == 0
            output[mask] = fill

        return output

    def __repr__(self):
        return '{}({}, {}, num_rels={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_rels)

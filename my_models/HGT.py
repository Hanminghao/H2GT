import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
from dgl.nn import edge_softmax


"""
----------------------------------------------------
Heterogeneous Graph Transformer (HGT)
----------------------------------------------------
reference: https://github.com/dmlc/dgl/tree/master/examples/pytorch/hgt
"""


class HGTLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 node_dict,
                 edge_dict,
                 n_heads,
                 dropout=0.2,
                 use_norm=False):
        super(HGTLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel = self.num_types * self.num_relations * self.num_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.skip = nn.Parameter(torch.ones(self.num_types))
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            # Filter edge dicts not exists in the graph
            edge_dict = {k: v for k, v in edge_dict.items() if k in G.canonical_etypes}
            new_feat_dict = {k: [] for k in h.keys()}

            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]

                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]

                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)

                e_id = self.edge_dict[(srctype, etype, dsttype)]

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)   #计算k和relation_att的矩阵乘法，为每种类型的边维护一个矩阵
                v = torch.einsum("bij,ijk->bik", v, relation_msg)   #

                sub_graph.srcdata['k'] = k
                sub_graph.dstdata['q'] = q
                sub_graph.srcdata['v_%d' % e_id] = v

                sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')

                sub_graph.edata['t'] = attn_score.unsqueeze(-1) #[num_edges, num_heads, 1]得到每条边在某个attention头上的注意力系数

            # multi_update_all的输入是一个dict，dict的键为边的名称，值对应为updata_all函数的message_func和reduce_func
            # fn.u_mul_e两输入维度不一定相等，比如srcnode只有100个，但edges可能有700个，但在计算时只会计算某节点入度的边，更新后的feature放在edata中
            G.multi_update_all({etype: (fn.u_mul_e('v_%d' % e_id, 't', 'm'), fn.sum('m', 't')) for etype, e_id in edge_dict.items()}, cross_reducer='mean')
                                

            new_h = {}
            for ntype in G.ntypes:
                '''
                    Step 3: Target-specific Aggregation
                    x = norm( W[node_type] * gelu( Agg(x) ) + x )
                '''
                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                try:
                    t = G.nodes[ntype].data['t'].view(-1, self.out_dim)
                except KeyError:
                    new_h[ntype] = h[ntype]
                    continue
                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out * alpha + h[ntype] * (1-alpha)
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h


class HGT(nn.Module):
    def __init__(self, node_dict, edge_dict, in_dim, hidden_dim, n_layers, n_heads,
                 use_norm=True, graph_pooling_type="mean"):
        super(HGT, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.gcs = nn.ModuleList()
        self.n_layers = n_layers
        self.adapt_ws = nn.ModuleList()

        self.pools = nn.ModuleList()
        self.linears_prediction = nn.ModuleDict({
            k: nn.ModuleList()
            for k, _ in node_dict.items()
        })

        for t in range(len(node_dict)):
            self.adapt_ws.append(nn.Linear(in_dim, hidden_dim))
        for _ in range(n_layers):
            self.gcs.append(HGTLayer(hidden_dim, hidden_dim, node_dict, edge_dict, n_heads, use_norm=use_norm))


    def forward(self, G, h=None):

        # Read features
        if h is None:
            h = {}
            for ntype in G.ntypes:
                n_id = self.node_dict[ntype]
                h[ntype] = F.gelu(self.adapt_ws[n_id](G.nodes[ntype].data['feat']))
        else:
            for ntype in G.ntypes:
                n_id = self.node_dict[ntype]
                h[ntype] = F.gelu(self.adapt_ws[n_id](h[ntype]))


        for i in range(self.n_layers):
            h = self.gcs[i](G, h)
        for ntype in G.ntypes:
            G.nodes[ntype].data['feat'] = h[ntype]
        return G

import torch
import torch.nn as nn
import torch.nn.functional as F
from my_models import HGT, HeteroRGCN
import yaml
import numpy as np
import pickle
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from collections import OrderedDict


def ordered_yaml():
    """
    yaml orderedDict support
    """
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

class HetHyper(nn.Module):
    def __init__(self, num_heads, embedding_dim, drop=0.25, specific=True) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.depth = embedding_dim // num_heads
        self.specific = specific

        self.Wq = nn.Linear(embedding_dim, embedding_dim)
        # KNN超边的K矩阵
        self.Wk_nebor = nn.Linear(embedding_dim, embedding_dim)
        # self.Wv_nebor = nn.Linear(embedding_dim, embedding_dim)

        # 拓扑结构的K矩阵
        if self.specific:
            self.Wk_topo = nn.Linear(embedding_dim, embedding_dim)
            # self.Wv_topo = nn.Linear(embedding_dim, embedding_dim)

            # 肿瘤侵袭的K矩阵
            self.Wk_stroma = nn.Linear(embedding_dim, embedding_dim)
            # self.Wv_stroma = nn.Linear(embedding_dim, embedding_dim)

        self.drop = nn.Dropout(drop)

        self.fc = nn.Linear(embedding_dim, embedding_dim)

        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(embedding_dim)

    def update_hyperedge(self, X, H):
        # X: [N, D]
        # H: [N, E]
        # return: [E, D]
        hyperedge_feat = torch.matmul(H.t(), X)
        norm_factor = torch.sum(H, dim=0).unsqueeze(1)
        return hyperedge_feat / norm_factor
    
    def updata_hypernode(self, X, E, H):
        X = self.norm(X)
        E = self.norm(E)

        q = self.Wq(X).view(-1, self.num_heads, self.depth).unsqueeze(dim=0).permute(0,2,1,3) # [B, num_heads, N, depth]   N is the num of nodes
        if self.specific:
            k_nebor = self.Wk_nebor(E[:-4]).view(-1, self.num_heads, self.depth).unsqueeze(dim=0).permute(0,2,1,3)    # [B, num_heads, E1, depth]    E1 is the num of KNN hyperedges
            k_topo = self.Wk_topo(E[-4:-1]).view(-1, self.num_heads, self.depth).unsqueeze(dim=0).permute(0,2,1,3)    # [B, num_heads, E2, depth]    E2 is the num of topology hyperedges
            k_stroma = self.Wk_stroma(E[-1]).view(-1, self.num_heads, self.depth).unsqueeze(dim=0).permute(0,2,1,3)   # [B, num_heads, E3, depth]    E3 is the num of stroma hyperedges
        else:
            k_nebor = self.Wk_nebor(E[:-4]).view(-1, self.num_heads, self.depth).unsqueeze(dim=0).permute(0,2,1,3)    # [B, num_heads, E1, depth]    E1 is the num of KNN hyperedges
            k_topo = self.Wk_nebor(E[-4:-1]).view(-1, self.num_heads, self.depth).unsqueeze(dim=0).permute(0,2,1,3)    # [B, num_heads, E2, depth]    E2 is the num of topology hyperedges
            k_stroma = self.Wk_nebor(E[-1]).view(-1, self.num_heads, self.depth).unsqueeze(dim=0).permute(0,2,1,3)   # [B, num_heads, E3, depth]    E3 is the num of stroma hyperedges
        k = torch.cat([k_nebor, k_topo, k_stroma], dim=2) # [B, num_heads, E, depth]
        
        att_nebor = torch.matmul(q, k_nebor.permute(0, 1, 3, 2)) / (self.depth ** 0.5)
        att_topo = torch.matmul(q, k_topo.permute(0, 1, 3, 2)) / (self.depth ** 0.5)
        att_stroma = torch.matmul(q, k_stroma.permute(0, 1, 3, 2)) / (self.depth ** 0.5)

        attention = torch.cat([att_nebor, att_topo, att_stroma], dim=-1)
        attention[:, :, H==0] = float('-inf')
        attention = self.drop(torch.softmax(attention, dim=-1))
        # attention[:, :, :, -1] = attention[:, :, :, -1] * 2

        feat_out = torch.matmul(attention, k).squeeze(dim=0) # [num_heads, N, depth]
        feat_out = feat_out.permute(1, 0, 2).reshape(-1, self.embedding_dim) # [N, num_heads*depth]

        return self.relu(self.fc(feat_out))

    
    def forward(self, X, H, alpha=0.5):
        hyperedge_feat = self.update_hyperedge(X, H)
        hypernode_feat = self.updata_hypernode(X, hyperedge_feat, H)
        hypernode_feat   = hypernode_feat * alpha + X * (1-alpha)
        return hypernode_feat, H


class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class H2GT(nn.Module):
    def __init__(self, config, visualize=False) -> None:
        super(H2GT, self).__init__()
        self.hyperprocess = nn.ModuleList()
        config_hgt = config["HGT"]
        self.specific = config['Hyper']['specific']
        if not self.specific:
            print('without type-specific projection')
        self.hid_dim = config_hgt["hidden_dim"]
        n_node_types = config_hgt["n_node_types"]
        etypes = config_hgt["edge_types"]
        self.visualize = visualize
        canonical_etypes = [
            (str(s), r, str(t))
            for r in etypes
            for s in range(n_node_types)
            for t in range(n_node_types)
        ]

        node_dict = {str(i): i for i in range(n_node_types)}
        if config_hgt['name'] == "HGT":
            edge_dict = {et: i for i, et in enumerate(canonical_etypes)}
            self.hgt = HGT(
                    node_dict,
                    edge_dict,
                    in_dim=config_hgt["in_dim"],
                    hidden_dim=config_hgt["hidden_dim"],
                    n_layers=config_hgt["num_layers"],
                    n_heads=config_hgt["num_heads"]
            )
        elif config_hgt['name'] == "RGCN":
            canonical_etypes_RGCN = {et: str(i) for i, et in enumerate(canonical_etypes)}
            self.hgt = HeteroRGCN(
                    in_dim=config_hgt["in_dim"],
                    hidden_dim=config_hgt["hidden_dim"],
                    n_layers=config_hgt["num_layers"],
                    etypes=canonical_etypes_RGCN,
                    node_dict=node_dict,
            )
        
        config_hyper = config['Hyper']
        n_layers = config_hyper['n_layers']
        self.n_layers = n_layers
        num_heads = config_hyper['num_heads']
        # process hypergraph
        for _ in range(n_layers):
            self.hyperprocess.append(HetHyper(num_heads=num_heads, embedding_dim=self.hid_dim, specific=self.specific))

        # gated_attention
        self.path_attention_head = Attn_Net_Gated(L=self.hid_dim, D=self.hid_dim, n_classes=1)

        out_dim = config['out_dim']
        self.out_linear = nn.Linear(self.hid_dim, out_dim)


    def get_hyper_feature(self, total_node_num, G_node, info_dict, G):
        feat_all = torch.zeros(total_node_num, self.hid_dim).to(self.device)
        for num, (_, value) in enumerate(info_dict.items()):
            feat = []
            for idx, type in zip(value['index'], value['type']):
                idx = int(idx)
                type = str(int(type))
                index = torch.where(G_node[type] == idx)[0]
                if G.ndata['feat'][type][index].shape[0] != 0:
                    feat.append(G.ndata['feat'][type][index])
            # feat = torch.cat(feat, dim=1)
            if feat == []:
                continue
            feat = torch.stack(feat, dim=0)
            feat = torch.mean(feat, dim=0)
            feat_all[num] = feat
        return feat_all

    
    def forward(self, g, case_name):
        info_dict = g['info_dict']
        HIM = g['HIM'][0][list(info_dict.keys()), :].to('cuda')
    
        g = g['het_graph']
        self.device = g.device
        # hgt
        G =  self.hgt(g)

        # chaotu
        G_node_ID = {}
        for type in G.ntypes:
            G_node_ID[type] = G.ndata['_ID'][type]
        HIM[:, :-4] = HIM[:, :-4] + torch.diag(torch.ones(HIM.shape[0])).cuda()
        feat_all = self.get_hyper_feature(HIM.shape[0], G_node_ID, info_dict, G)
        for i in range(self.n_layers):
            feat_all, HIM = self.hyperprocess[i](feat_all, HIM)

        # 使用gated_attention
        A, feat_all = self.path_attention_head(feat_all)

        A = torch.transpose(A, 1, 0)
        feat_all = torch.mm(F.softmax(A, dim=1), feat_all)
        # 求平均来合并
        # feat_all, _ = torch.max(feat_all, dim=0, keepdim=True)
        # # 使用可学习参数选择
        # # feat_all = torch.matmul(feat_all.T, self.select_linear).T
        
        out = self.out_linear(feat_all)
        visualize = []
        if self.visualize:
            keys = info_dict.keys()
            for idx, key in enumerate(keys):
                pos = info_dict[key]['pos']
                pos = (pos[0].item(), pos[1].item())
                score = F.softmax(A, dim=1)[:,idx].cpu().numpy()[0]
                if 1 in info_dict[key]['type'] and 3 in info_dict[key]['type']:
                    type = 'mix'
                else:
                    type = 'others'
                visualize.append([pos, score, type])
            return out, visualize
        return out
    
if __name__ == "__main__":
    graph_path = '/Dataset3/my_secpro_data/het_garph/BRCA/20X/TCGA-A2-A3KC-01Z-00-DX1.2532878B-49E2-48D5-82D5-00730C90EEF8.pkl'
    opt_path = './configs/BRCA/my_survival_hyper_HGNN.yml'
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    with open(opt_path, mode='r') as f:
        loader, _ = ordered_yaml()
        config = yaml.load(f, loader)
        print(f"Loaded configs from {opt_path}")
    config_gnn = config['GNN']
    model = H2GT(config=config_gnn)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数数量: {total_params}")


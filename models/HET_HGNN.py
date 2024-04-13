import torch
import torch.nn as nn
import torch.nn.functional as F
from my_models import HGT, HeteroRGCN
import math
from torch.nn.parameter import Parameter
from models.hyper_utils import *
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing, GCNConv, GATConv


def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    device = H.device
    H = torch.tensor(H, dtype=torch.float32)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = torch.ones(n_edge, dtype=torch.float32, device=device)
    # the degree of the node
    DV = torch.sum(H * W, dim=1)
    # the degree of the hyperedge
    DE = torch.sum(H, dim=0)

    invDE = torch.diag(torch.pow(DE, -1))
    DV2 = torch.diag(torch.pow(DV, -0.5))
    W = torch.diag(W)

    if variable_weight:
        DV2_H = torch.mm(DV2, H)
        invDE_HT_DV2 = torch.mm(torch.mm(invDE, H.t()), DV2)
        return DV2_H, W, invDE_HT_DV2
    else:
        G = torch.mm(torch.mm(torch.mm(torch.mm(DV2, H), W), invDE), torch.mm(H.t(), DV2))
        return G

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True, dropout=0.5):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.dropout = dropout

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        if self.dropout:
            x = F.dropout(x, self.dropout)
        return x
class SetGNN(nn.Module):
    def __init__(self,hid_dim=128,out_dim=128, norm=None):
        super(SetGNN, self).__init__()
        """
        args should contain the following:
        V_in_dim, V_enc_hid_dim, V_dec_hid_dim, V_out_dim, V_enc_num_layers, V_dec_num_layers
        E_in_dim, E_enc_hid_dim, E_dec_hid_dim, E_out_dim, E_enc_num_layers, E_dec_num_layers
        All_num_layers,dropout
        !!! V_in_dim should be the dimension of node features
        !!! E_out_dim should be the number of classes (for classification)
        """
#         Now set all dropout the same, but can be different
        self.All_num_layers = 2
        self.dropout = 0.5
        self.aggr = 'mean'
        self.NormLayer = 'lb'
        self.InputNorm = True
        self.GPR = False
        self.LearnMask = False
#         Now define V2EConvs[i], V2EConvs[i] for ith layers
#         Currently we assume there's no hyperedge features, which means V_out_dim = E_in_dim
#         If there's hyperedge features, concat with Vpart decoder output features [V_feat||E_feat]
        self.V2EConvs = nn.ModuleList()
        self.E2VConvs = nn.ModuleList()
        self.bnV2Es = nn.ModuleList()
        self.bnE2Vs = nn.ModuleList()

        if self.LearnMask:
            self.Importance = Parameter(torch.ones(norm.size()))

        if self.All_num_layers == 0:
            pass
        else:
            self.V2EConvs.append(HalfNLHconv(in_dim=512,
                                             hid_dim=hid_dim,
                                             out_dim=out_dim,
                                             num_layers=2,
                                             dropout=self.dropout,
                                             Normalization=self.NormLayer,
                                             InputNorm=self.InputNorm,
                                             heads=1,
                                             attention=True))
            self.bnV2Es.append(nn.BatchNorm1d(hid_dim))
            self.E2VConvs.append(HalfNLHconv(in_dim=hid_dim,
                                             hid_dim=hid_dim,
                                             out_dim=out_dim,
                                             num_layers=2,
                                             dropout=self.dropout,
                                             Normalization=self.NormLayer,
                                             InputNorm=self.InputNorm,
                                             heads=1,
                                             attention=True))
            self.bnE2Vs.append(nn.BatchNorm1d(hid_dim))
            for _ in range(self.All_num_layers-1):
                self.V2EConvs.append(HalfNLHconv(in_dim=hid_dim,
                                                 hid_dim=hid_dim,
                                                 out_dim=hid_dim,
                                                 num_layers=2,
                                                 dropout=self.dropout,
                                                 Normalization=self.NormLayer,
                                                 InputNorm=self.InputNorm,
                                                 heads=1,
                                                 attention=True))
                self.bnV2Es.append(nn.BatchNorm1d(hid_dim))
                self.E2VConvs.append(HalfNLHconv(in_dim=hid_dim,
                                                 hid_dim=hid_dim,
                                                 out_dim=hid_dim,
                                                 num_layers=2,
                                                 dropout=self.dropout,
                                                 Normalization=self.NormLayer,
                                                 InputNorm=self.InputNorm,
                                                 heads=1,
                                                 attention=True))
                self.bnE2Vs.append(nn.BatchNorm1d(hid_dim))
            if self.GPR:
                self.MLP = MLP(in_channels=512,
                               hidden_channels=hid_dim,
                               out_channels=hid_dim,
                               num_layers=2,
                               dropout=self.dropout,
                               Normalization=self.NormLayer,
                               InputNorm=False)
                self.GPRweights = nn.Linear(self.All_num_layers+1, 1, bias=False)


#         Now we simply use V_enc_hid=V_dec_hid=E_enc_hid=E_dec_hid
#         However, in general this can be arbitrary.


    def reset_parameters(self):
        for layer in self.V2EConvs:
            layer.reset_parameters()
        for layer in self.E2VConvs:
            layer.reset_parameters()
        for layer in self.bnV2Es:
            layer.reset_parameters()
        for layer in self.bnE2Vs:
            layer.reset_parameters()
        self.classifier.reset_parameters()
        if self.GPR:
            self.MLP.reset_parameters()
            self.GPRweights.reset_parameters()
        if self.LearnMask:
            nn.init.ones_(self.Importance)

    def forward(self, x, edge_index):
        """
        The data should contain the follows
        data.x: node features
        data.edge_index: edge list (of size (2,|E|)) where data.edge_index[0] contains nodes and data.edge_index[1] contains hyperedges
        !!! Note that self loop should be assigned to a new (hyper)edge id!!!
        !!! Also note that the (hyper)edge id should start at 0 (akin to node id)
        data.norm: The weight for edges in bipartite graphs, correspond to data.edge_index
        !!! Note that we output final node representation. Loss should be defined outside.
        """
#             The data should contain the follows
#             data.x: node features
#             data.V2Eedge_index:  edge list (of size (2,|E|)) where
#             data.V2Eedge_index[0] contains nodes and data.V2Eedge_index[1] contains hyperedges
        norm = torch.ones_like(edge_index[0])
        if self.LearnMask:
            norm = self.Importance*norm
        cidx = edge_index[1].min()
        edge_index[1] -= cidx  # make sure we do not waste memory
        reversed_edge_index = torch.stack(
            [edge_index[1], edge_index[0]], dim=0)
        if self.GPR:
            xs = []
            xs.append(F.relu(self.MLP(x)))
            for i, _ in enumerate(self.V2EConvs):
                x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
#                 x = self.bnV2Es[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr)
                x = F.relu(x)
                xs.append(x)
#                 x = self.bnE2Vs[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = torch.stack(xs, dim=-1)
            x = self.GPRweights(x).squeeze()
        else:
            x = F.dropout(x, p=0.2, training=self.training) # Input dropout
            for i, _ in enumerate(self.V2EConvs):
                x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
#                 x = self.bnV2Es[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(self.E2VConvs[i](
                    x, reversed_edge_index, norm, self.aggr))
#                 x = self.bnE2Vs[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x

class CEGAT(MessagePassing):
    def __init__(self,
                 in_dim=512,
                 hid_dim=128,
                 out_dim=128,
                 num_layers=2,
                 heads=1,
                 output_heads=1,
                 dropout=0.5,
                 Normalization='bn'
                 ):
        super(CEGAT, self).__init__()
        self.convs = nn.ModuleList()
        self.normalizations = nn.ModuleList()

        if Normalization == 'bn':
            self.convs.append(GATConv(in_dim, hid_dim, heads))
            self.normalizations.append(nn.BatchNorm1d(hid_dim))
            for _ in range(num_layers-2):
                self.convs.append(GATConv(heads*hid_dim, hid_dim))
                self.normalizations.append(nn.BatchNorm1d(hid_dim))

            self.convs.append(GATConv(heads*hid_dim, out_dim,
                                      heads=output_heads, concat=False))
        else:  # default no normalizations
            self.convs.append(GATConv(in_dim, hid_dim, heads))
            self.normalizations.append(nn.Identity())
            for _ in range(num_layers-2):
                self.convs.append(GATConv(hid_dim*heads, hid_dim))
                self.normalizations.append(nn.Identity())

            self.convs.append(GATConv(hid_dim*heads, out_dim,
                                      heads=output_heads, concat=False))

        self.dropout = dropout

    def reset_parameters(self):
        for layer in self.convs:
            layer.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ is 'Identity'):
                normalization.reset_parameters()

    def forward(self, x, edge_index):
        #         Assume edge_index is already V2V
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

class MLP_model(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, InputNorm=False):
        super(MLP_model, self).__init__()
        in_channels = 512
        hidden_channels = 128
        num_layers = 2
        dropout = 0.5
        Normalization = 'ln'

        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.InputNorm = InputNorm

        assert Normalization in ['bn', 'ln', 'None']
        if Normalization == 'bn':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
            else:
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.BatchNorm1d(hidden_channels))
        elif Normalization == 'ln':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
            else:
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.LayerNorm(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.LayerNorm(hidden_channels))
        else:
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                self.normalizations.append(nn.Identity())
            else:
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.Identity())
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.Identity())

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ is 'Identity'):
                normalization.reset_parameters()

    def forward(self, x):
        x = self.normalizations[0](x)
        for i, lin in enumerate(self.lins):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i+1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class HGNN_ATT(nn.Module):
    def __init__(self, input_size, n_hid, output_size, dropout=0.3):
        super(HGNN_ATT, self).__init__()
        self.dropout = dropout
        self.gat1 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2, transfer = False, concat=True)
        self.gat2 = HyperGraphAttentionLayerSparse(n_hid, output_size, dropout=self.dropout, alpha=0.2, transfer = True, concat=False)
        
    def forward(self, x, H):   
        x = self.gat1(x, H)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gat2(x, H)
        return x

class HyperGCN(nn.Module):
    def __init__(self, args): #node-feature [paper] feature config
        """
        d: initial node-feature dimension
        h: number of hidden units
        c: number of classes
        """
        super(HyperGCN, self).__init__()
        d, l, c = args['d'], args['depth'], args['c']
        cuda = torch.cuda.is_available()

        h = [d]
        for _ in range(l-1): #l = 2
            h.append(d//2)
        h.append(c)
        reapproximate = True
        self.layers = nn.ModuleList(
            [HyperGraphConvolution(h[i], h[i + 1], reapproximate, cuda) for i in range(l)])
        self.do, self.l = 0.3, args['depth']
        self.m = args['mediators']

    def forward(self, E, H):
        """
        an l-layer GCN
        """
        do, l, m = self.do, self.l, self.m

        for i, hidden in enumerate(self.layers):
            H = F.relu(hidden(E, H, m))
            if i < l - 1:
                H = F.dropout(H, do, training=self.training)

        return F.log_softmax(H, dim=1)

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

class H2GT_HGNN(nn.Module):
    def __init__(self, config, visualize=False) -> None:
        super(H2GT_HGNN, self).__init__()
        self.hyperprocess = nn.ModuleList()
        config_hgt = config["HGT"]
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
        self.hyper_name = config_hyper['name']
        self.n_layers = config_hyper['n_layers']
        # process hypergraph
        if self.hyper_name == 'HGNN':
            print('Train with HGNN')
            for _ in range(self.n_layers):
                self.hyperprocess.append(HGNN_conv(in_ft=self.hid_dim, out_ft=self.hid_dim, dropout=0.5))
            # gated_attention
            self.path_attention_head = Attn_Net_Gated(L=self.hid_dim, D=self.hid_dim, n_classes=1)
            out_dim = config['out_dim']
            self.out_linear = nn.Linear(self.hid_dim, out_dim)
        elif self.hyper_name == 'AllSetTransformer':
            print('Train with AllSetTransformer')
            self.hyperprocess.append(SetGNN(hid_dim=64, out_dim=64))
            self.path_attention_head = Attn_Net_Gated(L=64, D=64, n_classes=1)
            out_dim = config['out_dim']
            self.out_linear = nn.Linear(64, out_dim)
        elif self.hyper_name == 'CEGAT':
            print('Train with CEGAT')
            self.hyperprocess.append(CEGAT(hid_dim=128, out_dim=128))
            # gated_attention
            self.path_attention_head = Attn_Net_Gated(L=128, D=128, n_classes=1)
            out_dim = config['out_dim']
            self.out_linear = nn.Linear(128, out_dim)
        elif self.hyper_name == 'MLP':
            print('Train with MLP')
            self.hyperprocess.append(MLP_model(InputNorm=False))
            self.path_attention_head = Attn_Net_Gated(L=128, D=128, n_classes=1)
            out_dim = config['out_dim']
            self.out_linear = nn.Linear(128, out_dim)
        elif self.hyper_name == 'HyperGAT':
            print('Train with HyperGAT')
            self.hyperprocess.append(HGNN_ATT(input_size=self.hid_dim, n_hid=self.hid_dim, output_size=128))
            self.path_attention_head = Attn_Net_Gated(L=128, D=128, n_classes=1)
            out_dim = config['out_dim']
            self.out_linear = nn.Linear(128, out_dim)
        elif self.hyper_name == 'HyperGCN':
            print('Train with HyperGCN')
            GCN_dic = {'d': 512, 'depth':2, 'c':128, 'mediators': False}
            self.hyperprocess.append(HyperGCN(args=GCN_dic))
            self.path_attention_head = Attn_Net_Gated(L=128, D=128, n_classes=1)
            out_dim = config['out_dim']
            self.out_linear = nn.Linear(128, out_dim)



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
        HIM = g['HIM'][0][list(info_dict.keys()), :]
    
        g = g['het_graph']
        self.device = g.device
        # hgt
        G =  self.hgt(g)

        # HyperGraph
        G_node_ID = {}
        for type in G.ntypes:
            G_node_ID[type] = G.ndata['_ID'][type]
        
        HIM[:, :-4] = HIM[:, :-4] + torch.diag(torch.ones(HIM.shape[0]))
        feat_all = self.get_hyper_feature(HIM.shape[0], G_node_ID, info_dict, G)
        if self.hyper_name == 'HGNN':
            _G = generate_G_from_H(HIM.to(self.device))
            _G = _G.to(self.device)
            for i in range(self.n_layers):
                feat_all = self.hyperprocess[i](feat_all, _G)

        elif self.hyper_name == 'AllSetTransformer':
            nodes = []
            hyperedges = []
            nodes, hyperedges = np.where(HIM == 1)
            V2Eedge_index = torch.tensor([nodes, hyperedges], dtype=torch.long).to(self.device)
            feat_all = self.hyperprocess[0](x=feat_all, edge_index=V2Eedge_index)
        
        elif self.hyper_name == 'CEGAT':
            nodes = []
            hyperedges = []
            nodes, hyperedges = np.where(HIM == 1)
            edge_index = torch.stack((torch.tensor(nodes), torch.tensor(hyperedges)), dim=0)
            edge_index, norm = ConstructV2V(edge_index)
            edge_index, norm = gcn_norm(edge_index, norm, add_self_loops=False)
            feat_all = self.hyperprocess[0](feat_all, edge_index.to(self.device))
        elif self.hyper_name == 'MLP':
            feat_all = self.hyperprocess[0](feat_all)
        elif self.hyper_name == 'HyperGAT':
            feat_all = self.hyperprocess[0](feat_all.unsqueeze(0), HIM.T.unsqueeze(0).to(self.device))
            feat_all = feat_all.squeeze(0)
        elif self.hyper_name == 'HyperGCN':
            H_dic = {x: (HIM[:, x] != 0).nonzero(as_tuple=False).squeeze().tolist() for x in range(HIM.shape[1])}
            feat_all = self.hyperprocess[0](H_dic, feat_all)

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
import os
import pickle

import h5py
import glob
from typing import Any
import pandas as pd
import torchvision.transforms
import torch
from torch.utils.data import Dataset

import dgl
from dgl.data import DGLDataset
from dgl import transforms



class Drop_Node():
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, g):
        g = g.clone()

        # Fast path
        if self.p == 0:
            return g

        for ntype in g.ntypes:
            print('节点类型',ntype)
            length = g.num_nodes(ntype)
            print('节点数目', length)
            if length == 0:
                continue
            nids_to_remove = sorted(torch.randperm(length)[:int(0.3 * length)])
            # eids_to_remove = []
            # for node in nids_to_remove:
            #     for etype in g.etypes:
            #         eids_to_remove.extend(list(g.out_edges(node, etype=etype)))
            #         eids_to_remove.extend(list(g.in_edges(node, etype=etype)))
            g = dgl.remove_nodes(g, nids_to_remove, ntype=ntype)
            # g.remove_edges(eids_to_remove, etype=etype)
        return g

class Drop_graph():
    def __init__(self, p) -> None:
        self.p = p

    def __call__(self, g):
        transform = transforms.Compose(
    [
        transforms.DropNode(p=self.p),
        transforms.DropEdge(p=self.p),
        transforms.NodeShuffle(),   # 打乱节点顺序
        transforms.FeatMask(p=self.p, node_feat_names=['feat'])    # 随机掩盖节点特征
    ]
    )
        return transform(g)


class WSIData(Dataset):
    def __init__(self, data_root=None):
        self.data_root = data_root
        self.data_list = []
        types = ('*.svs', '*.tif')
        for type_ in types:
            self.data_list.extend(glob.glob(self.data_root + '/**/'+type_, recursive=True))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        wsi_path = self.data_list[index]
        return wsi_path


# class PatchData(Dataset):
#     def __init__(self, patch_path):
#         """
#         Args:
#             data_24: path to input data
#         """
#         self.patch_paths = [p for p in patch_path.glob("*")]
#         self.transforms = torchvision.transforms.Compose([
#             # torchvision.transforms.GaussianBlur(kernel_size=3),
#             # torchvision.transforms.RandomResizedCrop(size=256),
#             torchvision.transforms.Resize(256),
#             torchvision.transforms.ToTensor(),
#             # torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
#         # self.node_type = [p.split('.')[0].split('_')[-1] for p in os.listdir(patch_path)]
#         self.node_pos = [[int(p.split('.')[0].split('_')[0]), int(p.split('.')[0].split('_')[1])] for p in os.listdir(patch_path)]

#     def __len__(self):
#         return len(self.patch_paths)

#     def __getitem__(self, idx):

#         img = Image.open(self.patch_paths[idx]).convert('RGB')
#         img = self.transforms(img)
#         return img
    

class PatchData(Dataset):
    def __init__(self, h5_path, wsi, patch_size) -> None:
        f = h5py.File(h5_path, 'r')
        coords = f['coords'][:]
        self.wsi = wsi
        MAG = int(float(wsi.properties['aperio.AppMag']))
        if MAG == 40:
            self.patch_size = 512
        else:
            self.patch_size = 256

        self.transforms = torchvision.transforms.Compose([
            # torchvision.transforms.GaussianBlur(kernel_size=3),
            # torchvision.transforms.RandomResizedCrop(size=256),
            torchvision.transforms.Resize(patch_size),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.node_pos = coords

    def __len__(self):
        return len(self.node_pos)
    
    def __getitem__(self, idx) -> Any:
        x, y = self.node_pos[idx]
        img = self.wsi.read_region(location=(x, y), level=0, size=(self.patch_size, self.patch_size)).convert('RGB')
        img = self.transforms(img)
        return img
        




class GraphDataset(DGLDataset):
    def __init__(self, graph_path, normal_path, name_, type_, name='POINTHET'):
        """
        :param data_root: Root of graph
        :param normal_path: Path to the file contain list of normal images
        """
        self.graph_path = graph_path
        self.normal_path = normal_path
        self.name_ = name_
        self.type_ = type_

        super().__init__(name)

    def process(self):

        with open(self.graph_path) as g:
            self.graph_paths = [a.strip() for a in g.readlines()]

        if self.name_ == 'COAD' or self.name_ == 'BRCA':
            with open(self.normal_path) as f:
                # List of path to normal images
                self.normal_list = [l.strip() for l in f.readlines()]

    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, index):
        graph_path = self.graph_paths[index]

        with open(graph_path, 'rb') as f:
            dgl_graph = pickle.load(f)

        s = str(graph_path)

        if self.name_ == "COAD":
            # COAD training and testing data
            pos = s.find("TCGA")
            label = 0 if s[pos:pos+16] in self.normal_list else 1
        elif self.name_ == "BRCA":
            # BRCA training and testing data
            pos = s.find("TCGA")
            label = 0 if s[pos:pos+16] in self.normal_list else 1
        elif self.name_ == "ESCA":
            # BRCA training and testing data
            pos = s.find("TCGA")
            label = 0 if s[pos:pos+16] in self.normal_list else 1
        else:
            raise ValueError

        # if self.type_ == "train":
        #     dgl_graph = transform(dgl_graph)

        # Add self loop here for homogeneous graphs
        if dgl_graph.is_homogeneous:
            dgl_graph = dgl.add_self_loop(dgl_graph)

        return dgl_graph, label


class TCGACancerSurvivalDataset(DGLDataset):
    def __init__(self, graph_path, type_, name="tcga_survival"):
        """
        :param data_root: Root of graph
        :param label_path: Path to the file contain list of normal images
        """

        self.graph_path = graph_path
        self.type_ = type_

        super().__init__(name)

    def process(self):
        self.graph_df = pd.read_csv(self.graph_path)
        # self.graph_df = self.graph_df.sample(frac=0.5).reset_index(drop=True)


    def __len__(self):
        return len(self.graph_df)

    def __getitem__(self, index):
        graph_path = self.graph_df.loc[index, 'graph_path']

        with open(graph_path, 'rb') as f:
            dgl_graph = pickle.load(f)

        s = os.path.split(graph_path)[1]

        # COAD training and testing data
        pos = s.find("TCGA")
        case_name = s[pos:-4]
        label = []
        label.append(int(self.graph_df.loc[index, 'censorship']))
        label.append(int(self.graph_df.loc[index, 'label']))
        label.append(float(self.graph_df.loc[index, 'survival_months']))


        # Drop graph
        if type(dgl_graph) == dict:
            dgl_graph2 = dgl_graph['het_graph']
            if self.type_ == "train":
                num_ntype = []
                for ntype in dgl_graph2.ntypes:
                    num_ntype.append(dgl_graph2.num_nodes(ntype))
                if min(num_ntype) > 2:
                    transforms = Drop_graph(p=0.3)
                    dgl_graph2 = transforms(dgl_graph2)
                dgl_graph['het_graph'] = dgl_graph2
        else:
            if self.type_ == "train":
                num_ntype = []
                for ntype in dgl_graph.ntypes:
                    num_ntype.append(dgl_graph.num_nodes(ntype))
                if min(num_ntype) > 2:
                    transforms = Drop_graph(p=0.3)
                    dgl_graph = transforms(dgl_graph)

            # Add self loop here for homogeneous graphs
            if dgl_graph.is_homogeneous:
                dgl_graph = dgl.add_self_loop(dgl_graph)    # 为同构图增加自环边

        return dgl_graph, label, case_name

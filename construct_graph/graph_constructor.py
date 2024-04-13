import pickle
import sys
from pathlib import Path
from importlib import import_module
from tqdm import tqdm
import openslide
import random

from typing import OrderedDict
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import numpy as np
import dgl.sparse as dglsp
from termcolor import colored
import nmslib
from itertools import chain
from math import gcd
from functools import reduce

# Graph Network Packages
import dgl
from scipy.stats import pearsonr

# from .extractor import Extractor
from efficientnet_pytorch import EfficientNet
from  models import ctranspath

from data import PatchData
'''
Graph construction v2 for new format of patches

Node types:
From PanNuke dataset
0) No-label '0'
1) Neoplastic '1'
2) Inflammatory '2'
3) Connective '3'
4) Dead '4'
5) Non-Neoplastic Epithelial '5'

Edge types:
0 or 1 based on Personr correlation between nodes
'''


class Hnsw:
    """
    KNN model cloned from https://github.com/mahmoodlab/Patch-GCN/blob/master/WSI-Graph%20Construction.ipynb
    """

    def __init__(self, space='cosinesimil', index_params=None,
                 query_params=None, print_progress=True):
        self.space = space
        self.index_params = index_params
        self.query_params = query_params
        self.print_progress = print_progress

    def fit(self, X):
        index_params = self.index_params
        if index_params is None:
            index_params = {'M': 16, 'post': 0, 'efConstruction': 400}

        query_params = self.query_params
        if query_params is None:
            query_params = {'ef': 90}

        # this is the actual nmslib part, hopefully the syntax should
        # be pretty readable, the documentation also has a more verbiage
        # introduction: https://nmslib.github.io/nmslib/quickstart.html
        index = nmslib.init(space=self.space, method='hnsw')
        index.addDataPointBatch(X)
        index.createIndex(index_params, print_progress=self.print_progress)
        index.setQueryTimeParams(query_params)

        self.index_ = index
        self.index_params_ = index_params
        self.query_params_ = query_params
        return self

    def query(self, vector, topn):
        # the knnQuery returns indices and corresponding distance
        # we will throw the distance away for now
        indices, dist = self.index_.knnQuery(vector, k=topn)
        return indices

class knn:
    def __init__(self, k=20):
        self.k = k

    def calc_dist(self, x0, y0, x1, y1):
        result = (x0 - x1) ** 2 + (y0 - y1) ** 2
        if result == 0:
            return sys.maxsize
        return result
    
    def query(self, pos):
        length = len(pos)
        Dist = np.zeros((length, length))
        for i in range(length):
            j = i
            while j < length:
                Dist[i][j] = Dist[j][i] = self.calc_dist(pos[i][0], pos[i][1], pos[j][0], pos[j][1])
                j += 1

        source_node = []
        target_node = []
        for i in range(length):
            maxKArgs = np.argpartition(Dist[i], self.k)[:self.k]
            for j in maxKArgs:
                source_node.append(i)
                target_node.append(j)

        return  np.array(source_node), np.array(target_node)

def convert_pytorch_checkpoint(net_state_dict):
    variable_name_list = list(net_state_dict.keys())
    is_in_parallel_mode = all(v.split(".")[0] == "module" for v in variable_name_list)
    if is_in_parallel_mode:
        colored_word = colored("WARNING", color="red", attrs=["bold"])
        print(
            (
                    "%s: Detect checkpoint saved in data-parallel mode."
                    " Converting saved model to single GPU mode." % colored_word
            ).rjust(80)
        )
        net_state_dict = {
            ".".join(k.split(".")[1:]): v for k, v in net_state_dict.items()
        }
    return net_state_dict


class Hovernet_infer:
    """ Run HoverNet inference """

    def __init__(self, config, dataloader):
        self.dataloader = dataloader

        method_args = {
            'method': {
                'model_args': {'nr_types': config['nr_types'], 'mode': config['mode'], },
                'model_path': config['hovernet_model_path'],
            },
            'type_info_path': config['type_info_path'],
        }
        run_args = {
            'batch_size': config['batch_size'],
        }

        model_desc = import_module("models.hovernet.net_desc")
        model_creator = getattr(model_desc, "create_model")
        net = model_creator(**method_args['method']["model_args"])
        saved_state_dict = torch.load(method_args['method']["model_path"])["desc"]
        saved_state_dict = convert_pytorch_checkpoint(saved_state_dict)
        net.load_state_dict(saved_state_dict, strict=False)
        net = torch.nn.DataParallel(net)
        net = net.to("cuda")

        module_lib = import_module("models.hovernet.run_desc")
        run_step = getattr(module_lib, "infer_step")
        self.run_infer = lambda input_batch: run_step(input_batch, net)
        

    def predict(self):
        output_list = []
        features_list = []
        pbar = tqdm(desc='segment with hovernet', total=len(self.dataloader))
        for idx, data in enumerate(self.dataloader):
            data = data.permute(0, 3, 2, 1)
            output, features = self.run_infer(data)
            # curr_batch_size = output.shape[0]
            # output = np.split(output, curr_batch_size, axis=0)[0].flatten()
            features_list.append(features)
            for out in output:
                if out.any() == 0:
                    output_list.append(0)
                else:
                    out = out[out != 0]
                    max_occur_node_type = np.bincount(out).argmax()
                    output_list.append(max_occur_node_type)
            pbar.update(1)
        pbar.close()

        return output_list, np.concatenate(features_list)


class flatten(nn.Module):
    """docstring for BottleNeck"""

    def __init__(self, model):
        super(flatten, self).__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        return x


class KimiaNet_infer:
    def __init__(self, config, dataloader):
        self.config = config
        self.dataloader = dataloader

        self.model = torchvision.models.densenet121(weights='IMAGENET1K_V1')
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.features = nn.Sequential(self.model.features, nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.model_final = flatten(self.model.features)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_final = nn.DataParallel(self.model_final)
        self.model_final = self.model_final.to(self.device)
        # self.model_final = nn.DataParallel(self.model_final)
        state_dict = torch.load(self.config['kimianet_model_path'])
        sd = self.model_final.state_dict()
        for (k, v), ky in zip(state_dict.items(), sd.keys()):
            sd[ky] = v
        self.model_final.load_state_dict(sd)

    def predict(self):
        self.model_final.eval()
        features_list = []
        for idx, data in enumerate(self.dataloader):
            # data = data.permute(0, 3, 2, 1)
            data = data.to(self.device)
            output = self.model_final(data)
            output_1024 = output.cpu().detach().numpy()
            features_list.append(output_1024)
        return np.concatenate(features_list)


class EfficientNet_infer:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_final = EfficientNet.from_pretrained('efficientnet-b4', num_classes=1024).to(self.device)

    def predict(self):
        self.model_final.eval()
        features_list = []
        for idx, data in enumerate(self.dataloader):
            # data = data.permute(0, 3, 2, 1)
            data = data.to(self.device)
            output1 = self.model_final(data)
            output_1024 = output1.cpu().detach().numpy()
            features_list.append(output_1024)
        return np.concatenate(features_list)
    
class cTransPath_infer:
    def __init__(self, config, dataloader):
        def clean_state_dict_clip(state_dict):
            new_state_dict = {}
            for k, v in state_dict.items():
                if 'attn_mask' in k:
                    continue
                if 'visual.trunk.' in k:
                    new_state_dict[k.replace('visual.trunk.', '')] = v
            return new_state_dict
        
        self.config = config
        self.dataloader = dataloader

        self.model = ctranspath(img_size = 224)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.head = nn.Identity()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
        state_dict = torch.load(self.config['ctrans_model_path'], map_location="cpu")['state_dict']
        state_dict = clean_state_dict_clip(state_dict)
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        # print('missing keys: ', missing_keys)
        # print('unexpected keys: ', unexpected_keys)

    def predict(self):
        self.model.eval()
        features_list = []
        for idx, data in enumerate(self.dataloader):
            # data = data.permute(0, 3, 2, 1)
            data = data.to(self.device)
            output = self.model(data)
            output_1024 = output.cpu().detach().numpy()
            features_list.append(output_1024)
        return np.concatenate(features_list)

class GraphConstructor:
    def __init__(self, config: OrderedDict, encoder_config: OrderedDict, patch_data, type_info):
        self.config = config
        self.encoder_config = encoder_config
        self.patch_data = patch_data
        self.radius = config['radius']
        self.knn_model = knn(k=self.radius)

        patch_path = Path(patch_data)
        patch_dataset = PatchData(patch_path)
        self.node_type = [type_info[p] for p in patch_dataset.node_type]
        self.node_pos = patch_dataset.node_pos
        
        
        dataloader = data.DataLoader(
            patch_dataset,
            num_workers=4,
            batch_size=encoder_config['batch_size']*encoder_config['gpu_num'],
            shuffle=False
        )

        self.encoder_name = config['encoder_name']
        if self.encoder_name == "kimia":
            # print("Use KimiaNet pretrained model!")
            kimia_model = KimiaNet_infer(self.encoder_config, dataloader)
            self.features = kimia_model.predict()
        elif self.encoder_name == "efficientnet-b4":
            encoder = EfficientNet_infer(dataloader)
            self.features = encoder.predict()

    def construct_graph(self):

        ####################
        # Step 1 Fit the coordinate with KNN and construct edges
        ####################
        # Number of patches
        n_patches = self.features.shape[0]

        # Construct graph using spatial coordinates
        a, b = self.knn_model.query(self.node_pos)
        edge_spatial = torch.Tensor(np.stack([a, b])).type(torch.LongTensor)

        # Create edge types
        edge_type = []
        edge_sim = []
        for (idx_a, idx_b) in zip(a, b):
            metric = pearsonr
            corr = metric(self.features[idx_a], self.features[idx_b])[0]
            edge_type.append(1 if corr > 0 else 0)
            edge_sim.append(corr)

        # Construct dgl heterogeneous graph
        graph = dgl.graph((edge_spatial[0, :], edge_spatial[1, :]))
        graph.ndata.update({'_TYPE': torch.tensor(self.node_type)})
        self.features = torch.tensor(self.features, device='cpu').float()
        # self.patches_coords = torch.tensor(self.patches_coords, device='cpu').float()
        graph.ndata['feat'] = self.features
        # graph.ndata['patches_coords'] = self.patches_coords
        graph.edata.update({'_TYPE': torch.tensor(edge_type)})
        graph.edata.update({'sim': torch.tensor(edge_sim)})
        het_graph = dgl.to_heterogeneous(
            graph,
            [str(t) for t in range(self.config["n_node_type"])],
            ['neg', 'pos']
        )

        homo_graph = dgl.graph((edge_spatial[0, :], edge_spatial[1, :]))
        homo_graph.ndata['feat'] = self.features
        # homo_graph.ndata['patches_coords'] = self.patches_coords

        return het_graph, homo_graph, self.node_type


class HyperGraphConstructor:
    def __init__(self, config: OrderedDict, hovernet_config: OrderedDict, encoder_config: OrderedDict, 
                h5_path=None, wsi_path=None, pkl_path=None, just_extract_embeddings=False):
        self.config = config
        self.encoder_config = encoder_config
        self.hovernet_config = hovernet_config

        self.new_graph_type = config['new_graph_type']
        self.cell_radius = config['cell_radius']
        self.tissue_radius = config['tissue_radius']
        self.knn_model = Hnsw(space='l2')

        wsi = openslide.OpenSlide(wsi_path)
        MAG = int(float(wsi.properties['aperio.AppMag']))

        if not pkl_path or just_extract_embeddings:

            patch_dataset = PatchData(h5_path, wsi, config['encoder_patch_size'])
            dataloader = data.DataLoader(
                patch_dataset,
                num_workers=4,
                batch_size=encoder_config['batch_size']*encoder_config['gpu_num'],
                shuffle=False
            )
            self.node_pos = patch_dataset.node_pos

            self.encoder_name = config['encoder_name']
            if self.encoder_name == "kimia":
                # print("Use KimiaNet pretrained model!")
                kimia_model = KimiaNet_infer(self.encoder_config, dataloader)
                self.features = kimia_model.predict()
            elif self.encoder_name == "efficientnet-b4":
                encoder = EfficientNet_infer(dataloader)
                self.features = encoder.predict()
            elif self.encoder_name == "ctrans":
                ctranspath_model = cTransPath_infer(self.encoder_config, dataloader)
                self.features = ctranspath_model.predict()

            if not pkl_path:    #没有 pkl 文件，则需要用 hovernet 进行预测
                if self.encoder_name == 'ctrans':
                    patch_dataset = PatchData(h5_path, wsi, self.hovernet_config['hover_patch_size'])
                    dataloader = data.DataLoader(
                        patch_dataset,
                        num_workers=4,
                        batch_size=encoder_config['batch_size']*encoder_config['gpu_num'],
                        shuffle=False
                    )
                hovernet_model = Hovernet_infer(self.hovernet_config, dataloader)
                self.node_type, _ = hovernet_model.predict()

            elif just_extract_embeddings:   # 仅提取特征，则不需要用 hovernet 进行预测，只使用 encoder 进行特征提取
                with open(pkl_path, 'rb') as f:
                    data2 = pickle.load(f)
                self.node_type = data2['type']
            self.HIM, self.info_dict = self.process_big_patch(MAG, use_all=False)


        else:
            with open(pkl_path, 'rb') as f:
                data2 = pickle.load(f)
            self.node_pos = data2['pos']
            self.node_type = data2['type']
            self.features = data2['feat']
            self.HIM, self.info_dict = self.process_big_patch(MAG, use_all=False)

    def construct_graph(self, single_edge=False):

        ####################
        # Step 1 Fit the coordinate with KNN and construct edges
        ####################
        # Number of patches
        n_patches = self.features.shape[0]

        # Construct graph using spatial coordinates
        self.knn_model.fit(self.node_pos)
        a = np.repeat(range(n_patches), self.cell_radius-1)
        b = np.fromiter(
            chain(
                *[self.knn_model.query(self.node_pos[v_idx], topn=self.cell_radius)[1:] for v_idx in range(n_patches)]
            ), dtype=int
        )
        edge_spatial = torch.Tensor(np.stack([a, b])).type(torch.LongTensor)
        

        # Create edge types
        if single_edge:
            edge_type = [0 for _ in range(len(a))]
            b = b.reshape(-1, self.cell_radius-1)
            # Construct dgl heterogeneous graph
            graph = dgl.graph((edge_spatial[0, :], edge_spatial[1, :]))
            graph.ndata.update({'_TYPE': torch.tensor(self.node_type)})
            self.features = torch.tensor(self.features, device='cpu').float()
            graph.ndata['feat'] = self.features
            graph.edata.update({'_TYPE': torch.tensor(edge_type)})
            graph.ndata['neighbor'] = torch.tensor(b)
            het_graph = dgl.to_heterogeneous(
                graph,
                [str(t) for t in range(self.config["n_node_type"])],
                ['edge']
            )
        else:
            # 两种 edgetype
            edge_type = []
            edge_sim = []
            for (idx_a, idx_b) in zip(a, b):
                metric = pearsonr
                corr = metric(self.features[idx_a], self.features[idx_b])[0]
                edge_type.append(1 if corr > 0 else 0)
                edge_sim.append(corr)
            
            b = b.reshape(-1, self.cell_radius-1)

            # Construct dgl heterogeneous graph
            graph = dgl.graph((edge_spatial[0, :], edge_spatial[1, :]))
            graph.ndata.update({'_TYPE': torch.tensor(self.node_type)})
            self.features = torch.tensor(self.features, device='cpu').float()
            graph.ndata['feat'] = self.features
            graph.edata.update({'_TYPE': torch.tensor(edge_type)})
            graph.edata.update({'sim': torch.tensor(edge_sim)})
            graph.ndata['neighbor'] = torch.tensor(b)
            het_graph = dgl.to_heterogeneous(
                graph,
                [str(t) for t in range(self.config["n_node_type"])],
                ['neg', 'pos']
            )

        return het_graph
    
    def process_big_patch(self, MAG, use_all=False):
        tissue_info = {}
        tissue_patch_pos = []
        tissue_features = []
        mix_type_index = []



        if MAG ==40:
            patch_size = 512
        elif MAG == 20:
            patch_size = 256

        index = 0
        x_sum = 0
        y_sum = 0
        node_pos_dict = {(point[0], point[1]): i for i, point in enumerate(self.node_pos)}
        _x_dict = {}

        has_proce_dic = {}
        for (x,y) in self.node_pos:
            if has_proce_dic.get((x,y), False):
                continue
            else:
                patch_idx_list = [node_pos_dict.get((x, y), None),
                node_pos_dict.get((x + patch_size, y), None),
                node_pos_dict.get((x, y + patch_size), None),
                node_pos_dict.get((x + patch_size, y + patch_size), None)]
                if any(idx is None for idx in patch_idx_list):
                    continue
                else:
                    has_proce_dic[(x,y)] = True
                    has_proce_dic[(x + patch_size, y)] = True
                    has_proce_dic[(x, y + patch_size)] = True
                    has_proce_dic[(x + patch_size, y + patch_size)] = True
                    sub_hyper_dict = {
                        'pos': [x, y],
                        'index': patch_idx_list,
                        'type': [self.node_type[idx] for idx in patch_idx_list]
                    }

                    if {1, 3}.issubset(sub_hyper_dict['type']):
                        mix_type_index.append(index)

                    tissue_info[index] = sub_hyper_dict
                    tissue_features.append(self.features[patch_idx_list[0]])
                    tissue_patch_pos.append([x, y])
                    x_sum += x
                    y_sum += y
                    index += 1
                    if x not in _x_dict.keys():
                        _x_dict[x] = [y]
                    else:
                        _x_dict[x].append(y)


        # x_min = np.min(self.node_pos[:, 0])
        # x_max = np.max(self.node_pos[:, 0])
        # y_min = np.min(self.node_pos[:, 1])
        # y_max = np.max(self.node_pos[:, 1])

        # for x in range(x_min, x_max, 2 * patch_size):
        #     for y in range(y_min, y_max, 2 * patch_size):
        #         patch_idx_list = [node_pos_dict.get((x, y), None),
        #                         node_pos_dict.get((x + patch_size, y), None),
        #                         node_pos_dict.get((x, y + patch_size), None),
        #                         node_pos_dict.get((x + patch_size, y + patch_size), None)]

        #         if any(idx is None for idx in patch_idx_list):
        #             continue

        #         sub_hyper_dict = {
        #             'pos': [x, y],
        #             'index': patch_idx_list,
        #             'type': [self.node_type[idx] for idx in patch_idx_list]
        #         }

        #         if {1, 3}.issubset(sub_hyper_dict['type']):
        #             mix_type_index.append(index)

        #         tissue_info[index] = sub_hyper_dict
        #         tissue_features.append(self.features[patch_idx_list[0]])
        #         tissue_patch_pos.append([x, y])
        #         x_sum += x
        #         y_sum += y
        #         index += 1
        #         if x not in _x_dict.keys():
        #             _x_dict[x] = [y]
        #         else:
        #             _x_dict[x].append(y)
            

        node_pos_dict = {(point[0], point[1]): i for i, point in enumerate(tissue_patch_pos)}
        center_point = np.array((x_sum // len(tissue_patch_pos), y_sum // len(tissue_patch_pos)))

        selected_boundary_index = []    #boundary
        selected_boundary_index_2 = []  #2/3 boundary
        selected_boundary_index_3 = []  #1/3 boundary
        for key,value in _x_dict.items():
            if len(value) < 2:
                continue
            min_value = min(value)
            max_value = max(value)
            selected_boundary_index.append(node_pos_dict.get((key, min_value), None))
            selected_boundary_index.append(node_pos_dict.get((key, max_value), None))
        
        for idx in selected_boundary_index:
            if idx is None:
                continue
            point_2 = np.array([int((tissue_patch_pos[idx][0]+center_point[0])*2/3), int((tissue_patch_pos[idx][1]+center_point[1])*2/3)])
            point_3 = np.array([int((tissue_patch_pos[idx][0]+center_point[0])/3), int((tissue_patch_pos[idx][1]+center_point[1])/3)])
            distances_to_point2 = np.linalg.norm(np.array(tissue_patch_pos) - point_2, axis=1)
            distances_to_point3 = np.linalg.norm(np.array(tissue_patch_pos) - point_3, axis=1)
            closest_index_2 = np.argmin(np.abs(distances_to_point2))
            closest_index_3 = np.argmin(np.abs(distances_to_point3))
            selected_boundary_index_2.append(closest_index_2)
            selected_boundary_index_3.append(closest_index_3)
        selected_boundary_index = list(set(selected_boundary_index))
        selected_boundary_index_2 = list(set(selected_boundary_index_2))
        selected_boundary_index_3 = list(set(selected_boundary_index_3))
        all_index = range(len(tissue_patch_pos))
        remain_index = list(set(all_index) - set(selected_boundary_index) - set(selected_boundary_index_2) - set(selected_boundary_index_3) - set(mix_type_index))
        if use_all or len(remain_index) < 500:
            random_select = remain_index
        else:
            random_select = random.sample(remain_index, 500)

        if self.new_graph_type == 'except_topo':
            remain_index =  list(set(all_index) - set(mix_type_index))
            selected_boundary_index = random.sample(remain_index, len(selected_boundary_index))
            # remain_index =  list(set(remain_index) - set(selected_boundary_index))

            selected_boundary_index_2 = random.sample(remain_index, len(selected_boundary_index_2))
            # remain_index =  list(set(remain_index) - set(selected_boundary_index_2))

            selected_boundary_index_3 = random.sample(remain_index, len(selected_boundary_index_3))
        elif self.new_graph_type == 'except_mix':
            ramain_index = list(set(all_index) - set(selected_boundary_index) - set(selected_boundary_index_2) - set(selected_boundary_index_3))
            mix_type_index = random.sample(ramain_index, len(mix_type_index))
            
        selected_tissue_patch_index = list(set(selected_boundary_index + selected_boundary_index_2 + selected_boundary_index_3 + mix_type_index + random_select))
        selected_features = [tissue_features[idx] for idx in selected_tissue_patch_index]
        _dict = {i: idx for i, idx in enumerate(selected_tissue_patch_index)}
        self.knn_model.fit(selected_features)

        hedge_list = [i for i in range(len(selected_tissue_patch_index)) for _ in range(self.tissue_radius-1)]
        _node_list = np.fromiter(
            chain(
                *[self.knn_model.query(selected_features[v_idx], topn=self.tissue_radius)[1:] for v_idx in range(len(selected_features))]
            ), dtype=int
        )
        node_list = [_dict.get(i, None) for i in _node_list]

        offset = len(selected_tissue_patch_index)
        node_list.extend(selected_boundary_index)
        node_list.extend(selected_boundary_index_2)
        node_list.extend(selected_boundary_index_3)
        node_list.extend(mix_type_index)

        hedge_list.extend([offset] * len(selected_boundary_index))
        hedge_list.extend([offset+1] * len(selected_boundary_index_2))
        hedge_list.extend([offset+2] * len(selected_boundary_index_3))
        hedge_list.extend([offset+3] * len(mix_type_index))
        H = dglsp.spmatrix(torch.LongTensor([node_list, hedge_list]), shape=(len(tissue_patch_pos), len(selected_tissue_patch_index)+4))
        H = H.to_dense()
        return_dict = {key: tissue_info[key] for key in selected_tissue_patch_index}
        return H, return_dict

class get_label_features:
    def __init__(self, config: OrderedDict, hovernet_config: OrderedDict, encoder_config: OrderedDict, h5_path, wsi_path):
        self.config = config
        self.encoder_config = encoder_config
        self.hovernet_config = hovernet_config
        wsi = openslide.OpenSlide(wsi_path)
        patch_dataset = PatchData(h5_path, wsi, config['encoder_patch_size'])
        
        self.node_pos = patch_dataset.node_pos
        
        
        dataloader = data.DataLoader(
            patch_dataset,
            num_workers=4,
            batch_size=encoder_config['batch_size']*encoder_config['gpu_num'],
            shuffle=False
        )

        self.encoder_name = config['encoder_name']
        if self.encoder_name == "kimia":
            # print("Use KimiaNet pretrained model!")
            kimia_model = KimiaNet_infer(self.encoder_config, dataloader)
            self.features = kimia_model.predict()
        elif self.encoder_name == "efficientnet-b4":
            encoder = EfficientNet_infer(dataloader)
            self.features = encoder.predict()
        elif self.encoder_name == "ctrans":
            ctranspath_model = cTransPath_infer(self.encoder_config, dataloader)
            self.features = ctranspath_model.predict()
        if self.encoder_name == 'ctrans':
            patch_dataset = PatchData(h5_path, wsi, self.hovernet_config['hover_patch_size'])
            dataloader = data.DataLoader(
                patch_dataset,
                num_workers=4,
                batch_size=encoder_config['batch_size']*encoder_config['gpu_num'],
                shuffle=False
            )
        hovernet_model = Hovernet_infer(self.hovernet_config, dataloader)
        self.node_type, _ = hovernet_model.predict()

    def get(self):
        return_dict = {}
        return_dict['pos'] = self.node_pos
        return_dict['type'] = self.node_type
        return_dict['feat'] = self.features

        return return_dict
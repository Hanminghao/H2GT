import argparse
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from collections import OrderedDict

import os
import pickle
import glob
from pathlib import Path
from construct_graph import HyperGraphConstructor
from tqdm import tqdm


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




parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to option YMAL file.', default="configs/graph_construction_config.yml")
# parser.add_argument('--pkl_path', type=str, help='The patch type and feature have been processed in advance', default='/home/dataset2/hmh_data/my_secpro_data/patch_label_features/BLCA/HoverNet_CTrans_20X/')
parser.add_argument('--pkl_path', type=str, help='The patch type and feature have been processed in advance', default=None)
parser.add_argument('--gpu_list', type=str, help='Which Gpus to use', default='6,7')
parser.add_argument('--just_extract_embeddings', type=bool, help='Whether to just extract embeddings', default=False)

args = parser.parse_args()

opt_path = args.config
gpu_list = args.gpu_list
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

type_info = {
     "nolae": 0,
     "neopla": 1,
     "inflam": 2,
     "connec": 3,
     "necros": 4,
     "no-neo": 5,
}


def main():

    with open(opt_path, mode='r') as f:
            loader, _ = ordered_yaml()
            config = yaml.load(f, loader)
            print(f"Loaded configs from {opt_path}")

    graph_config = config['graph_constructor']
    encoder_config = config['encoder_config']
    encoder_config['gpu_num'] = (len(gpu_list)+1)//2
    hovernet_config = config['hovernet_config']
    if not args.pkl_path:   #从头开始，包括适用 HoverNet 分割和 Encodr 提取特征
        h5_paths = glob.glob(graph_config['h5_path'] + '/*')
        pbar = tqdm(desc='construct graph from h5 files' ,total=len(h5_paths))
        for h5_path in h5_paths:
            try:
                _, tail = os.path.split(h5_path)
                tail = tail[:-3]
                het_output_file = os.path.join(graph_config['out_dir'] + tail + '.pkl')
                if Path(het_output_file).exists():
                        pbar.update(1)
                        continue
                wsi_path = os.path.join(graph_config['wsi_path'], tail + '.svs')
                graph_constructor = HyperGraphConstructor(graph_config, hovernet_config, encoder_config, h5_path=h5_path, 
                                                        wsi_path=wsi_path)
                het_graph, HIM, info_dict = graph_constructor.construct_graph()
                if int((HIM.sum(dim=1) == 0).sum()) > 0:
                    pbar.update(1)
                    continue
                save_dict = {
                    'het_graph': het_graph,
                    'HIM': HIM,
                    'info_dict': info_dict
                }
                # Make directory
                if not Path(graph_config['out_dir']).exists():
                    Path(graph_config['out_dir']).mkdir(parents=True)

                with open(het_output_file, 'wb') as f:
                    pickle.dump(save_dict, f)
                pbar.update(1)

            except (ValueError, KeyError, IndexError, RuntimeError, FileNotFoundError):
                print("Failed to construct {} graph, moves to next WSI image".format(tail))
                pbar.update(1)
            except ZeroDivisionError:
                print("Failed to construct {} graph,this svs file is to small, moves to next WSI image".format(tail))
                pbar.update(1)
        pbar.close()

    else:
        pkl_paths = glob.glob(args.pkl_path + '/*')
        pbar = tqdm(desc='construct graph from pkl files' ,total=len(pkl_paths))
        for pkl_path in pkl_paths:
            try:
                _, tail = os.path.split(pkl_path)
                if tail not in os.listdir(graph_config['normal_graph_path']):
                    continue
                tail = tail[:-4]
                wsi_path = os.path.join(graph_config['wsi_path'], tail + '.svs')
                h5_path = os.path.join(graph_config['h5_path'], tail + '.h5')
                het_output_file = os.path.join(graph_config['out_dir']  + tail + '.pkl')
                if Path(het_output_file).exists():
                        pbar.update(1)
                        continue
                graph_constructor = HyperGraphConstructor(graph_config, hovernet_config, encoder_config, pkl_path=pkl_path, h5_path=h5_path,
                                    wsi_path=wsi_path, just_extract_embeddings=args.just_extract_embeddings)
                if int((graph_constructor.HIM[:,-4:].sum(dim=0) == 0).sum()) > 0:
                    pbar.update(1)
                    continue
                het_graph = graph_constructor.construct_graph()
                save_dict = {
                    'het_graph': het_graph,
                    'HIM': graph_constructor.HIM,
                    'info_dict': graph_constructor.info_dict
                }
                # Make directory
                if not Path(graph_config['out_dir']).exists():
                    Path(graph_config['out_dir']).mkdir(parents=True)

                with open(het_output_file, 'wb') as f:
                    pickle.dump(save_dict, f)
                pbar.update(1)

            except (ValueError, KeyError, IndexError, RuntimeError, FileNotFoundError):
                print("Failed to construct {} graph, moves to next WSI image".format(tail))
                pbar.update(1)
            except ZeroDivisionError:
                print("Failed to construct {} graph,this svs file is to small, moves to next WSI image".format(tail))
                pbar.update(1)
        pbar.close()
    print('The process is complete. The code is over !')





if __name__ == '__main__':
    main()

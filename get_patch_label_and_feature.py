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
from construct_graph import get_label_features
from tqdm import tqdm
import os
import pickle
from pathlib import Path
from tqdm import tqdm

from typing import OrderedDict




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
parser.add_argument('-config', type=str, help='Path to option YMAL file.', default="configs/my_config.yml")
parser.add_argument('--gpu_list', type=str, help='Which Gpus to use', default='4,5,6')
parser.add_argument('--pkl_path', type=str, default=None)
parser.add_argument('--save_graph_path', type=str)
parser.add_argument('--get_graph', type=bool, default=False)
args = parser.parse_args()

opt_path = args.config
gpu_list = args.gpu_list
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
fh5_path = '/Dataset4/CLAM/BRCA_processed_20X/patches/'
wsi_paths = '/Dataset3/BRCA/'

type_info = {
     "nolae": 0,
     "neopla": 1,
     "inflam": 2,
     "connec": 3,
     "necros": 4,
     "no-neo": 5,
}

def get_graph():
    return 1

def main():

    with open(opt_path, mode='r') as f:
            loader, _ = ordered_yaml()
            config = yaml.load(f, loader)
            print(f"Loaded configs from {opt_path}")

    graph_config = config['graph_constructor']
    encoder_config = config['encoder_config']
    encoder_config['gpu_num'] = (len(gpu_list)+1)//2
    hovernet_config = config['hovernet_config']


    h5_paths = glob.glob(fh5_path + '/*')

    if not args.pkl_path:
        pbar = tqdm(desc='construct graph' ,total=len(h5_paths))

        ### 优先构建diag的图
        diag_lines = []  
        with open('/Dataset3/hover_net_output/Diagnostic.txt', 'r') as file:
            for line in file:
                line = line.strip()[:-4]  # 去除行尾的换行符和空白字符
                diag_lines.append(line)
        ####

        for h5_path in h5_paths:
            #print("processing {}/{} wsi".format(i, len(patch_path)))
            _, tail = os.path.split(h5_path)
            tail = tail[:-3]
            if tail not in diag_lines:
                pbar.update(1)
                print("Skip this image, not diag image")
                continue
            output_file = os.path.join(graph_config['out_dir'] +  tail + '.pkl')
            if Path(output_file).exists():
                pbar.update(1)
                print("Has processed, skip")
                continue
            wsi_path = os.path.join(wsi_paths, tail + '.svs')
            get_constructor = get_label_features(graph_config, hovernet_config, encoder_config, h5_path, wsi_path)
            dic = get_constructor.get()

            # Make directory
            if not Path(graph_config['out_dir']).exists():
                Path(graph_config['out_dir']).mkdir(parents=True)

            with open(output_file, 'wb') as f:
                pickle.dump(dic, f)
            get_graph()
            pbar.update(1)

        pbar.close()
        print('The process is complete. The code is over !')





if __name__ == '__main__':
    main()

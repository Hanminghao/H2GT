import numpy as np
import torch

from torch.nn import functional as F
from .evaluator import Evaluator
from data import TCGACancerSurvivalDataset
from parser_1 import parse_gnn_model, parse_loss
from sksurv.metrics import concordance_index_censored
from checkpoint import CheckpointManager
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm


class HomoGraphEvaluator_Survival(Evaluator):

    def __init__(self, config, verbose=True, pkl_path=None, fold=None, val=True, visualize=False):
        super().__init__(config, verbose)
        
        self.visualize = visualize
        self.checkpoint_manager = CheckpointManager(path=config['checkpoint']["path"]+ '/' +  config['desc'] + '/' + str(fold))

        if verbose:
            print(
                f"Loaded checkpoint with path {config['checkpoint']['path']} version {self.checkpoint_manager.version}"
            )
        # Initialize GNN model and optimizer
        self.gnn = parse_gnn_model(self.config_gnn, visualize=visualize).to(self.device)

        # Load trained checkpoint
        if pkl_path is None:
            state_dict = self.checkpoint_manager.load_model()
        else:
            state_dict = torch.load(pkl_path)
            print("Model Weight load model from ", pkl_path)
        self.gnn.load_state_dict(state_dict)
        self.gnn.eval()
        self.name = self.config_data["dataset"]        
        self.loss_fcn = parse_loss(self.config_train)
        
        # Load testing data
        if fold is None:
            if val:
                val_path = self.config_data["valid_path"]
            else:
                val_path = self.config_data["test_path"]
            self.test_data = self.load_data(val_path)
        else:
            if val:
                val_path = self.config_data["tvt_root"] + '/' + str(fold) + self.config_data["valid_path"]
            else:
                val_path = self.config_data["tvt_root"] + '/' + str(fold) + self.config_data["test_path"]
            print("Load data from ", val_path)
            self.test_data = self.load_data(val_path)
            print("val data size: ", len(self.test_data))
            
    def load_data(self, path):
        name = self.config_data["dataset"]
        task = self.config_data["task"]
        if name == "BRCA" or name == "GBMLGG" or name == 'BLCA' or name == 'LIHC' and task == 'cancer survival':
            self.average = "macro"
            test_data = TCGACancerSurvivalDataset(path, 'val')
        return test_data


    def test_one_survival_step(self, graphs, label, case_name):

        censorship = label[0].to(self.device)
        event_time = label[2].to(self.device)
        label = label[1].to(self.device)
        vis_info = None

        with torch.no_grad():
            if type(graphs) == dict:
                graphs['het_graph'] = graphs['het_graph'].to(self.device)
                if self.visualize:
                    pred, vis_info = self.gnn(graphs, case_name)
                    vis_info.append(event_time)
                else:
                    pred = self.gnn(graphs, case_name)
            else:
                graphs = graphs.to(self.device)
                if self.visualize:
                    pred, vis_info = self.gnn(graphs, case_name)
                    vis_info.append(event_time)
                else:
                    pred = self.gnn(graphs, case_name)

            Y_hat = torch.topk(pred, 1, dim = 1)[1]
            hazards = torch.sigmoid(pred)
            S = torch.cumprod(1 - hazards, dim=1)
            
        loss = self.loss_fcn(hazards=hazards, S=S, Y=label, c=censorship)
        if self.visualize:
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            vis_info.append(risk)
        

        return loss, S, censorship, event_time, hazards, vis_info

    def eval(self):
        vis_info_dict = {}
        self.dataloader = GraphDataLoader(
            self.test_data,
            batch_size=1,
            shuffle=True,
            drop_last=False
        )

        val_loss = 0.
        all_risk_scores = np.zeros((len(self.test_data)))
        all_censorships = np.zeros((len(self.test_data)))
        all_event_times = np.zeros((len(self.test_data)))

        # pbar = tqdm(desc='Valid one epoch' ,total=len(self.dataloader), ncols=150)
        for idx, (graph, label, case_name) in enumerate(self.dataloader):
            loss, S, c, event_time, hazards, vis_info = self.test_one_survival_step(graph, label, case_name)
            
            loss_value = loss.item()
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            vis_info_dict[case_name[0]] = vis_info
            all_risk_scores[idx] = risk
            all_censorships[idx] = c.item()
            all_event_times[idx] = event_time

            val_loss += loss_value
        #     pbar.update(1)
        # pbar.close()

        val_loss = val_loss/len(self.test_data)
        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        dic = {}
        for idx in range(len(all_risk_scores)):
            dic[all_risk_scores[idx]] =[all_event_times[idx], (1-all_censorships[idx])]
        return c_index, val_loss, vis_info_dict, dic
from collections import OrderedDict
import os
import sys
import torch
import pickle
import numpy as np
from tqdm import tqdm
from trainer import Trainer
from checkpoint import CheckpointManager
from dgl.dataloading import GraphDataLoader
from data import  TCGACancerSurvivalDataset
from torch.utils.tensorboard import SummaryWriter   
from evaluator import  HomoGraphEvaluator_Survival
from sksurv.metrics import concordance_index_censored
from parser_1 import parse_optimizer, parse_gnn_model, parse_loss


class GNNTrainer(Trainer):
    def __init__(self, config: OrderedDict, fold):
        super().__init__(config)

        # Initialize number of fold
        self.fold = fold
        # Initialize GNN model and optimizer
        self.gnn = parse_gnn_model(self.config_gnn).to(self.device)
        total_params = sum(p.numel() for p in self.gnn.parameters())
        print(f"Total Parameters: {total_params}")
        self.optimizer = parse_optimizer(self.config_optim, self.gnn)

        # Parse loss function
        self.loss_fcn = parse_loss(self.config_train)
        # Define checkpoint managers
        self.checkpoint_manager = CheckpointManager(path=self.config_checkpoint['path'] + '/' + config['desc'] + '/' + str(self.fold))
        # self.earlystopping = EarlyStopping(warmup=config['Early_stop']['warmup'], patience=config['Early_stop']['patience'], stop_epoch=config['Early_stop']['stop'], verbose = True)

        train_path = self.config_data["tvt_root"] + "/" + str(self.fold) + self.config_data["train_path"]
        self.valid_path = self.config_data["tvt_root"] + "/" + str(self.fold) + self.config_data["valid_path"]

        name = self.config_data["dataset"]
        task = self.config_data["task"]
        if name == "BRCA" or name == "GBMLGG" or name == 'BLCA' or name == 'LIHC' and task == 'cancer survival':
            self.average = "macro"
            print('Load data from {}'.format(train_path))
            train_data = TCGACancerSurvivalDataset(train_path, 'train')


        self.dataloader = GraphDataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )

    def train_one_survival_step(self, graphs, label, case_name):
        censorship = label[0].to(self.device)
        event_time = label[2].to(self.device)
        label = label[1].to(self.device)

        if type(graphs) == dict:
            graphs['het_graph'] = graphs['het_graph'].to(self.device)
            pred = self.gnn(graphs, case_name)
        else:
            graphs = graphs.to(self.device)
            pred = self.gnn(graphs, case_name)


        hazards = torch.sigmoid(pred)
        S = torch.cumprod(1 - hazards, dim=1)
        loss = self.loss_fcn(hazards=hazards, S=S, Y=label, c=censorship)
        return loss, S, censorship, event_time, hazards
    
    def train(self) -> None:
        print(f"Start training Homogeneous GNN in " + str(self.fold))
        log_path = './tensorboard_log/' + self.config_data['dataset'] + '/' + self.config['desc'] + '/' + str(self.fold)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        writer = SummaryWriter(log_path)
        sys.stdout.write("number of training data: " + str(len(self.dataloader)) + "\n")
        for epoch in tqdm(range(self.n_epoch), ncols=150):
                self.gnn.train()
                # 生存分析
                if self.config_data["task"] == 'cancer survival':
                    train_loss = 0.
                    all_risk_scores = np.zeros((len(self.dataloader)))
                    all_censorships = np.zeros((len(self.dataloader)))
                    all_event_times = np.zeros((len(self.dataloader))) 
                    # pabr = tqdm(desc='Train one epoch' ,total=len(self.dataloader), ncols=150)

                    # for idx, (graphs, label, case_name) in tqdm(enumerate(self.dataloader), total=len(self.dataloader), ncols=150):
                    for idx, (graphs, label, case_name) in enumerate(self.dataloader):
                        loss, S, c, event_time, hazards = self.train_one_survival_step(graphs, label, case_name)
                        loss_value = loss.item()

                        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
                        all_risk_scores[idx] = risk
                        all_censorships[idx] = c.item()
                        all_event_times[idx] = event_time

                        train_loss += loss_value
                        
                        loss = loss / self.config_optim['gc']

                        loss.backward()
                        if (idx + 1) % self.config_optim['gc'] == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                        # pabr.update(1)
                    # pabr.close()
                    


                    train_loss = train_loss / len(self.dataloader)
                    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

                    # Perform validation and testing
                    self.checkpoint_manager.save_model(self.gnn.state_dict())
                    evaluator = HomoGraphEvaluator_Survival(self.config, verbose=False, fold=self.fold, val=True)
                    val_CIndex, val_loss, _, _ = evaluator.eval()
                    # if val_CIndex > best_val_Cindex and epoch > 80:
                    #     best_val_Cindex = val_CIndex
                    #     evaluator.test_data = evaluator.load_data(self.config_data["tvt_root"] + "/" + str(self.fold) + self.config_data["test_path"])
                    #     test_CIndex, test_loss = evaluator.eval()
                    #     print("Test C-Index: {:.4f}, Test Loss: {:.4f}".format(test_CIndex, test_loss))


                    # best_model_path = self.earlystopping(epoch=epoch, val_cindex=val_CIndex, model=self.gnn, ckpt_name=os.path.join(self.config_checkpoint['path'] + '/'+self.config['desc']+ '/' +str(self.fold), 'best_model_epoch{}.pt'.format(epoch+1)))
                    # if best_model_path is not None:
                    #     self.best_model_path = best_model_path

                    sys.stdout.write('\n')
                    sys.stdout.write("Epoch {} | train_loss: {:.4f} val_loss: {:.4f} [Train C-Index: {:.4f} | Val C-Index: {:.4f}]\n".format(epoch+1, train_loss, val_loss, c_index, val_CIndex))
                    sys.stdout.flush()


                    epoch_stats = {
                        "Epoch": epoch + 1,
                        "Train Loss: ": train_loss,
                        "Val Loss: ": val_loss,
                        "Training C-Index": c_index,
                        "Validation C-Index": val_CIndex,
                    }

                    writer.add_scalars('CIndex', {'train': c_index, 'val': val_CIndex}, epoch+1)
                    writer.add_scalars('loss', {'train': train_loss, 'val': val_loss}, epoch+1)



                # State dict of the model including embeddings
                self.checkpoint_manager.write_new_version(
                    self.config,
                    self.gnn.state_dict(),
                    epoch_stats
                )

                # Remove previous checkpoint
                self.checkpoint_manager.remove_old_version()

                # if self.earlystopping.early_stop:
                #     print("Early stopping")
                #     break
        return val_CIndex

    def test(self) -> None:
        print(f"Start testing Homogeneous GNN")
        test_evaluator = HomoGraphEvaluator_Survival(self.config, verbose=False, pkl_path=self.best_model_path, fold=self.fold, val=True, visualize=self.config_eval['visualize'])
        # test_evaluator.test_data = test_evaluator.load_data(self.config_data["tvt_root"] + "/" + str(self.fold) + self.config_data["test_path"])
        test_CIndex, test_loss, vis_info_dict, dic = test_evaluator.eval()
        vis_save_path = os.path.join(self.config_eval['explain_save_path'], str(self.fold))
        os.makedirs(vis_save_path, exist_ok=True)
        if self.config_eval['visualize']:
            with open(os.path.join(vis_save_path, 'vislization.pkl'), 'wb') as file:
                pickle.dump(vis_info_dict, file)
        with open(os.path.join(vis_save_path, 'risk_scores.pkl'), 'wb') as file:
            pickle.dump(dic, file)
        print("Test C-Index: {:.4f}, Test Loss: {:.4f}".format(test_CIndex, test_loss))
        return(test_CIndex)

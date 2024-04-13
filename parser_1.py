from torch import optim, nn
import torch.nn.functional as F

from models import (
    H2GT_HGNN,
    H2GT,
    HEATNet2,
)
from utils import CoxSurvLoss, CrossEntropySurvLoss, NLLSurvLoss, CoxSurvLoss

def parse_optimizer(config_optim, model):
    opt_method = config_optim["opt_method"].lower()
    alpha = config_optim["lr"]
    weight_decay = config_optim["weight_decay"]
    print('lr: ',alpha)
    print('weight decay: ', weight_decay)

    if opt_method == "adagrad":
        optimizer = optim.Adagrad(
            model.parameters(),
            lr=alpha,
            lr_decay=weight_decay,
            weight_decay=weight_decay,
        )
    elif opt_method == "adadelta":
        optimizer = optim.Adadelta(
            model.parameters(),
            lr=alpha,
            weight_decay=weight_decay,
        )
    elif opt_method == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=alpha,
            weight_decay=weight_decay,
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=alpha,
            weight_decay=weight_decay,
            momentum=config_optim["momentum"],
        )
    return optimizer


def parse_gnn_model(config_gnn, visualize=False):
    gnn_name = config_gnn["name"]

    if gnn_name == "H2GT_HGNN":
        return H2GT_HGNN(config=config_gnn)
    elif gnn_name == "H2GT":
        return H2GT(config=config_gnn, visualize=visualize)
    elif gnn_name == "HEAT2":
        n_node_types = config_gnn["n_node_types"]
        node_dict = {str(i): i for i in range(n_node_types)}
        return HEATNet2(
            in_dim=config_gnn["in_dim"],
            hidden_dim=config_gnn["hidden_dim"],
            out_dim=config_gnn["out_dim"],
            n_layers=config_gnn["num_layers"],
            n_heads=config_gnn["n_heads"],
            node_dict=node_dict,
            dropuout=config_gnn["feat_drop"],
            graph_pooling_type=config_gnn["graph_pooling_type"]
        )
    else:
        raise NotImplementedError("This GNN model is not implemented")


def parse_loss(config_train):
    loss_name = config_train["loss"]
    alpha = config_train["alpha"]

    if loss_name == "BCE":
        return nn.BCELoss()
    elif loss_name == "CE":
        return nn.CrossEntropyLoss()
    elif loss_name == "ce_surv":
        return CrossEntropySurvLoss(alpha)
    elif loss_name == "nll_surv":
        return NLLSurvLoss(alpha)
    elif loss_name == 'cox_surv':
        return CoxSurvLoss()
    else:
        raise NotImplementedError("This Loss is not implemented")
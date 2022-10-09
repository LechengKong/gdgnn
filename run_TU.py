import os
import torch
import random
import argparse
import numpy as np
import os.path as osp
import logging
import time
import dgl
import scipy.stats as st
from datetime import datetime
from datasets import HGVerGDGraphDataset

from gnnfree.managers.manager import Manager
from gnnfree.managers.learner import *
from gnnfree.managers.trainer import *
from gnnfree.utils.evaluators import BinaryAccEvaluator, BinaryHNEvaluator
from gnnfree.utils.utils import *
from gnnfree.nn.loss import *
from gnnfree.nn.models.task_predictor import GraphClassifier
from gnnfree.nn.models.GNN import HomogeneousGNN

from sklearn.model_selection import KFold, StratifiedKFold

from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from learners import GraphPredictionLearner
from models.graph_classifier import GDGraphClassifier


def main(params):
    params = parser.parse_args()
    if not osp.exists('./saved_exp'):
        os.mkdir('./saved_exp')

    curtime = datetime.now()
    params.exp_dir = osp.join('./saved_exp', str(curtime))
    os.mkdir(params.exp_dir)

    save_params(osp.join(params.exp_dir,'command'), params)
    params.prefix = params.train_data_set+"_"
    params.model_name = osp.join(params.exp_dir, params.prefix)


    if not torch.cuda.is_available() or params.gpuid==-1:
        params.device = torch.device('cpu')
    else:
        params.device = torch.device('cuda:'+str(params.gpuid))
    
    gnn_model = HomogeneousGNN

    data = dgl.data.TUDataset(params.train_data_set)
    
    params.num_tasks = len(data[0][1])
    params.has_inp_feat = 'feat' in data[0][0].ndata
    # params.has_inp_feat = False
    if params.has_inp_feat:
        params.inp_dim = data[0][0].ndata['feat'].size()[1]
    else:
        params.inp_dim = 1

    evlter = BinaryAccEvaluator('acc_eval')
    eval_metric = 'acc'
    
    c = 0
    for g, l in data:
        # print(l)
        c = max(c, l)
    loss = MultiClassLoss()
    params.num_class = c+1
    
    labels = [val[1].item() for val in data]

    fold = params.fold
    folds = k_fold_ind(labels, fold=fold)
    splits = k_fold2_split(folds, len(labels))
    data_col = []

    def construct_hg_ver_data(data, labels, ind):
        graph = [data[i][0] for i in ind]
        label = [labels[i] for i in ind]
        if not params.has_inp_feat:
            for g in graph:
                g.ndata['feat'] = torch.ones((g.num_nodes(),1))
        d = HGVerGDGraphDataset(graph, label, params)
        return d

    for s in splits:
        train = construct_hg_ver_data(data, labels, s[0])
        test = construct_hg_ver_data(data, labels, s[1])
        val = construct_hg_ver_data(data, labels, s[2])
        data_col.append([train,test,val])

    def run_exp(data, args):
        train, test, val = data
        args.reach_dist = args.num_layers
        gnn = gnn_model(args.num_layers, args.inp_dim, args.emb_dim)
        if args.gdgnn:
            model = GDGraphClassifier(args.num_class, args.emb_dim, gnn, gd_deg=args.gd_deg)
        else:
            model = GraphClassifier(args.num_class, args.emb_dim, gnn)
        model = model.to(args.device)

        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        scheduler = ExponentialLR(optimizer, gamma=0.9)

        train_learner = GraphPredictionLearner('train', train, model, args.batch_size)
        val_learner = GraphPredictionLearner('val', val, model, args.eval_batch_size)
        test_learner = GraphPredictionLearner('test', test, model, args.eval_batch_size)

        trainer = Trainer(evlter, loss, args.num_workers)

        manager = Manager(args.model_name)

        manager.train(train_learner, val_learner, trainer, optimizer, eval_metric, device=args.device, num_epochs=args.num_epochs, scheduler=scheduler)
        manager.load_model(train_learner)

        val_res = manager.eval(val_learner, trainer, device=args.device)
        test_res = manager.eval(test_learner, trainer, device=args.device)

        return val_res, test_res

    hparams={'num_layers':[2,3,4,5]}

    best_res = hyperparameter_grid_search(hparams, data_col, multi_data_average_exp, params, eval_metric, evlter, exp_arg=run_exp)
    
    write_res_to_file(osp.join(params.exp_dir, 'result'), params.train_data_set, eval_metric, best_res['test_mean'], best_res['val_mean'], best_res['test_std'], best_res['val_std'], best_res['params'], json.dumps(best_res))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gnn')

    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--train_data_set", type=str, default="DD")

    parser.add_argument("--emb_dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--JK", type=str, default='sum')
    parser.add_argument("--hidden_dim", type=int, default=32)

    parser.add_argument("--dropout",type=float,default=0)

    parser.add_argument("--lr",type=float,default=0.001)
    parser.add_argument("--l2",type=float,default=0)
    parser.add_argument("--batch_size",type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)

    parser.add_argument("--num_workers",type=int, default=20)
    parser.add_argument("--save_every",type=int, default=1)

    parser.add_argument("--num_epochs", type=int, default=50)

    parser.add_argument("--reach_dist", type=int, default=3)

    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--fold", type=int, default=10)

    parser.add_argument('--gdgnn', type=bool, default=True)
    parser.add_argument('--gd_deg', type=bool, default=True)

    parser.add_argument("--psearch", type=bool, default=False)



    params = parser.parse_args()
    set_random_seed(1)
    main(params)
    


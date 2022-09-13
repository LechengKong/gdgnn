import os
import torch
import argparse
import os.path as osp
from datetime import datetime
from datasets import HGVerGDGraphDataset

from gnnfree.managers.manager import Manager
from gnnfree.managers.learner import *
from gnnfree.managers.trainer import *
from gnnfree.utils.evaluators import BinaryAccEvaluator, BinaryHNEvaluator
from gnnfree.utils.utils import *
from gnnfree.nn.loss import *
from gnnfree.nn.models.task_predictor import GraphClassifier

from ogb.graphproppred import DglGraphPropPredDataset

from sklearn.model_selection import KFold, StratifiedKFold

from torch.optim import Adam
from learners import GraphPredictionLearner
from models.GNN import MOLGNNVN, MOLGNN
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
    
    gnn_model = MOLGNNVN if params.train_data_set=='ogbg-molpcba' else MOLGNN

    params.data_folder_path = osp.join(params.data_path, params.train_data_set)

    data = DglGraphPropPredDataset(name=params.train_data_set, root=params.data_path)
    split_idx = data.get_idx_split()
    train_ind = split_idx['train']
    test_ind = split_idx['test']
    val_ind = split_idx['valid']
    train_graph = [data[i][0] for i in train_ind]
    test_graph = [data[i][0] for i in test_ind]
    val_graph = [data[i][0] for i in val_ind]

    train_label = [data[i][1].numpy() for i in train_ind]
    test_label = [data[i][1].numpy() for i in test_ind]
    val_label = [data[i][1].numpy() for i in val_ind]
    
    params.num_class = len(train_label[0])
    params.atom_dim = train_graph[0].ndata['feat']
    params.bond_dim = train_graph[0].edata['feat']
    params.inp_dim = 0

    evlter = BinaryHNEvaluator('acc_eval')
    if params.train_data_set=='ogbg-molpcba':
        eval_metric = 'apr'
    else:
        eval_metric = 'auc'

    def prepare_eval_data(res, data):
        return [res, data.labels]
    
    loss = BinaryLoss()

    def construct_hg_ver_data(data, label_t):
        graph = data
        for g in graph:
            g.ndata['atom_feat'] = g.ndata['feat']
            g.edata['bond_feat'] = g.edata['feat']
            g.ndata['feat'] = torch.zeros([g.num_nodes(),0])
            g.edata['feat'] = torch.zeros([g.num_edges(),0])
        # d = HGNeighborGraphDataset(graph, label_t, params)
        d = HGVerGDGraphDataset(graph, label_t, params)
        return d

    data_col = [construct_hg_ver_data(train_graph, train_label), construct_hg_ver_data(test_graph, test_label), construct_hg_ver_data(val_graph, val_label)]

    def run_exp(data, args):
        train, test, val = data
        gnn = gnn_model(args.num_layers, args.inp_dim, args.emb_dim, args.mol_emb_dim, params.inp_dim, drop_ratio=args.dropout)
        if args.gdgnn:
            model = GDGraphClassifier(args.num_class, args.emb_dim, gnn, gd_deg=args.gd_deg)
        else:
            model = GraphClassifier(args.num_class, args.emb_dim, gnn)
        model = model.to(args.device)

        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

        train_learner = GraphPredictionLearner('train', train, model, loss, optimizer, args.batch_size)
        val_learner = GraphPredictionLearner('val', val, model, None, None, args.eval_batch_size)
        test_learner = GraphPredictionLearner('test', test, model, None, None, args.eval_batch_size)

        trainer = Trainer(evlter, prepare_eval_data, args.num_workers)

        manager = Manager(args.model_name)

        manager.train(train_learner, val_learner, trainer, optimizer, eval_metric, device=args.device, num_epochs=args.num_epochs)

        manager.load_model(train_learner)

        val_res = manager.eval(val_learner, trainer, device=args.device)
        test_res = manager.eval(test_learner, trainer, device=args.device)

        return val_res, test_res

    hparams={'num_layers':[2,3,4,5]}

    best_res = hyperparameter_grid_search(hparams, data_col, run_exp, params, eval_metric, evlter)
    
    write_res_to_file(osp.join(params.exp_dir, 'result'), params.train_data_set, eval_metric, best_res['test_mean'], best_res['val_mean'], best_res['test_std'], best_res['val_std'], best_res['params'], json.dumps(best_res))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gnn')

    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--train_data_set", type=str, default="ogbg-molhiv")

    parser.add_argument("--emb_dim", type=int, default=32)
    parser.add_argument("--mol_emb_dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--JK", type=str, default='last')
    parser.add_argument("--hidden_dim", type=int, default=32)

    parser.add_argument("--dropout",type=float,default=0)

    parser.add_argument("--lr",type=float,default=0.0001)
    parser.add_argument("--l2",type=float,default=0)
    parser.add_argument("--batch_size",type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)

    parser.add_argument("--num_workers",type=int, default=20)
    parser.add_argument("--save_every",type=int, default=1)

    parser.add_argument("--num_epochs", type=int, default=100)

    parser.add_argument("--reach_dist", type=int, default=3)

    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--fold", type=int, default=10)

    parser.add_argument('--gdgnn', type=bool, default=False)
    parser.add_argument('--gd_deg', type=bool, default=False)

    parser.add_argument("--psearch", type=bool, default=True)



    params = parser.parse_args()
    set_random_seed(1)
    main(params)
    


import os
import torch
import argparse
import os.path as osp
from datetime import datetime
from datasets import HGHorGDDataset, HGVerGDDataset, HGVerGDGraphDataset, HGVerGDNodeDataset

from gnnfree.managers.manager import Manager
from gnnfree.managers.learner import *
from gnnfree.managers.trainer import *
from gnnfree.utils.evaluators import BinaryAccEvaluator, BinaryHNEvaluator, Evaluator
from gnnfree.utils.graph import construct_graph_from_edges
from gnnfree.utils.utils import *
from gnnfree.nn.loss import *
from gnnfree.nn.models.GNN import HomogeneousGNN

from torch.optim import Adam
from learners import NodePredictionLearner, PrecomputeNENCLearner
from models.node_classifier import GDNodeClassifier

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

    params.data_folder_path = osp.join(params.data_path, params.train_data_set)
    e2i = {}
    with open(osp.join(params.data_folder_path, 'graph.txt')) as f:
        edges = [line.split(',') for line in f.read().split('\n')]
    edges = np.array(edges).astype(int)
    nid=0
    c_edges = []
    for e in edges:
        if e[0] not in e2i:
            e2i[e[0]]=nid
            nid+=1
        if e[1] not in e2i:
            e2i[e[1]]=nid
            nid+=1
        c_edges.append([e2i[e[0]], e2i[e[1]]])
    edges = np.array(c_edges)

    g = construct_graph_from_edges(edges[:,0], edges[:,1], n_entities=np.max(edges)+1, inverse_edge=True)
    g.ndata['feat'] = torch.ones((g.num_nodes(),1))

    with open(osp.join(params.data_folder_path, 'label.txt')) as f:
        label = [line.split(',') for line in f.read().split('\n')]
    label = np.array(label).astype(int)

    for i in range(len(label)):
        label[i,0] = e2i[label[i,0]]

    params.num_tasks = np.max(label[:, 1])+1
    params.inp_dim = 1
    params.num_class = params.num_tasks

    evlter = BinaryAccEvaluator('acc_eval')
    eval_metric = 'acc'

    def prepare_eval_data(res, data):
        return [res, data.labels]
    
    loss = MultiClassLoss()

    def construct_hg_ver_data(data, ind):
        node = [data[i,0] for i in ind]
        label = [data[i,1] for i in ind]
        d = HGVerGDNodeDataset(g, node, label, params)
        return d

    fold = 10
    folds = k_fold_ind(label[:,1], fold=fold)

    test_arr = np.zeros(len(label), dtype=bool)
    test_arr[folds[0]]=1
    val_arr = np.zeros(len(label), dtype=bool)
    val_arr[folds[int((1)%fold)]]=1
    train_arr = np.logical_not(np.logical_or(test_arr, val_arr))
    train_ind = train_arr.nonzero()[0]
    test_ind = test_arr.nonzero()[0]
    val_ind = val_arr.nonzero()[0]
    train = construct_hg_ver_data(label, train_ind)
    test = construct_hg_ver_data(label, test_ind)
    val = construct_hg_ver_data(label, val_ind)
    
    def run_exp(data, args):
        train, test, val = data
        args.reach_dist = args.num_layers
        gnn = gnn_model(args.num_layers, args.inp_dim, args.emb_dim, drop_ratio=args.dropout)
        model = GDNodeClassifier(args.num_class, args.emb_dim, gnn)
        model = model.to(args.device)

        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

        train_learner = NodePredictionLearner('train', train, model, loss, optimizer, args.batch_size)
        val_learner = PrecomputeNENCLearner('val', val, model, None, None, args.eval_batch_size)
        test_learner = PrecomputeNENCLearner('test', test, model, None, None, args.eval_batch_size)

        trainer = Trainer(evlter, prepare_eval_data, args.num_workers)

        manager = Manager(args.model_name)

        manager.train(train_learner, val_learner, trainer, optimizer, eval_metric, device=args.device, num_epochs=args.num_epochs)

        manager.load_model(train_learner)

        val_res = manager.eval(val_learner, trainer, device=args.device)
        test_res = manager.eval(test_learner, trainer, device=args.device)

        return val_res, test_res

    hparams={'num_layers':[2,3,4,5]} if params.psearch else {'num_layers':[params.num_layers]}

    best_res = hyperparameter_grid_search(hparams, [train, test, val], run_exp, params, eval_metric, evlter)
    
    write_res_to_file(osp.join(params.exp_dir, 'result'), params.train_data_set, eval_metric, best_res['test_mean'], best_res['val_mean'], best_res['test_std'], best_res['val_std'], best_res['params'], json.dumps(best_res))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gnn')

    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--train_data_set", type=str, default="brazil_airport")

    parser.add_argument("--emb_dim", type=int, default=32)
    parser.add_argument("--mol_emb_dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=2)
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

    parser.add_argument("--gd_type", type=str, default='HorGD')

    parser.add_argument('--gdgnn', type=bool, default=False)
    parser.add_argument('--gd_deg', type=bool, default=False)

    parser.add_argument("--psearch", type=bool, default=False)



    params = parser.parse_args()
    set_random_seed(1)
    main(params)
    


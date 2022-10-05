import os
import torch
import argparse
import os.path as osp
from datetime import datetime
from datasets import HGHorGDDataset, HGHorGDInterDataset, HGHorSampleGDDataset, HGVerGDDataset, HGVerGDGraphDataset, HGVerGDInterDataset, HGVerGDSampleDataset, SimpleSampleClass

from gnnfree.managers.manager import Manager
from gnnfree.managers.learner import *
from gnnfree.managers.trainer import *
from gnnfree.utils.evaluators import BinaryAccEvaluator, BinaryHNEvaluator, Evaluator, HNEvaluator
from gnnfree.utils.graph import construct_graph_from_edges
from gnnfree.utils.io import save_load_torch_data
from gnnfree.utils.utils import *
from gnnfree.nn.loss import *
from gnnfree.nn.models.task_predictor import LinkPredictor
from gnnfree.nn.models.GNN import HomogeneousGNN

from torch.optim import Adam
from learners import LinkPredictionLearner, PrecomputeNELPLearner
from models.link_predictor import GDLinkPredictor

from scipy.sparse import tril, coo_matrix
from ogb.linkproppred import DglLinkPropPredDataset

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

    if params.gd_type == 'HorGD':
        TrainSet = HGHorGDDataset
    elif params.gd_type == 'VerGD':
        TrainSet = HGVerGDDataset
    elif params.gd_type == '':
        TrainSet = None

    params.data_folder_path = osp.join(params.data_path, params.train_data_set)
    dgl_data = DglLinkPropPredDataset(name=params.train_data_set, root=params.data_path)
    split_edge = dgl_data.get_edge_split()
    train_num_entities = dgl_data.graph[0].num_nodes()
    adj_mat = dgl_data.graph[0].adj(scipy_fmt='coo')
    
    params.train_data_set = '_'.join(params.train_data_set.split('-'))
    params.data_folder_path = osp.join(params.data_path, params.train_data_set)
    row = split_edge['train']['edge'][:,0].numpy()
    col = split_edge['train']['edge'][:,1].numpy()
    train_edges = np.zeros((len(row),2), dtype=int)
    train_edges[:,0] = row
    train_edges[:,1] = col
    inverse_row = np.concatenate([row, split_edge['test']['edge'][:,0].numpy(), split_edge['valid']['edge'][:,0].numpy()])
    inverse_col = np.concatenate([col, split_edge['test']['edge'][:,1].numpy(), split_edge['valid']['edge'][:,1].numpy()])

    inversed_row = np.concatenate([inverse_row,inverse_col])
    inversed_col = np.concatenate([inverse_col,inverse_row])
    data = np.ones(len(inversed_row))
    train_adj = coo_matrix((data, (inversed_row, inversed_col)), shape=adj_mat.shape)
    train_adj = train_adj.tocsr()
    
    dtst = SimpleSampleClass(len(split_edge['train']['edge']), train_adj, train_num_entities)
    neg_data, fold = save_load_torch_data(params.data_folder_path, dtst, data_fold=50, data_name='neg_links')
    train_neg = np.concatenate(neg_data)
    converted_triplets = {"train":train_edges,"train_neg":train_neg, "test":split_edge['test']['edge'], "test_neg":split_edge['test']['edge_neg'], "valid":split_edge['valid']['edge'], "valid_neg":split_edge['valid']['edge_neg']}
    for k in converted_triplets:
        np_arr = np.zeros((len(converted_triplets[k]), 3), dtype=int)
        np_arr[:,0] = converted_triplets[k][:,0]
        np_arr[:,2] = converted_triplets[k][:,1]
        converted_triplets[k] = np_arr

    g = construct_graph_from_edges(converted_triplets['train'].T[0], converted_triplets['train'].T[2], n_entities=train_num_entities, inverse_edge=True)
    if params.train_data_set == 'ogbl_ppa':
        g.ndata['feat'] = dgl_data.graph[0].ndata['feat'].to(torch.float)
        params.inp_dim = g.ndata['feat'].size()[1]
        hn = 100
        train_sample_size = 10000000
        if params.gd_type == 'HorGD':
            # TrainSet = HGHorSampleGDDataset
            TrainSet = HGHorGDInterDataset
        elif params.gd_type == 'VerGD':
            # TrainSet = HGVerGDSampleDataset
            TrainSet = HGVerGDInterDataset
        elif params.gd_type == '':
            TrainSet = None
    else:
        g.ndata['feat'] = torch.ones([g.num_nodes(), 1], dtype=torch.float32)
        params.inp_dim = 1
        hn = 50
        train_sample_size = None
        if params.gd_type == 'HorGD':
            # TrainSet = HGHorSampleGDDataset
            TrainSet = HGHorGDInterDataset
        elif params.gd_type == 'VerGD':
            # TrainSet = HGVerGDSampleDataset
            TrainSet = HGVerGDInterDataset
        elif params.gd_type == '':
            TrainSet = None

    evlter = HNEvaluator('acc_eval', hn=hn)
    eval_metric = 'h'+str(hn)

    def prepare_eval_data(res, data):
        return [res, data.labels]
    
    loss = BinaryLoss()

    train_edge = np.concatenate([converted_triplets['train'], converted_triplets['train_neg']], axis=0)
    train_label = np.zeros(len(train_edge), dtype=int)
    train_label[:len(converted_triplets['train'])]=1
    train = TrainSet(g, train_edge, train_label, train_num_entities, params)
    params.train_edges = len(train)

    test_edge = np.concatenate([converted_triplets['test'], converted_triplets['test_neg']], axis=0)
    test_label = np.zeros(len(test_edge), dtype=int)
    test_label[:len(converted_triplets['test'])]=1
    valid_edge = np.concatenate([converted_triplets['valid'], converted_triplets['valid_neg']], axis=0)
    valid_label = np.zeros(len(valid_edge), dtype=int)
    valid_label[:len(converted_triplets['valid'])]=1
    test = TrainSet(g, test_edge, test_label, train_num_entities, params, mode='valid')
    val = TrainSet(g, valid_edge, valid_label, train_num_entities, params, mode='valid')
    params.test_size = len(test)
    params.val_size = len(val)

    print(f'Training graph has {params.train_edges} edges, validataion set has {params.val_size} query edges, test set has {params.test_size} query edges')
    
    def run_exp(data, args):
        train, test, val = data
        args.reach_dist = args.num_layers
        gnn = gnn_model(args.num_layers, args.inp_dim, args.emb_dim, drop_ratio=args.dropout)
        model = GDLinkPredictor(args.emb_dim, gnn, [args.gd_type, 'dist'])
        model = model.to(args.device)

        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

        train_learner = LinkPredictionLearner('train', train, model, loss, optimizer, args.batch_size)
        val_learner = PrecomputeNELPLearner('val', val, model, None, None, args.eval_batch_size)
        test_learner = PrecomputeNELPLearner('test', test, model, None, None, args.eval_batch_size)

        trainer = Trainer(evlter, prepare_eval_data, args.num_workers, train_sample_size=train_sample_size)

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
    parser.add_argument("--train_data_set", type=str, default="ogbl-collab")

    parser.add_argument("--emb_dim", type=int, default=32)
    parser.add_argument("--mol_emb_dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--JK", type=str, default='last')
    parser.add_argument("--hidden_dim", type=int, default=32)

    parser.add_argument("--dropout",type=float,default=0)

    parser.add_argument("--lr",type=float,default=0.0001)
    parser.add_argument("--l2",type=float,default=0)
    parser.add_argument("--batch_size",type=int, default=4096)
    parser.add_argument("--eval_batch_size", type=int, default=512)

    parser.add_argument("--num_workers",type=int, default=32)
    parser.add_argument("--save_every",type=int, default=1)

    parser.add_argument("--num_epochs", type=int, default=5)

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
    


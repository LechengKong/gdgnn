from datetime import datetime
import torch
import random
import argparse
import numpy as np
import os.path as osp
import os
import logging
import dgl
import scipy.io as io


from torch.optim import Adam
from datasets import KGHorGDRankingDataset, KGVerGDRankingDataset
from gnnfree.managers.manager import Manager
from gnnfree.managers.trainer import FilteredMaxTrainer, MaxTrainer
from gnnfree.nn.loss import NegLogLoss
from gnnfree.utils.evaluators import VarSizeRankingEvaluator
from gnnfree.utils.graph import construct_graph_from_edges
from gnnfree.utils.utils import save_params
from graph_utils import sample_filtered_neg_head, sample_filtered_neg_tail
from learners import LinkFixedSizeRankingLearner, PrecomputeNELPLearner
from models.link_predictor import LinkPredictor


from gnnfree.utils.io import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gnn')

    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--train_data_set", type=str, default="WN18RR")
    parser.add_argument("--test_data_set", type=str, default="WN18RR")
    parser.add_argument("--data_type", type=str, default='name')
    parser.add_argument("--val_ind", type=bool, default=False)
    parser.add_argument("--test_filtered", type=bool, default=False)
    parser.add_argument("--val_filtered", type=bool, default=False)

    parser.add_argument("--emb_dim", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--JK", type=str, default='last')
    parser.add_argument("--num_bases", type=int, default=4)

    parser.add_argument("--dropout",type=float,default=0)

    parser.add_argument("--optimizer",type=str, default='Adam')
    parser.add_argument("--lr",type=float,default=0.001)
    parser.add_argument("--l2",type=float,default=0)
    parser.add_argument("--batch_size",type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--train_neg_samples", type=int, default=50)
    parser.add_argument("--eval_neg_samples", type=int, default=50)

    parser.add_argument("--num_workers",type=int, default=20)
    parser.add_argument("--save_every",type=int, default=1)

    parser.add_argument("--val_hn",type=int, default=50)

    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--eval", type=bool, default=False)
    parser.add_argument("--eval_every", type=int, default=1)

    parser.add_argument("--reach_dist", type=int, default=4)

    parser.add_argument("--gd_type", type=str, default='hor')
    parser.add_argument("--gnn", type=str, default='rgcn')
    parser.add_argument("--use_dist", type=bool, default=False)
    parser.add_argument("--hidden_unit", type=int, default=32)
    parser.add_argument("--use_gd_repr", type=bool, default=True)
    parser.add_argument("--gd_max", type=int, default=10)
    parser.add_argument("--use_gd_deg", type=bool, default=False)

    parser.add_argument("--gpuid", type=int, default=0)

    params = parser.parse_args()
    if not osp.exists('./saved_exp'):
        os.mkdir('./saved_exp')

    curtime = datetime.now()
    params.exp_dir = osp.join('./saved_exp', str(curtime))
    os.mkdir(params.exp_dir)

    save_params(osp.join(params.exp_dir,'command'), params)
    params.prefix = params.train_data_set+"_"+params.gd_type+"_"  + ("dist_" if params.use_dist else "")
    params.model_name = osp.join(params.exp_dir, params.prefix)

    print(params.prefix)
    torch.manual_seed(50)
    random.seed(50)
    np.random.seed(50)

    if not torch.cuda.is_available() or params.gpuid==-1:
        params.device = torch.device('cpu')
    else:
        params.device = torch.device('cuda:'+str(params.gpuid))

    dgl_model = LinkPredictor
    learner_type = LinkFixedSizeRankingLearner
    test_learner_type = PrecomputeNELPLearner
    evaluator_type = VarSizeRankingEvaluator

    if params.gd_type == 'hor':
        TrainSet = KGHorGDRankingDataset
    else:
        TrainSet = KGVerGDRankingDataset

    logging.basicConfig(level=logging.INFO)

    database_path = osp.join(params.data_path, params.train_data_set)
    params.data_folder_path = osp.join(params.data_path, params.train_data_set)
    files = {}
    use_data = ["train", "test", "valid"]
    for f in use_data:
        files[f] = osp.join(database_path,f'{f}.txt')
    trans_adj_list, trans_converted_triplets, trans_entity2id, trans_relation2id, trans_id2entity, trans_id2relation = read_knowledge_graph(files)
    # print(relation2id)
    trans_num_entities = len(trans_entity2id)
    trans_num_rel = len(trans_relation2id)

    test_database_path = osp.join(params.data_path, params.test_data_set)
    files = {}
    for f in use_data:
        files[f] = osp.join(test_database_path,f'{f}.txt')

    ind_adj_list, ind_converted_triplets, ind_entity2id, ind_relation2id_test, ind_id2entity, ind_id2relation = read_knowledge_graph(files, relation2id=trans_relation2id)
    ind_num_entities = len(ind_entity2id)
    ind_num_rel = len(ind_relation2id_test)
    params.num_rels = trans_num_rel
    params.aug_num_rels = trans_num_rel*2
    print(f'Train Dataset {params.train_data_set} has {trans_num_entities} entities and {trans_num_rel} relations')
    print(f'Test Dataset {params.test_data_set} has {ind_num_entities} entities and {ind_num_rel} relations')

    train_triplets = trans_converted_triplets
    train_adj_list = trans_adj_list
    train_num_rel = trans_num_rel
    train_num_entities = trans_num_entities
    train_graph = construct_graph_from_edges(train_triplets['train'].T[0], train_triplets['train'].T[2], train_num_entities, inverse_edge=True, edge_type=train_triplets['train'].T[1], num_rels=train_num_rel)
    train_graph.ndata['feat'] = torch.ones((train_graph.num_nodes(),1))

    test_triplets = ind_converted_triplets
    test_adj_list = ind_adj_list
    test_num_rel = ind_num_rel
    test_num_entities = ind_num_entities
    test_graph = construct_graph_from_edges(test_triplets['train'].T[0], test_triplets['train'].T[2], test_num_entities, inverse_edge=True, edge_type=test_triplets['train'].T[1], num_rels=test_num_rel)
    test_graph.ndata['feat'] = torch.ones((test_graph.num_nodes(),1))

    if params.val_ind:
        val_triplets = ind_converted_triplets
        val_num_entities = ind_num_entities
        val_num_rel = ind_num_rel
        val_adj_list = ind_adj_list
        val_graph = test_graph
    else:
        val_triplets = trans_converted_triplets
        val_num_entities = trans_num_entities
        val_num_rel = trans_num_rel
        val_adj_list = trans_adj_list
        val_graph = train_graph
    params.inp_dim = 1

    model = dgl_model(params).to(device=params.device)

    train = TrainSet(train_graph, train_triplets['train'], params, train_adj_list, mode='train', neg_link_per_sample=params.train_neg_samples)

    train_learner = learner_type('train', train, model, NegLogLoss(params.train_neg_samples), Adam, params.batch_size)
    train_learner.setup_optimizer([ {'lr':params.lr, 'weight_decay':params.l2}])

    params.train_edges = len(train)
    
    if params.test_filtered:
        test_head = TrainSet(test_graph, test_triplets['test'], params, test_adj_list, mode='eval', sample_method=sample_filtered_neg_tail)
        test_tail = TrainSet(test_graph, test_triplets['test'], params, test_adj_list, mode='eval', sample_method=sample_filtered_neg_head)
        params.test_size = len(test_head)
        test_learner = [test_learner_type('test_head_learner', test_head, model, None, None, params.eval_batch_size), test_learner_type('test_tail_learner', test_tail, model, None, None, params.eval_batch_size)]
        test_trainer = FilteredMaxTrainer(params)
    else:
        test = TrainSet(test_graph, test_triplets['test'], params, test_adj_list, mode='eval', neg_link_per_sample=params.eval_neg_samples)
        params.test_size = len(test)
        test_learner = test_learner_type('test_learner', test, model, None, None, params.eval_batch_size)
        test_trainer = MaxTrainer(params)
    if params.val_filtered:
        val_head = TrainSet(val_graph, val_triplets['valid'], params, val_adj_list, mode='eval', sample_method=sample_filtered_neg_tail)
        val_tail = TrainSet(val_graph, val_triplets['valid'], params, val_adj_list, mode='eval', sample_method=sample_filtered_neg_head)
        params.val_size = len(val_head)
        val_learner = [test_learner_type('val_head_learner', val_head, model, None, None, params.eval_batch_size), test_learner_type('val_tail_learner', val_tail, model, None, None, params.eval_batch_size)]
        val_trainer = FilteredMaxTrainer(params)
    else:
        val = TrainSet(val_graph, val_triplets['valid'], params, val_adj_list, mode='eval', neg_link_per_sample=params.eval_neg_samples)
        params.val_size = len(val)
        val_learner = test_learner_type('val_learner', val, model, None, None, params.eval_batch_size)
        val_trainer = MaxTrainer(params)
    print(f'Training graph has {train_graph.num_edges()} edges, validataion set has {params.val_size} query edges')
    print(f'Test graph has {test_graph.num_edges()} edges, and {params.test_size} query edges.')

    manager = Manager(params.model_name)
    eval_metric = 'mrr'
    if not params.eval:
        manager.train([train_learner, val_learner], val_trainer, VarSizeRankingEvaluator('varsize_eval', num_workers=params.num_workers), eval_metric, device=params.device, num_epochs=params.num_epochs)
    # for l in test_learner:
    if osp.exists(params.model_name+"best.pth"):
        train_learner.load_model(params.model_name+"best.pth")
    print(manager.eval(val_learner, val_trainer, VarSizeRankingEvaluator('varsize_eval', num_workers=params.num_workers), params.device))
    print(manager.eval(test_learner, test_trainer, VarSizeRankingEvaluator('varsize_eval', num_workers=params.num_workers), params.device))

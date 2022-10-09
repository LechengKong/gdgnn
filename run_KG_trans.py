import os
import torch
import argparse
import os.path as osp
from datetime import datetime
from datasets import HGHorGDDataset, HGVerGDDataset, HGVerGDGraphDataset, KGHorGDFileredDataset, KGHorGDNegSampleDataset, KGVerGDFileredDataset, KGVerGDNegSampleDataset

from gnnfree.managers.manager import Manager
from gnnfree.managers.learner import *
from gnnfree.managers.trainer import *
from gnnfree.utils.evaluators import BinaryAccEvaluator, BinaryHNEvaluator, Evaluator, VarSizeRankingEvaluator
from gnnfree.utils.graph import construct_graph_from_edges
from gnnfree.utils.io import read_knowledge_graph
from gnnfree.utils.utils import *
from gnnfree.nn.loss import *
from gnnfree.nn.models.task_predictor import LinkPredictor
from gnnfree.nn.models.GNN import RGCN

from torch.optim import Adam
from learners import KGPrecomputeNELPLearner, LinkFixedSizeRankingLearner, LinkPredictionLearner, PrecomputeNELPLearner
from models.link_predictor import GDLinkPredictor, RelLinkPredictor

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

    if params.gd_type == 'HorGD':
        TrainSet = KGHorGDNegSampleDataset
        TestSet = KGHorGDFileredDataset
    elif params.gd_type == 'VerGD':
        TrainSet = KGVerGDNegSampleDataset
        TestSet = KGVerGDFileredDataset
    elif params.gd_type == '':
        TrainSet = None

    database_path = osp.join(params.data_path, params.train_data_set)
    params.data_folder_path = osp.join(params.data_path, params.train_data_set)
    files = {}
    use_data = ["train", "test", "valid"]
    for f in use_data:
        files[f] = osp.join(database_path,f'{f}.txt')
    trans_adj_list, trans_triplets, trans_entity2id, trans_relation2id, trans_id2entity, trans_id2relation = read_knowledge_graph(files)

    trans_num_entities = len(trans_entity2id)
    trans_num_rel = len(trans_relation2id)

    params.num_rels = trans_num_rel
    params.aug_num_rels = trans_num_rel*2
    

    trans_graph = construct_graph_from_edges(trans_triplets['train'].T[0], trans_triplets['train'].T[2], n_entities=trans_num_entities, inverse_edge=True, edge_type=trans_triplets['train'].T[1], num_rels=trans_num_rel)

    trans_graph.ndata['feat'] = torch.ones([trans_graph.num_nodes(), 1], dtype=torch.float32)
    params.inp_dim = 1

    evlter = VarSizeRankingEvaluator('varsizeeval', params.num_workers)
    eval_metric = 'mrr'
    
    # loss = NegLogLoss(params.neg_sample)
    loss = FirstPosNegLoss(params.neg_sample)

    trans_reverse_adj_list = [adj.tocsc().T for adj in trans_adj_list]

    train = TrainSet(trans_graph, trans_triplets['train'], params, trans_adj_list, mode='train', neg_link_per_sample=params.neg_sample, reverse_dir_adj=trans_reverse_adj_list)
    fix_head_test = TestSet(trans_graph, trans_triplets['test'], params, trans_adj_list, mode='eval')
    fix_tail_test = TestSet(trans_graph, trans_triplets['test'], params, trans_reverse_adj_list, mode='eval', head_first=False)

    fix_head_val = TestSet(trans_graph, trans_triplets['valid'], params, trans_adj_list, mode='eval')
    fix_tail_val = TestSet(trans_graph, trans_triplets['valid'], params, trans_reverse_adj_list, mode='eval', head_first=False)

    params.train_edges = len(train)

    params.test_size = len(fix_head_test)
    params.val_size = len(fix_head_val)

    print(f'Training graph has {params.train_edges} edges, validataion set has {params.val_size} query edges, test set has {params.test_size} query edges')
    
    def run_exp(data, args):
        train, fix_head_test, fix_tail_test, fix_head_val, fix_tail_val = data
        args.reach_dist = args.num_layers
        gnn = RGCN(args.num_layers, args.aug_num_rels, args.inp_dim, args.emb_dim, drop_ratio=args.dropout)
        model = RelLinkPredictor(args.emb_dim, args.num_rels, gnn, [args.gd_type, 'dist'])
        model = model.to(args.device)

        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

        train_learner = LinkFixedSizeRankingLearner('train', train, model, args.batch_size)
        val_learners = [KGPrecomputeNELPLearner('head_val', fix_head_val, model, args.eval_batch_size), KGPrecomputeNELPLearner('tail_val', fix_tail_val, model, args.eval_batch_size)]
        test_learners = [KGPrecomputeNELPLearner('head_test', fix_head_test, model, args.eval_batch_size), KGPrecomputeNELPLearner('tail_val', fix_tail_test, model, args.eval_batch_size)]

        trainer = FilteredTrainer(evlter, loss, args.num_workers)

        manager = Manager(args.model_name)

        manager.train(train_learner, val_learners, trainer, optimizer, eval_metric, device=args.device, num_epochs=args.num_epochs)

        manager.load_model(train_learner)

        val_res = manager.eval(val_learners, trainer, device=args.device)
        test_res = manager.eval(test_learners, trainer, device=args.device)

        return val_res, test_res

    hparams={'num_layers':[2,3,4,5]} if params.psearch else {'num_layers':[params.num_layers]}

    best_res = hyperparameter_grid_search(hparams, [train, fix_head_test, fix_tail_test, fix_head_val, fix_tail_val], run_exp, params, eval_metric, evlter)
    
    write_res_to_file(osp.join(params.exp_dir, 'result'), params.train_data_set, eval_metric, best_res['test_mean'], best_res['val_mean'], best_res['test_std'], best_res['val_std'], best_res['params'], json.dumps(best_res))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gnn')

    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--train_data_set", type=str, default="FB15K237")
    parser.add_argument("--val_method", type=str, default="trans")

    parser.add_argument("--emb_dim", type=int, default=32)
    parser.add_argument("--mol_emb_dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--JK", type=str, default='last')
    parser.add_argument("--hidden_dim", type=int, default=32)

    parser.add_argument("--dropout",type=float,default=0)

    parser.add_argument("--lr",type=float,default=0.0001)
    parser.add_argument("--l2",type=float,default=0)
    parser.add_argument("--batch_size",type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--neg_sample", type=int, default=50)

    parser.add_argument("--num_workers",type=int, default=20)
    parser.add_argument("--save_every",type=int, default=1)

    parser.add_argument("--num_epochs", type=int, default=10)

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
    


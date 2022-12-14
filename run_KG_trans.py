import os
import torch
import argparse
import os.path as osp
from datetime import datetime
from datasets import (
    KGHorGDFileredDataset,
    KGHorGDNegSampleDataset,
    KGVerGDFileredDataset,
    KGVerGDNegSampleDataset,
)

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from gnnfree.utils.evaluators import VarSizeRankingEvaluator
from gnnfree.utils.graph import construct_graph_from_edges
from gnnfree.utils.io import read_knowledge_graph
from gnnfree.utils.utils import save_params, set_random_seed
from gnnfree.nn.loss import FirstPosNegLoss

from gnnfree.nn.models.GNN import RGCN

from models.link_predictor import GDLinkPredictor

from lightning_template import KGFilteredLinkPred


def main(params):
    # -------------------- Prepare Exp -------------------- #
    params = parser.parse_args()
    if not osp.exists("./saved_exp"):
        os.mkdir("./saved_exp")

    curtime = datetime.now()
    params.exp_dir = osp.join("./saved_exp", str(curtime))
    os.mkdir(params.exp_dir)

    save_params(osp.join(params.exp_dir, "command"), params)
    params.prefix = params.train_data_set + "_"
    params.model_name = osp.join(params.exp_dir, params.prefix)

    # -------------------- Data Preparation ---------------- #
    if params.gd_type == "HorGD":
        TrainSet = KGHorGDNegSampleDataset
        TestSet = KGHorGDFileredDataset
    elif params.gd_type == "VerGD":
        TrainSet = KGVerGDNegSampleDataset
        TestSet = KGVerGDFileredDataset
    elif params.gd_type == "":
        TrainSet = None

    database_path = osp.join(params.data_path, params.train_data_set)
    params.data_folder_path = osp.join(params.data_path, params.train_data_set)
    files = {}
    use_data = ["train", "test", "valid"]
    for f in use_data:
        files[f] = osp.join(database_path, f"{f}.txt")
    (
        trans_adj_list,
        trans_triplets,
        trans_entity2id,
        trans_relation2id,
        trans_id2entity,
        trans_id2relation,
    ) = read_knowledge_graph(files)

    trans_num_entities = len(trans_entity2id)
    trans_num_rel = len(trans_relation2id)

    params.num_rels = trans_num_rel
    params.aug_num_rels = trans_num_rel * 2

    trans_graph = construct_graph_from_edges(
        trans_triplets["train"].T[0],
        trans_triplets["train"].T[2],
        n_entities=trans_num_entities,
        inverse_edge=True,
        edge_type=trans_triplets["train"].T[1],
        num_rels=trans_num_rel,
    )

    trans_graph.ndata["feat"] = torch.ones(
        [trans_graph.num_nodes(), 1], dtype=torch.float32
    )
    params.inp_dim = 1

    trans_reverse_adj_list = [adj.tocsc().T for adj in trans_adj_list]

    train = TrainSet(
        trans_graph,
        trans_triplets["train"],
        params,
        trans_adj_list,
        mode="train",
        neg_link_per_sample=params.neg_sample,
        reverse_dir_adj=trans_reverse_adj_list,
    )
    fix_head_test = TestSet(
        trans_graph,
        trans_triplets["test"],
        params,
        trans_adj_list,
        mode="eval",
    )
    fix_tail_test = TestSet(
        trans_graph,
        trans_triplets["test"],
        params,
        trans_reverse_adj_list,
        mode="eval",
        head_first=False,
    )

    fix_head_val = TestSet(
        trans_graph,
        trans_triplets["valid"],
        params,
        trans_adj_list,
        mode="eval",
    )
    fix_tail_val = TestSet(
        trans_graph,
        trans_triplets["valid"],
        params,
        trans_reverse_adj_list,
        mode="eval",
        head_first=False,
    )

    params.train_edges = len(train)

    params.test_size = len(fix_head_test)
    params.val_size = len(fix_head_val)

    params.train_sample_size = None
    params.eval_sample_size = None

    print(
        f"Training graph has {params.train_edges} edges, validataion set has {params.val_size} query edges, test set has {params.test_size} query edges"
    )

    # ---------------- Data Preparation End ------------- #

    feature_list = ["head", "tail", "dist", "Rel"]
    if params.gd_type == "VerGD":
        feature_list.append("HeadVerGD" + ("Deg" if params.gd_deg else ""))
        feature_list.append("TailVerGD" + ("Deg" if params.gd_deg else ""))
    elif params.gd_type == "HorGD":
        feature_list.append("HorGD")
    params.feature_list = feature_list

    gnn = RGCN(
        params.reach_dist,
        params.aug_num_rels,
        params.inp_dim,
        params.emb_dim,
        params.num_bases,
    )

    link_predictor = GDLinkPredictor(
        params.emb_dim, gnn, feature_list, params.num_rels
    )

    evlter = VarSizeRankingEvaluator("varsizeeval", params.num_workers)
    eval_metric = "mrr"

    # loss = NegLogLoss(params.neg_sample)
    loss = FirstPosNegLoss(params.neg_sample)

    link_pred = KGFilteredLinkPred(
        params.exp_dir,
        {
            "train": train,
            "val": [fix_head_val, fix_tail_val],
            "test": [fix_head_test, fix_tail_test],
        },
        params,
        gnn,
        link_predictor,
        loss,
        evlter,
    )

    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=3,
        callbacks=[
            TQDMProgressBar(refresh_rate=20),
            ModelCheckpoint(monitor=eval_metric, mode="max"),
        ],
        logger=CSVLogger(save_dir=params.exp_dir),
    )
    trainer.fit(link_pred)

    trainer.test()

    # hparams = (
    #     {"num_layers": [2, 3, 4, 5]}
    #     if params.psearch
    #     else {"num_layers": [params.num_layers]}
    # )

    # best_res = hyperparameter_grid_search(
    #     hparams,
    #     [train, fix_head_test, fix_tail_test, fix_head_val, fix_tail_val],
    #     run_exp,
    #     params,
    #     eval_metric,
    #     evlter,
    # )

    # write_res_to_file(
    #     osp.join(params.exp_dir, "result"),
    #     params.train_data_set,
    #     eval_metric,
    #     best_res["test_mean"],
    #     best_res["val_mean"],
    #     best_res["test_std"],
    #     best_res["val_std"],
    #     best_res["params"],
    #     json.dumps(best_res),
    # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gnn")

    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--train_data_set", type=str, default="WN18RR")
    parser.add_argument("--val_method", type=str, default="trans")

    parser.add_argument("--emb_dim", type=int, default=32)
    parser.add_argument("--mol_emb_dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--JK", type=str, default="last")
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--num_bases", type=int, default=4)

    parser.add_argument("--dropout", type=float, default=0)

    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--l2", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--neg_sample", type=int, default=50)

    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=1)

    parser.add_argument("--num_epochs", type=int, default=10)

    parser.add_argument("--reach_dist", type=int, default=3)

    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--fold", type=int, default=10)

    parser.add_argument("--gd_type", type=str, default="HorGD")

    parser.add_argument("--gdgnn", type=bool, default=False)
    parser.add_argument("--gd_deg", type=bool, default=False)

    parser.add_argument("--psearch", type=bool, default=False)

    params = parser.parse_args()
    set_random_seed(1)
    main(params)

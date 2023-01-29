import os
import torch
import argparse
import os.path as osp
import numpy as np
import json
from datetime import datetime
from datasets import HGVerGDNodeDataset

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from gnnfree.utils.evaluators import BinaryAccEvaluator
from gnnfree.utils.graph import construct_graph_from_edges
from gnnfree.utils.utils import (
    save_params,
    set_random_seed,
    k_fold_ind,
    k_fold2_split,
    hyperparameter_grid_search,
    multi_data_average_exp,
)
from gnnfree.nn.loss import MultiClassLoss
from gnnfree.utils.utils import write_res_to_file

from gnnfree.nn.models.GNN import HomogeneousGNN

from models.node_classifier import GDNodeClassifier

from lightning_template import HGNodePred


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

    params.data_folder_path = osp.join(params.data_path, params.train_data_set)
    e2i = {}
    with open(osp.join(params.data_folder_path, "graph.txt")) as f:
        edges = [line.split(",") for line in f.read().split("\n")]
    edges = np.array(edges).astype(int)
    nid = 0
    c_edges = []
    for e in edges:
        if e[0] not in e2i:
            e2i[e[0]] = nid
            nid += 1
        if e[1] not in e2i:
            e2i[e[1]] = nid
            nid += 1
        c_edges.append([e2i[e[0]], e2i[e[1]]])
    edges = np.array(c_edges)

    g = construct_graph_from_edges(
        edges[:, 0],
        edges[:, 1],
        n_entities=np.max(edges) + 1,
        inverse_edge=True,
    )
    g.ndata["feat"] = torch.ones((g.num_nodes(), 1))

    with open(osp.join(params.data_folder_path, "label.txt")) as f:
        label = [line.split(",") for line in f.read().split("\n")]
    label = np.array(label).astype(int)

    for i in range(len(label)):
        label[i, 0] = e2i[label[i, 0]]
    labels = label

    params.num_tasks = np.max(label[:, 1]) + 1
    params.inp_dim = 1
    params.num_class = params.num_tasks

    def construct_hg_ver_data(data, ind):
        node = [data[i, 0] for i in ind]
        label = [data[i, 1] for i in ind]
        d = HGVerGDNodeDataset(g, node, label, params)
        return d

    fold = 10
    folds = k_fold_ind(label[:, 1], fold=fold)
    splits = k_fold2_split(folds, len(labels))

    train = construct_hg_ver_data(label, splits[0][0])
    test = construct_hg_ver_data(label, splits[0][1])
    val = construct_hg_ver_data(label, splits[0][2])

    # ---------------- Data Preparation End ------------- #

    eval_metric = "acc"
    evlter = BinaryAccEvaluator("acc_eval")
    loss = MultiClassLoss()
    rep = 2

    def run_exp(data, params):
        params.reach_dist = params.num_layers
        train, test, val = data
        gnn = HomogeneousGNN(params.reach_dist, params.inp_dim, params.emb_dim)

        node_classifier = GDNodeClassifier(params.emb_dim, gnn, params.gd_deg)
        evlter.reset()

        node_pred = HGNodePred(
            params.exp_dir,
            {"train": train, "test": test, "val": val},
            params,
            gnn,
            node_classifier,
            loss,
            evlter,
            out_dim=params.num_class,
        )

        trainer = Trainer(
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,
            max_epochs=params.num_epochs,
            callbacks=[
                TQDMProgressBar(refresh_rate=20),
                ModelCheckpoint(monitor=eval_metric, mode="max"),
            ],
            logger=CSVLogger(save_dir=params.exp_dir),
        )
        trainer.fit(node_pred)
        valid_res = trainer.validate()[0]
        test_res = trainer.test()[0]
        return valid_res, test_res

    hparams = (
        {"num_layers": [2, 3, 4, 5]}
        if params.psearch
        else {"num_layers": [params.num_layers]}
    )

    best_res = hyperparameter_grid_search(
        hparams,
        [[train, test, val] for _ in range(rep)],
        multi_data_average_exp,
        params,
        eval_metric,
        evlter,
        run_exp,
    )

    write_res_to_file(
        osp.join(params.exp_dir, "result"),
        params.train_data_set,
        eval_metric,
        best_res["test_mean"],
        best_res["val_mean"],
        best_res["test_std"],
        best_res["val_std"],
        best_res["params"],
        json.dumps(best_res),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gnn")

    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--train_data_set", type=str, default="brazil_airport")

    parser.add_argument(
        "--emb_dim", type=int, default=32, help="overall embedding dimension"
    )
    parser.add_argument(
        "--mol_emb_dim",
        type=int,
        default=32,
        help="embedding dimension for atom/bond",
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="number of GNN layers"
    )
    parser.add_argument(
        "--JK",
        type=str,
        default="last",
        help="jumping knowledge, should be 'last' or 'sum'",
    )

    parser.add_argument("--dropout", type=float, default=0)

    parser.add_argument(
        "--lr", type=float, default=0.0001, help="learning rate"
    )
    parser.add_argument(
        "--l2", type=float, default=0, help="l2 regularization strength"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="training batch size"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=64, help="evaluation batch size"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=20,
        help="number of workers in dataloading",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=200,
        help="number of epochs in one training routine",
    )

    parser.add_argument(
        "--reach_dist",
        type=int,
        default=3,
        help="max cutoff distance to find geodesics",
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=10,
        help="number of fold for cross validation",
    )

    parser.add_argument(
        "--gd_type",
        type=str,
        default="HorGD",
        help="geodesic types, should be VerGD or HorGD",
    )

    parser.add_argument(
        "--gd_deg",
        type=bool,
        default=False,
        help="whether to use geodesic degrees for VerGD",
    )

    parser.add_argument(
        "--psearch",
        type=bool,
        default=False,
        help="perform hyperparameter search",
    )

    params = parser.parse_args()
    set_random_seed(1)
    main(params)

import os
import torch
import argparse
import os.path as osp
import numpy as np
import json
from datetime import datetime
from datasets import HGHorGDDataset, HGVerGDDataset

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from gnnfree.nn.models.GNN import HomogeneousGNN

from gnnfree.utils.evaluators import HNEvaluator
from gnnfree.utils.graph import construct_graph_from_edges
from gnnfree.utils.utils import (
    save_params,
    set_random_seed,
    hyperparameter_grid_search,
    write_res_to_file,
)
from gnnfree.nn.loss import BinaryLoss

from models.link_predictor import GDLinkPredictor

from lightning_template import HGLinkPred


def main(params):
    # ---------------------- Exp Prepare ------------------- #
    params = parser.parse_args()
    if not osp.exists("./saved_exp"):
        os.mkdir("./saved_exp")

    curtime = datetime.now()
    params.exp_dir = osp.join("./saved_exp", str(curtime))
    os.mkdir(params.exp_dir)

    save_params(osp.join(params.exp_dir, "command"), params)
    params.prefix = params.train_data_set + "_"
    params.model_name = osp.join(params.exp_dir, params.prefix)

    # ---------------------- Data Prepare ------------------- #

    if params.gd_type == "HorGD":
        TrainSet = HGHorGDDataset
    elif params.gd_type == "VerGD":
        TrainSet = HGVerGDDataset
    elif params.gd_type == "":
        TrainSet = None

    params.data_folder_path = osp.join(params.data_path, params.train_data_set)
    data = np.genfromtxt(
        osp.join(params.data_folder_path, params.train_data_set + ".txt"),
        delimiter=",",
    )
    data = data.astype(int)
    e2i = {}

    c_edges = []
    nid = 0
    for e in data:
        if e[0] not in e2i:
            e2i[e[0]] = nid
            nid += 1
        if e[1] not in e2i:
            e2i[e[1]] = nid
            nid += 1
        c_edges.append([e2i[e[0]], e2i[e[1]]])
    data = np.array(c_edges)

    train_num_entities = np.max(data) + 1
    head, tail = data[:, 0], data[:, 1]

    k = np.ones((train_num_entities, train_num_entities))
    k[head, tail] = 0
    k[tail, head] = 0
    nh, nt = k.nonzero()
    neg_perm = np.random.permutation(len(nt))
    perm = np.random.permutation(len(head))
    train_ind = int(len(perm) * 0.85)
    test_ind = int(len(perm) * 0.95)
    edges = np.zeros((len(head), 3), dtype=int)
    edges[:, 0] = head
    edges[:, 2] = tail
    neg_edges = np.zeros((len(head), 3), dtype=int)
    neg_edges[:, 0] = nh[neg_perm[: len(head)]]
    neg_edges[:, 2] = nt[neg_perm[: len(head)]]
    converted_triplets = {
        "train": edges[perm[:train_ind]],
        "train_neg": neg_edges[perm[:train_ind]],
        "test": edges[perm[train_ind:test_ind]],
        "test_neg": neg_edges[perm[train_ind:test_ind]],
        "valid": edges[perm[test_ind:]],
        "valid_neg": neg_edges[perm[test_ind:]],
    }

    g = construct_graph_from_edges(
        converted_triplets["train"].T[0],
        converted_triplets["train"].T[2],
        n_entities=train_num_entities,
        inverse_edge=True,
    )
    g.ndata["feat"] = torch.ones([g.num_nodes(), 1], dtype=torch.float32)
    params.inp_dim = 1

    train_edge = np.concatenate(
        [converted_triplets["train"], converted_triplets["train_neg"]], axis=0
    )
    train_label = np.zeros(len(train_edge), dtype=int)
    train_label[: len(converted_triplets["train"])] = 1
    train = TrainSet(g, train_edge, train_label, train_num_entities, params)
    params.train_edges = len(train)

    test_edge = np.concatenate(
        [converted_triplets["test"], converted_triplets["test_neg"]], axis=0
    )
    test_label = np.zeros(len(test_edge), dtype=int)
    test_label[: len(converted_triplets["test"])] = 1
    valid_edge = np.concatenate(
        [converted_triplets["valid"], converted_triplets["valid_neg"]], axis=0
    )
    valid_label = np.zeros(len(valid_edge), dtype=int)
    valid_label[: len(converted_triplets["valid"])] = 1
    test = TrainSet(
        g, test_edge, test_label, train_num_entities, params, mode="valid"
    )
    val = TrainSet(
        g, valid_edge, valid_label, train_num_entities, params, mode="valid"
    )
    params.test_size = len(test)
    params.val_size = len(val)
    params.train_sample_size = None
    params.eval_sample_size = None
    print(
        f"Training graph has {params.train_edges} edges, validataion set has"
        f"{params.val_size} query edges, test set has {params.test_size} query"
        f"edges"
    )
    # --------------------- Data Prep End --------------------- #
    loss = BinaryLoss()
    evlter = HNEvaluator("hne")
    eval_metric = "auc"

    def run_exp(data, params):
        params.reach_dist = params.num_layers
        feature_list = ["head", "tail", "dist"]
        if params.gd_type == "VerGD":
            feature_list.append("HeadVerGD" + ("Deg" if params.gd_deg else ""))
            feature_list.append("TailVerGD" + ("Deg" if params.gd_deg else ""))
        elif params.gd_type == "HorGD":
            feature_list.append("HorGD")
        params.feature_list = feature_list

        gnn = HomogeneousGNN(params.reach_dist, params.inp_dim, params.emb_dim)
        link_predictor = GDLinkPredictor(
            params.emb_dim, gnn, params.feature_list
        )

        link_pred = HGLinkPred(
            params.exp_dir,
            {"train": train, "val": val, "test": test},
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
        valid_res = trainer.validate()[0]
        test_res = trainer.test()[0]
        return valid_res, test_res

    hparams = (
        {"num_layers": [2, 3, 4, 5]}
        if params.psearch
        else {"num_layers": [params.num_layers]}
    )

    best_res = hyperparameter_grid_search(
        hparams, [train, test, val], run_exp, params, eval_metric, evlter
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
        default=100,
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

# -*- encoding: utf-8 -*-
"""
@Time       : 2020/07/16 15:39:00
@Author     : Yize Jiang
@Email      : yize.jiang@galixir.com
@File       : core.py
@Project    : X-DPI
@Description: DeepCPI核心预测
"""

import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import (accuracy_score, f1_score, mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score)
from torch.utils.data import DataLoader

from data import CPIDataset
from model.gnn import MultiGCN
from model.transformer import Decoder, Encoder
from optim.lookahead import Lookahead
from optim.radam import RAdam
from preprocessing.protein import ProteinFeatureManager


def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def ci(y, f):
    y = np.array(y)
    f = np.array(f)
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci


class Predictor(nn.Module):
    def __init__(self, args):
        super(Predictor, self).__init__()

        self.protein_dim = args.protein_dim
        self.atom_dim = args.atom_dim
        self.embedding_dim = args.embedding_dim
        self.mol2vec_embedding_dim = args.mol2vec_embedding_dim
        if args.objective == 'classification':
            decoder_hid_dim = args.hid_dim * 4  # for GalaxyDB
        else:
            decoder_hid_dim = args.hid_dim  # for Davis

        self.decoder = Decoder(
            atom_dim=decoder_hid_dim,
            hid_dim=args.hid_dim,
            n_layers=args.decoder_layers,
            n_heads=args.n_heads,
            pf_dim=args.pf_dim,
            dropout=args.dropout,
        )

        self.compound_gcn = MultiGCN(args, self.atom_dim, args.compound_gnn_dim)

        self.mol2vec_fc = nn.Linear(self.mol2vec_embedding_dim, self.mol2vec_embedding_dim)
        self.mol_concat_fc = nn.Linear(self.mol2vec_embedding_dim + args.compound_gnn_dim, decoder_hid_dim)
        self.mol_concat_ln = nn.LayerNorm(decoder_hid_dim)

        self.tape_fc = nn.Linear(self.embedding_dim, args.hid_dim * 4)

        self.protein_gcn = MultiGCN(args, self.protein_dim, args.protein_gnn_dim)

        self.protein_gcn_ones = MultiGCN(args, self.protein_dim, args.protein_gnn_dim)
        self.encoder = Encoder(
            protein_dim=args.protein_gnn_dim,
            hid_dim=args.hid_dim,
            n_layers=args.cnn_layers,
            kernel_size=args.cnn_kernel_size,
            dropout=args.dropout,
        )

        self.concat_fc = nn.Linear(args.hid_dim * 5, args.hid_dim)
        self.concat_ln = nn.LayerNorm(args.hid_dim)

        self.objective = args.objective

        if self.objective == "classification":
            self.fc = nn.Linear(256, 2)
        elif self.objective == "regression":
            self.fc = nn.Linear(256, 1)

    def make_masks(self, atom_num, protein_num, compound_max_len, protein_max_len):
        batch_size = len(atom_num)
        compound_mask = torch.zeros((batch_size, compound_max_len)).type_as(atom_num)
        protein_mask = torch.zeros((batch_size, protein_max_len)).type_as(atom_num)

        for i in range(batch_size):
            compound_mask[i, :atom_num[i]] = 1
            protein_mask[i, :protein_num[i]] = 1
        compound_mask = compound_mask.unsqueeze(1).unsqueeze(2)
        protein_mask = protein_mask.unsqueeze(1).unsqueeze(2)
        return compound_mask, protein_mask

    def make_single_masks(self, num, max_len):
        batch_size = len(num)
        mask = torch.zeros((batch_size, max_len)).type_as(num)
        for i in range(batch_size):
            mask[i, :num[i]] = 1
        mask = mask.unsqueeze(1).unsqueeze(2)
        return mask

    def forward(self, batch):
        compound_max_len = batch["COMPOUND_NODE_FEAT"].shape[1]
        protein_max_len = batch["PROTEIN_NODE_FEAT"].shape[1]

        compound_mask, protein_mask = self.make_masks(
            batch["COMPOUND_NODE_NUM"],
            batch["PROTEIN_NODE_NUM"],
            compound_max_len,
            protein_max_len,
        )

        compound_gcn = self.compound_gcn(batch["COMPOUND_NODE_FEAT"], batch["COMPOUND_ADJ"])

        mol2vec_embedding = self.mol2vec_fc(batch['COMPOUND_WORD_EMBEDDING'])
        compound = torch.cat([mol2vec_embedding, compound_gcn], dim=-1)
        compound = self.mol_concat_fc(compound)
        compound = self.mol_concat_ln(compound)

        gcn_protein = self.protein_gcn(batch["PROTEIN_NODE_FEAT"], batch["PROTEIN_MAP"])

        gcn_ones_protein = self.protein_gcn_ones(batch["PROTEIN_NODE_FEAT"], torch.ones_like(batch["PROTEIN_MAP"]))
        gcn_protein = gcn_protein + gcn_ones_protein

        enc_src = self.encoder(gcn_protein)

        tape_embedding = self.tape_fc(batch['PROTEIN_EMBEDDING'])
        enc_src = torch.cat([tape_embedding, enc_src], dim=-1)
        enc_src = self.concat_fc(enc_src)
        enc_src = self.concat_ln(enc_src)
        out = self.decoder(compound, enc_src, compound_mask, protein_mask)  # for attention

        if self.objective == "classification":
            out = self.fc(out)
        elif self.objective == "regression":
            out = self.fc(out).squeeze(1)
            out = torch.sigmoid(out) * 14
        return out


class DeepCPIModel(pl.LightningModule):
    """DeepCPIModel-Lightning模型

    Args:
        args: 模型参数, 参见main.py文件
    """
    def __init__(self, args):
        super(DeepCPIModel, self).__init__()
        self.args = args
        self.root_data_path = args.root_data_path

        self.predictor = Predictor(args)

        self.protein_feature_manager = ProteinFeatureManager(self.root_data_path)

        if args.objective == "classification":
            self.criterion = nn.CrossEntropyLoss()
        elif args.objective == "regression":
            self.criterion = nn.MSELoss()

    def forward(self, batch):
        return self.predictor(batch)

    def train_dataloader(self):
        train_set = CPIDataset(os.path.join(self.root_data_path, "split_data/train.csv"), self.protein_feature_manager,
                               self.args)
        print('The train data number is {}'.format(len(train_set)))
        return DataLoader(
            train_set,
            batch_size=self.args.batch_size,
            collate_fn=train_set.collate_fn,
            shuffle=True,
            num_workers=self.args.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        valid_set = CPIDataset(os.path.join(self.root_data_path, "split_data/valid.csv"), self.protein_feature_manager,
                               self.args)
        print('The valid data number is {}'.format(len(valid_set)))
        return DataLoader(
            valid_set,
            batch_size=self.args.batch_size,
            collate_fn=valid_set.collate_fn,
            shuffle=False,
            num_workers=self.args.num_workers,
        )

    def test_dataloader(self):
        if self.args.mode == 'test':
            if self.args.valid_test:
                test_set = CPIDataset(os.path.join(self.root_data_path + "split_data/valid.csv"),
                                      self.protein_feature_manager, self.args)
                print('Testing on validation data')
                self.args.valid_test = False
            else:
                print('Testing on test data')
                test_set = CPIDataset(os.path.join(self.root_data_path, "split_data/test.csv"),
                                      self.protein_feature_manager, self.args)
        else:
            test_set = CPIDataset(os.path.join(self.root_data_path + "split_data/test.csv"),
                                  self.protein_feature_manager, self.args)
        print('The test data number is {}'.format(len(test_set)))
        return DataLoader(
            test_set,
            batch_size=self.args.batch_size,
            collate_fn=test_set.collate_fn,
            shuffle=False,
            num_workers=self.args.num_workers,
        )

    def configure_optimizers(self):
        optimizer_inner = RAdam(self.parameters(), lr=self.args.learning_rate, weight_decay=1e-4)
        optimizer = Lookahead(optimizer_inner, k=5, alpha=0.5)
        return optimizer

    def training_step(self, batch, batch_idx):
        interactions = batch["LABEL"]
        if self.args.objective == "classification":
            interactions = interactions.long()

        outputs = self(batch)
        loss = self.criterion(outputs, interactions)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        interactions = batch["LABEL"]
        if self.args.objective == "classification":
            interactions = interactions.long()

        outputs = self(batch)
        loss = self.criterion(outputs, interactions)
        if self.args.objective == "classification":
            scores = F.softmax(outputs, dim=1)[:, 1].to("cpu").data.tolist()
            correct_labels = interactions.to("cpu").data.tolist()
            return {
                "val_loss": loss,
                "scores": scores,
                "correct_labels": correct_labels,
            }
        else:
            scores = outputs.to("cpu").data.tolist()
            labels = interactions.to("cpu").data.tolist()
            return {"val_loss": loss, "scores": scores, "labels": labels}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs])
        avg_loss = avg_loss.mean()

        tensorboard_logs = {"val_loss": avg_loss}

        if self.args.objective == "classification":
            scores, correct_labels = [], []
            for x in outputs:
                scores.extend(x["scores"])
                correct_labels.extend(x["correct_labels"])

            auc = torch.tensor(roc_auc_score(correct_labels, scores))
            tensorboard_logs["auc"] = auc
            return {"val_loss": avg_loss, "auc": auc, "log": tensorboard_logs}
        else:
            scores, correct_labels = [], []
            for x in outputs:
                scores.extend(x["scores"])
                correct_labels.extend(x["labels"])

            rmse = torch.tensor(np.sqrt(mean_squared_error(correct_labels, scores)))
            tensorboard_logs["rmse"] = rmse
            return {"val_loss": avg_loss, "rmse": rmse, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        interactions = batch["LABEL"]
        if self.args.objective == "classification":
            interactions = interactions.long()

        outputs = self(batch)
        loss = self.criterion(outputs, interactions)

        if self.args.objective == "classification":
            scores = F.softmax(outputs, dim=1)[:, 1].to("cpu").data.tolist()
            correct_labels = interactions.to("cpu").data.tolist()
            predict_labels = [1. if i >= 0.50 else 0. for i in scores]
            return {
                "test_loss": loss,
                "scores": scores,
                "correct_labels": correct_labels,
                "predict_labels": predict_labels,
                "sequences": batch["SEQUENCE"],
            }
        elif self.args.objective == "regression":
            predict_values = outputs.to("cpu").data.tolist()
            correct_values = interactions.to("cpu").data.tolist()
            return {
                "test_loss": loss,
                "predict_values": predict_values,
                "correct_values": correct_values,
            }

    def target_eval(self, gt_label, pre_label, scores, pros):
        seqs = set(pros)
        pros = np.array(pros)
        auc_t = []
        acc_t = []
        recall_t = []
        precision_t = []
        f1_score_t = []
        seq_t = []
        print('Target Test Results: Evaluate on {} proteins'.format(len(seqs)))
        for seq in seqs:
            index = pros == seq
            gt_label_t = gt_label[index]
            pre_label_t = pre_label[index]
            scores_t = scores[index]
            auc_t.append(roc_auc_score(gt_label_t, scores_t))
            acc_t.append(accuracy_score(gt_label_t, pre_label_t))
            recall_t.append(recall_score(gt_label_t, pre_label_t))
            precision_t.append(precision_score(gt_label_t, pre_label_t))
            f1_score_t.append(f1_score(gt_label_t, pre_label_t))
            seq_t.append(seq)
        auc_t = np.array(auc_t)
        acc_t = np.array(acc_t)
        recall_t = np.array(recall_t)
        precision_t = np.array(precision_t)
        f1_score_t = np.array(f1_score_t)
        seq_t = np.array(seq_t)
        print(" acc: {} (std: {}),".format(round(np.mean(acc_t), 4),
                                           round(np.std(acc_t),
                                                 4)), " auc: {}(std: {}),".format(round(np.mean(auc_t), 4),
                                                                                  round(np.std(auc_t), 4)),
              " precision: {}(std: {}),".format(round(np.mean(precision_t), 4), round(np.std(precision_t), 4)),
              " recall: {}(std: {}),".format(round(np.mean(recall_t), 4), round(np.std(recall_t), 4)),
              " f1_score: {}(std: {})".format(round(np.mean(f1_score_t), 4), round(np.std(f1_score_t), 4)))

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()

        if self.args.objective == "classification":
            scores, correct_labels, predict_labels, seqs = [], [], [], []
            for x in outputs:
                scores.extend(x["scores"])
                correct_labels.extend(x["correct_labels"])
                seqs.extend((x['sequences']))
            auc = roc_auc_score(correct_labels, scores)
            print(seqs[:10])
            print('auc', auc)
            thres = [0.5]
            for t in thres:
                print('=' * 50)
                print('The threshold is ', t)
                predict_labels = [1. if i >= t else 0. for i in scores]
                acc = accuracy_score(correct_labels, predict_labels)
                precision = precision_score(correct_labels, predict_labels)
                recall = recall_score(correct_labels, predict_labels)
                f1 = f1_score(correct_labels, predict_labels)
                # self.target_eval(np.array(correct_labels), np.array(predict_labels), np.array(scores), seqs)
                print(
                    " acc: {},".format(acc),
                    " precision: {},".format(precision),
                    " recall: {},".format(recall),
                    " f1_score: {}".format(f1),
                )
            return {
                "test_loss": avg_loss.item(),
                "auc": auc,
                "acc": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        elif self.args.objective == "regression":
            predict_values, correct_values = [], []
            for x in outputs:
                correct_values.extend(x["correct_values"])
                predict_values.extend(x["predict_values"])

            mse = mean_squared_error(correct_values, predict_values)
            rmse = np.sqrt(mean_squared_error(correct_values, predict_values))
            r2 = r2_score(correct_values, predict_values)
            pr = pearson(predict_values, correct_values)
            sr = spearman(predict_values, correct_values)
            # data = pd.read_csv(os.path.join(self.root_data_path, "split_data/test.csv"))
            # data['pred_label'] = predict_values
            # data.to_csv('evaluate_results.csv', index=False)
            return {
                "test_loss": avg_loss.item(),
                "mse": mse,
                "rmse": rmse,
                "r2_score": r2,
                "pearson": pr,
                "spearman": sr,
                'ci': ci(correct_values, predict_values)
            }

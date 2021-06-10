# -*- encoding: utf-8 -*-
'''
@Time       : 2020/07/15 10:52:39
@Author     : Yize Jiang
@Email      : yize.jiang@galixir.com
@File       : data.py
@Project    : X-DPI
@Description: DeepCPI数据集定义
'''

import pandas as pd
import torch
from gensim.models import Word2Vec, word2vec
from torch.utils.data import DataLoader, Dataset

from preprocessing.compound import get_mol2vec_features, get_mol_features
from preprocessing.protein import ProteinFeatureManager


class CPIDataset(Dataset):
    """CPI数据集

    Args:
        file_path: 数据文件路径（包含化合物SMILES、蛋白质氨基酸序列、标签）
        protein_feature_manager: 蛋白质特征管理器
    """
    def __init__(self, file_path, protein_feature_manager, args):
        self.args = args
        self.raw_data = pd.read_csv(file_path)
        self.smiles_values = self.raw_data['COMPOUND_SMILES'].values
        self.sequence_values = self.raw_data['PROTEIN_SEQUENCE'].values
        if args.objective == 'classification':
            self.label_values = self.raw_data['CLF_LABEL'].values
        else:
            self.label_values = self.raw_data['REG_LABEL'].values

        self.atom_dim = args.atom_dim
        self.protein_feature_manager = protein_feature_manager
        self.mol2vec_model = word2vec.Word2Vec.load("mol2vec/model_300dim.pkl")

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        smiles = self.smiles_values[idx]
        sequence = self.sequence_values[idx]
        label = self.label_values[idx]

        compound_node_features, compound_adj_matrix, _ = get_mol_features(smiles, self.atom_dim)
        compound_word_embedding = get_mol2vec_features(self.mol2vec_model, smiles)
        protein_node_features = self.protein_feature_manager.get_node_features(sequence)
        protein_contact_map = self.protein_feature_manager.get_contact_map(sequence)
        protein_seq_embedding = self.protein_feature_manager.get_pretrained_embedding(sequence)
        return {
            'COMPOUND_NODE_FEAT': compound_node_features,
            'COMPOUND_ADJ': compound_adj_matrix,
            'COMPOUND_WORD_EMBEDDING': compound_word_embedding,
            'PROTEIN_NODE_FEAT': protein_node_features,
            'PROTEIN_MAP': protein_contact_map,
            'PROTEIN_EMBEDDING': protein_seq_embedding,
            'LABEL': label,
            'SEQUENCE': sequence,
        }

    def collate_fn(self, batch):
        """自定义数据合并方法，将数据集中的数据通过Padding构造成相同Size

        Args:
            batch: 原始数据列表

        Returns:
            batch: 经过Padding等处理后的PyTorch Tensor字典
        """
        batch_size = len(batch)

        compound_node_nums = [item['COMPOUND_NODE_FEAT'].shape[0] for item in batch]
        protein_node_nums = [item['PROTEIN_NODE_FEAT'].shape[0] for item in batch]
        max_compound_len = max(compound_node_nums)
        max_protein_len = max(protein_node_nums)

        compound_node_features = torch.zeros((batch_size, max_compound_len, batch[0]['COMPOUND_NODE_FEAT'].shape[1]))
        compound_adj_matrix = torch.zeros((batch_size, max_compound_len, max_compound_len))
        compound_word_embedding = torch.zeros(
            (batch_size, max_compound_len, batch[0]['COMPOUND_WORD_EMBEDDING'].shape[1]))
        protein_node_features = torch.zeros((batch_size, max_protein_len, batch[0]['PROTEIN_NODE_FEAT'].shape[1]))
        protein_contact_map = torch.zeros((batch_size, max_protein_len, max_protein_len))
        protein_seq_embedding = torch.zeros((batch_size, max_protein_len, batch[0]['PROTEIN_EMBEDDING'].shape[1]))
        labels, seqs = list(), list()
        for i, item in enumerate(batch):
            v = item['COMPOUND_NODE_FEAT']
            compound_node_features[i, :v.shape[0], :] = torch.FloatTensor(v)
            v = item['COMPOUND_ADJ']
            compound_adj_matrix[i, :v.shape[0], :v.shape[0]] = torch.FloatTensor(v)

            v = item['COMPOUND_WORD_EMBEDDING']
            compound_word_embedding[i, :v.shape[0], :] = torch.FloatTensor(v)

            v = item['PROTEIN_NODE_FEAT']
            protein_node_features[i, :v.shape[0], :] = torch.FloatTensor(v)
            v = item['PROTEIN_MAP']
            protein_contact_map[i, :v.shape[0], :v.shape[0]] = torch.FloatTensor(v)
            
            v = item['PROTEIN_EMBEDDING']
            if self.args.objective == 'classification':
                v[:, 358] = (v[:, 358] - 7.8) / 6.5  # for GalaxyDB
            protein_seq_embedding[i, :v.shape[0], :] = torch.FloatTensor(v)[:max_protein_len, :]

            labels.append(item['LABEL'])
            seqs.append(item['SEQUENCE'])

        compound_node_nums = torch.LongTensor(compound_node_nums)
        protein_node_nums = torch.LongTensor(protein_node_nums)
        labels = torch.tensor(labels).type(torch.float32)

        return {
            'COMPOUND_NODE_FEAT': compound_node_features,
            'COMPOUND_ADJ': compound_adj_matrix,
            'COMPOUND_NODE_NUM': compound_node_nums,
            'COMPOUND_WORD_EMBEDDING': compound_word_embedding,
            'PROTEIN_NODE_FEAT': protein_node_features,
            'PROTEIN_MAP': protein_contact_map,
            'PROTEIN_EMBEDDING': protein_seq_embedding,
            'PROTEIN_NODE_NUM': protein_node_nums,
            'LABEL': labels,
            'SEQUENCE': seqs
        }


if __name__ == "__main__":
    galaxydb_data_path = 'data/GalaxyDB/'
    protein_feature_manager = ProteinFeatureManager(galaxydb_data_path)

    train_set = CPIDataset(galaxydb_data_path + 'split_data/train.csv', protein_feature_manager)
    valid_set = CPIDataset(galaxydb_data_path + 'split_data/valid.csv', protein_feature_manager)
    test_set = CPIDataset(galaxydb_data_path + 'split_data/test.csv', protein_feature_manager)

    item = train_set[3]
    print('Test Item:')
    print('Compound Node Feature Shape:', item['COMPOUND_NODE_FEAT'].shape)
    print('Compound Adjacency Matrix Shape:', item['COMPOUND_ADJ'].shape)
    print('Protein Node Feature Shape:', item['PROTEIN_NODE_FEAT'].shape)
    print('Protein Contact Map Shape:', item['PROTEIN_MAP'].shape)
    print('Protein Embedding Shape:', item['PROTEIN_EMBEDDING'].shape)
    print('Label:', item['LABEL'])

    train_loader = DataLoader(
        train_set,
        batch_size=32,
        collate_fn=train_set.collate_fn,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    print('')
    print('Test Batch:')
    for batch in train_loader:
        print('Compound Node Feature Shape:', batch['COMPOUND_NODE_FEAT'].shape)
        print('Compound Adjacency Matrix Shape:', batch['COMPOUND_ADJ'].shape)
        print('Compound Node Numbers Shape:', batch['COMPOUND_NODE_NUM'].shape)
        print('Protein Node Feature Shape:', batch['PROTEIN_NODE_FEAT'].shape)
        print('Protein Contact Map Shape:', batch['PROTEIN_MAP'].shape)
        print('Protein Embedding Shape:', batch['PROTEIN_EMBEDDING'].shape)
        print('Protein Node Numbers Shape:', batch['PROTEIN_NODE_NUM'].shape)
        print('Label Shape:', batch['LABEL'].shape)
        break

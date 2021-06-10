# -*- encoding: utf-8 -*-
'''
@Time       : 2020/07/15 11:09:30
@Author     : Yize Jiang
@Email      : yize.jiang@galixir.com
@File       : protein.py
@Project    : X-DPI
@Description: 蛋白质相关特征
'''

import os
import pickle

import numpy as np
import pandas as pd


def read_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class ProteinFeatureManager():
    def __init__(self, data_path):
        map_data = pd.read_csv(os.path.join(data_path, 'mapping.csv'))
        sequence_to_id = {}
        for seq, uniprot in zip(list(map_data['sequences']), list(map_data['uniprot'])):
            sequence_to_id[seq] = uniprot
        self.sequence_to_id = sequence_to_id

        with open(os.path.join(data_path, 'protein_embedding/bert_embedding_Nongram.pkl'), 'rb') as f:
            self.bert_embed_dict = pickle.load(f)

        self.data_path = data_path

    def get_node_features(self, sequence):
        return np.load(os.path.join(self.data_path, 'protein_node_features', self.sequence_to_id[sequence] + '.npy'))

    def get_contact_map(self, sequence):
        return np.load(os.path.join(self.data_path, 'protein_contact_map', self.sequence_to_id[sequence] + '.npy'))

    def get_pretrained_embedding(self, sequence):
        bert_embed = self.bert_embed_dict[sequence]
        return bert_embed


if __name__ == "__main__":
    protein_feature_manager = ProteinFeatureManager('data/GalaxyDB/')

    test_sequence = ('MWGLKVLLLPVVSFALYPEEILDTHWELWKKTHRKQYNNKVDEISRRLIWEKNLKYISIHNLEASLGVHTYELAM'
                     'NHLGDMTSEEVVQKMTGLKVPLSHSRSNDTLYIPEWEGRAPDSVDYRKKGYVTPVKNQGQCGSCWAFSSVGALEG'
                     'QLKKKTGKLLNLSPQNLVDCVSENDGCGGGYMTNAFQYVQKNRGIDSEDAYPYVGQEESCMYNPTGKAAKCRGYR'
                     'EIPEGNEKALKRAVARVGPVSVAIDASLTSFQFYSKGVYYDESCNSDNLNHAVLAVGYGIQKGNKHWIIKNSWGE'
                     'NWGNKGYILMARNKNNACGIANLASFPKM')
    node_features = protein_feature_manager.get_node_features(test_sequence)
    contact_map = protein_feature_manager.get_contact_map(test_sequence)
    pretrained_embedding = protein_feature_manager.get_pretrained_embedding(test_sequence)

    print(len(test_sequence))
    print(node_features.shape)
    print(contact_map.shape)
    print(pretrained_embedding.shape)

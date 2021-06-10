# -*- encoding: utf-8 -*-
'''
@Time       : 2020/07/15 12:11:48
@Author     : Yize Jiang
@Email      : yize.jiang@galixir.com
@File       : main.py
@Project    : X-DPI
@Description: 模型训练与预测程序入口
'''
import argparse
import os
import shutil

import pytorch_lightning as pl
import torch

from core import DeepCPIModel


def run(args: argparse.Namespace):
    """进行模型训练（测试）

    Args:
        args: 程序运行所需的命令行参数
    """
    model = DeepCPIModel(args)
    if args.objective == 'classification':
        monitor, mode = 'auc', 'max'
    else:
        monitor, mode = 'rmse', 'min'
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(min_delta=0,
                                                                    patience=args.early_stop_round,
                                                                    verbose=True,
                                                                    monitor=monitor,
                                                                    mode=mode)

    checkpoints_path = args.ckpt_save_path
    checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(filepath=checkpoints_path,
                                                                        save_top_k=1,
                                                                        verbose=True,
                                                                        monitor=monitor,
                                                                        mode=mode,
                                                                        prefix='')

    if args.mode == 'train':
        if not os.path.exists(checkpoints_path):
            os.mkdir(checkpoints_path)
        else:
            shutil.rmtree(checkpoints_path)
            os.mkdir(checkpoints_path)

        trainer = pl.Trainer(
            default_root_dir=checkpoints_path,
            max_epochs=args.max_epochs,
            gpus=args.gpus,
            num_sanity_val_steps=10,
            distributed_backend='ddp',
            early_stop_callback=early_stop_callback,
            checkpoint_callback=checkpoint_callback,
            deterministic=True,
            profiler=True,
        )
        trainer.fit(model)
        trainer.save_checkpoint(os.path.join(checkpoints_path, 'final.ckpt'))

    elif args.mode == 'test':
        trainer = pl.Trainer(
            default_root_dir='../',
            gpus='3',
            resume_from_checkpoint=args.ckpt_path,
        )
        if args.valid_test:
            valid_result = trainer.test(model)
            print(valid_result)
        result = trainer.test(model)
        print(result)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='test', help='Running Mode (train / test)')
    parser.add_argument('--ckpt_path',
                        type=str,
                        default='ckpts/GalaxyDB/final_model/_ckpt_epoch_12.ckpt',
                        help='CheckPoint File Path for Test')
    parser.add_argument('--ckpt_save_path', type=str, default='ckpts/debug/', help='CheckPoint File Path for save')

    parser.add_argument('--root_data_path', type=str, default='data/GalaxyDB', help='Raw Data Path')
    parser.add_argument('--objective',
                        type=str,
                        default='classification',
                        help='Objective (classification / regression)')

    parser.add_argument('--seed', type=int, default=2020, help='Random Seed')

    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size for Train(Validation/Test)')
    parser.add_argument('--max_epochs', type=int, default=50, help='Max Trainning Epochs')
    parser.add_argument('--max_steps', type=int, help='Max Trainning Steps for Native CPU')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of Subprocesses for Data Loading')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning Rate for Trainning')
    parser.add_argument('--early_stop_round', type=int, default=5, help='Early Stopping Round in Validation')

    parser.add_argument('--decoder_layers', type=int, default=3, help='Number of Layers for Decoder')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of Heads for Attention')
    parser.add_argument('--gnn_layers', type=int, default=3, help='Layers of GNN')
    parser.add_argument('--protein_gnn_dim', type=int, default=64, help='Hidden Dimension for Attention')
    parser.add_argument('--compound_gnn_dim', type=int, default=34, help='Hidden Dimension for Attention')
    parser.add_argument('--mol2vec_embedding_dim', type=int, default=300, help='Dimension for Mol2vec Embedding')
    parser.add_argument('--hid_dim', type=int, default=64, help='Hidden Dimension for Attention')
    parser.add_argument('--pf_dim', type=int, default=256, help='Hidden Dimension for Positional Feed Forward')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout Rate')

    parser.add_argument('--cnn_layers', type=int, default=3, help='CNN Conv Layers')
    parser.add_argument('--cnn_kernel_size', type=int, default=7, help='CNN Conv Layers')
    parser.add_argument('--valid_test', type=bool, default=False, help='Testing for validation data')

    parser.add_argument('--protein_dim', type=int, default=64, help='Dimension for Protein')
    parser.add_argument('--atom_dim', type=int, default=34, help='Dimension for Atom')
    parser.add_argument('--embedding_dim', type=int, default=768, help='Dimension for Embedding')

    return parser.parse_args()


if __name__ == '__main__':
    params = parse_args()
    params.gpus = '4, 5, 6, 7'
    # params.gpus = '0'
    print(params)
    pl.seed_everything(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    run(params)

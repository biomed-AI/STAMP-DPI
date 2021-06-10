# export ROOT_DATA_PATH=data/GalaxyDB
# export OBJECTIVE=classification
export ROOT_DATA_PATH=data/Davis
export OBJECTIVE=regression

python src_v1/main.py \
--mode train \
--objective $OBJECTIVE \
--root_data_path $ROOT_DATA_PATH \
--gpus 4 \
--batch_size 32 \
--max_epochs 100 \
--num_workers 4 \
--learning_rate 1e-4 \
--early_stop_round 100 \
--seed 2020 \
--decoder_layers 3 \
--cnn_layers 3 \
--cnn_kernel_size 7 \
--gnn_layers 3 \
--protein_gnn_dim 64 \
--compound_gnn_dim 34 \
--mol2vec_embedding_dim 300 \
--n_heads 8 \
--hid_dim 64 \
--pf_dim 256 \
--atom_dim 34 \
--protein_dim 64 \
--embedding_dim 768 \
--dropout 0.2 \
--ckpt_save_path ckpts/debug/

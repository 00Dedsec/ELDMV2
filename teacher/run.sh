python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --node_rank=0 train.py --config config/default.config --gpu 0,1
python train.py --config config/default.config --gpu 0,1 --checkpoint ./output/done_models/teacher_model/0.pkl
python train.py --config config/default.config --gpu 0,1
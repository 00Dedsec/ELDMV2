python train.py --config config/default.config --gpu 0,1 
python train.py --config config/SMASHRNN.config --gpu 1
python train.py --config config/BertWord.config --gpu 1
python train.py --config config/PLI.config --gpu 1
python train.py --config config/TextCNN.config --gpu 1
python train.py --config config/LFESM.config --gpu 0,1
python train.py --config config/cLawformer.config --gpu 1
python train.py --config config/BertWord.config --gpu 1 --checkpoint ./output/done_models/Bertword/Bertword/12.pkl
python test.py --config config/PLI.config --gpu 0,1 --checkpoint ./output/done_models/PLI/PLI/5.pkl --result ./output/result/prediction_PLI.txt
python train.py --config config/PLI.config --gpu 0 --checkpoint ./output/done_models/PLI/PLI/0.pkl
python -m torch.distributed.launch train.py --config config/LFESM.config --gpu 0,1

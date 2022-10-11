python train.py --config config/default.config --gpu 0,1 
python train.py --config config/SMASHRNN.config --gpu 1
python train.py --config config/BertWord.config --gpu 1
python train.py --config config/TextCNN.config --gpu 0
python train.py --config config/TextCNN.config --gpu 0 --checkpoint ./output/done_models/SMASHRNN/SMASHRNN/8.pkl
python test.py --config config/TextCNN.config --gpu 0 --checkpoint ./output/done_models/TextCNN/TextCNN/2.pkl --result ./output/result/prediction_texhcnn.txt
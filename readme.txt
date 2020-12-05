


安装paddle1.8.4
pip install paddlepaddle-gpu==1.8.4.post107 -i https://mirror.baidu.com/pypi/simple



安装paddle2.0
nvidia-smi
pip install pycocotools
python -m pip install paddlepaddle_gpu==2.0.0rc0 -f https://paddlepaddle.org.cn/whl/stable.html
cd ~/w*


下载预训练模型ppyolo.pdparams
cd ~/w*
wget https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams
python 1_ppyolo_2x_2paddle.py
rm -f ppyolo.pdparams



下载预训练模型ResNet50_vd_ssld_pretrained.tar
cd ~/w*
wget https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_ssld_pretrained.tar
tar -xf ResNet50_vd_ssld_pretrained.tar
python 1_r50vd_ssld_2paddle.py
rm -f ResNet50_vd_ssld_pretrained.tar
rm -rf ResNet50_vd_ssld_pretrained



下载预训练模型ppyolo_r18vd.pdparams
cd ~/w*
wget https://paddlemodels.bj.bcebos.com/object_detection/ppyolo_r18vd.pdparams
python 1_ppyolo_r18vd_2paddle.py
rm -f ppyolo_r18vd.pdparams






# 安装依赖、解压COCO2017数据集
nvidia-smi
cd ~
pip install pycocotools
cd data
cd data7122
unzip ann*.zip
unzip val*.zip
unzip tes*.zip
unzip image_info*.zip
unzip train*.zip
cd ~/w*



# 安装依赖、解压voc数据集
nvidia-smi
cd ~
pip install pycocotools
cd data
cd data4379
unzip pascalvoc.zip
cd ~/w*







-------------------------------- PPYOLO --------------------------------
训练
cd ~/w*
python train.py --config=0

cd ~/w*
python train.py --config=2




预测
cd ~/w*
python demo.py --config=0

cd ~/w*
python demo.py --config=2





验证
cd ~/w*
python eval.py --config=0

cd ~/w*
python eval.py --config=2




跑test_dev
cd ~/w*
python test_dev.py --config=0

cd ~/w*
python test_dev.py --config=2























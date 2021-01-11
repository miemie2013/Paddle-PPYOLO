


下载预训练模型ppyolo.pdparams
cd ~/w*
wget https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams
mkdir ./output/
mkdir ./output/ppyolo/
mv ppyolo.pdparams ./output/ppyolo/ppyolo.pdparams



下载预训练模型ppyolo_r18vd.pdparams
cd ~/w*
wget https://paddlemodels.bj.bcebos.com/object_detection/ppyolo_r18vd.pdparams




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

cd ~/w*
mkdir ~/data/data4379/pascalvoc/VOCdevkit/VOC2012/annotation_json/
cp voc2012_train.json ~/data/data4379/pascalvoc/VOCdevkit/VOC2012/annotation_json/voc2012_train.json
cp voc2012_val.json ~/data/data4379/pascalvoc/VOCdevkit/VOC2012/annotation_json/voc2012_val.json



cd ~/w*
rm -rf log*.txt


cd ~/w*
unzip P*.zip


-------------------------------- PPYOLO --------------------------------
训练
python tools/train.py -c configs/ppyolo/ppyolo_2x.yml --eval









导出后的预测
python tools/export_model.py  -c configs/ppyolo/ppyolo.yml --output_dir=./inference_model -o weights=output/ppyolo/model_final
python deploy/python/infer.py --model_dir=inference_model/ppyolo_2x --image_file=./images/test/000000000019.jpg --use_gpu=True


TensrRT FP32
python tools/export_model.py  -c configs/ppyolo/ppyolo.yml --output_dir=./inference_model -o weights=output/ppyolo/model_final --exclude_nms
CUDA_VISIBLE_DEVICES=0
python deploy/python/infer.py --model_dir=inference_model/ppyolo_2x --image_file=./images/test/000000000019.jpg --use_gpu=True --run_benchmark=True


TensrRT FP16
python tools/export_model.py  -c configs/ppyolo/ppyolo.yml --output_dir=./inference_model -o weights=output/ppyolo/model_final --exclude_nms
CUDA_VISIBLE_DEVICES=0
python deploy/python/infer.py --model_dir=inference_model/ppyolo_2x --image_file=./images/test/000000000019.jpg --use_gpu=True --run_benchmark=True --run_mode=trt_fp16


















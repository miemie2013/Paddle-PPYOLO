


下载预训练模型ppyolo.pdparams
cd ~/w*
wget https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams
mkdir ./output/
mkdir ./output/ppyolo/
mv ppyolo.pdparams ./output/ppyolo/ppyolo.pdparams



下载预训练模型ppyolo_r18vd.pdparams
cd ~/w*
wget https://paddlemodels.bj.bcebos.com/object_detection/ppyolo_r18vd.pdparams



下载预训练模型cascade_rcnn_cbr200_vd_fpn_dcnv2_nonlocal_softnms.tar
cd ~/w*
wget https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_cbr200_vd_fpn_dcnv2_nonlocal_softnms.tar
tar -xf cascade_rcnn_cbr200_vd_fpn_dcnv2_nonlocal_softnms.tar
mkdir ./output/
mkdir ./output/cascade_rcnn_cbr200_vd_fpn_dcnv2_nonlocal_softnms/
mv cascade_rcnn_cbr200_vd_fpn_dcnv2_nonlocal_softnms ./output/cascade_rcnn_cbr200_vd_fpn_dcnv2_nonlocal_softnms/cascade_rcnn_cbr200_vd_fpn_dcnv2_nonlocal_softnms
rm -f cascade_rcnn_cbr200_vd_fpn_dcnv2_nonlocal_softnms.tar


下载预训练模型fcos_r50_fpn_multiscale_2x.pdparams
cd ~/w*
wget https://paddlemodels.bj.bcebos.com/object_detection/fcos_r50_fpn_multiscale_2x.pdparams
mkdir ./output/
mkdir ./output/fcos_r50_fpn_multiscale_2x/
mv fcos_r50_fpn_multiscale_2x.pdparams ./output/fcos_r50_fpn_multiscale_2x/fcos_r50_fpn_multiscale_2x.pdparams


下载预训练模型fcos_dcn_r50_fpn_1x.pdparams
cd ~/w*
wget https://paddlemodels.bj.bcebos.com/object_detection/fcos_dcn_r50_fpn_1x.pdparams
mkdir ./output/
mkdir ./output/fcos_dcn_r50_fpn_1x/
mv fcos_dcn_r50_fpn_1x.pdparams ./output/fcos_dcn_r50_fpn_1x/fcos_dcn_r50_fpn_1x.pdparams





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
mkdir ~/data/data4379/pascalvoc/VOCdevkit/VOC2012/annotation_json/
cp voc2012_train.json ~/data/data4379/pascalvoc/VOCdevkit/VOC2012/annotation_json/voc2012_train.json
cp voc2012_val.json ~/data/data4379/pascalvoc/VOCdevkit/VOC2012/annotation_json/voc2012_val.json



cd ~/w*
rm -rf log*.txt


cd ~/w*
unzip P*.zip


-------------------------------- PPYOLO --------------------------------
训练
cd ~/w*
python tools/train.py -c configs/ppyolo/ppyolo_2x.yml --eval

cd ~/w*
python tools/train.py -c configs/dcn/cascade_rcnn_cbr200_vd_fpn_dcnv2_nonlocal_softnms.yml --eval

cd ~/w*
python tools/train.py -c configs/anchor_free/fcos_dcn_r50_fpn_1x.yml --eval

cd ~/w*
python tools/train.py -c configs/anchor_free/fcos_r50_fpn_multiscale_2x.yml --eval




恢复训练
cd ~/w*
python tools/train.py -c configs/ppyolo/ppyolo_2x.yml --eval -r output/ppyolo_2x/24000

cd ~/w*
python tools/train.py -c configs/dcn/cascade_rcnn_cbr200_vd_fpn_dcnv2_nonlocal_softnms.yml --eval -r output/aaaaa/24000




预测
cd ~/w*
python tools/infer.py -c configs/ppyolo/ppyolo_2x.yml --infer_dir=./demo/



验证
cd ~/w*
python tools/eval.py -c configs/ppyolo/ppyolo_2x.yml







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





















安装paddle1.8.4
pip install paddlepaddle-gpu==1.8.4.post107 -i https://mirror.baidu.com/pypi/simple



安装paddle2.0
nvidia-smi
pip install pycocotools
python -m pip install paddlepaddle_gpu==2.0.0rc0 -f https://paddlepaddle.org.cn/whl/stable.html
cd ~/w*


pip install --user paddlepaddle-gpu==2.0.0rc0 -i https://mirror.baidu.com/pypi/simple



# 解压预训练模型
nvidia-smi
cd ~/w*
cp ../data/data64119/dygraph_fcos_rt_dla34_fpn_4x.pdparams ./dygraph_fcos_rt_dla34_fpn_4x.pdparams
cp ../data/data64119/dygraph_fcos_rt_r50_fpn_4x.pdparams ./dygraph_fcos_rt_r50_fpn_4x.pdparams



下载预训练模型fcos_r50_fpn_multiscale_2x.pdparams
cd ~/w*
wget https://paddlemodels.bj.bcebos.com/object_detection/fcos_r50_fpn_multiscale_2x.pdparams
python 1_paddle_fcos_r50_fpn_multiscale_2x2paddle.py
rm -f fcos_r50_fpn_multiscale_2x.pdparams




下载预训练模型ResNet50_vd_ssld_pretrained.tar
cd ~/w*
wget https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_ssld_pretrained.tar
tar -xf ResNet50_vd_ssld_pretrained.tar
python 1_paddle_r50vd_ssld_2paddle.py
rm -f ResNet50_vd_ssld_pretrained.tar
rm -rf ResNet50_vd_ssld_pretrained






# 安装依赖、解压COCO2017数据集
nvidia-smi
cd ~
pip install pycocotools
pip install shapely
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







-------------------------------- 一些命令 --------------------------------
parser.add_argument('-c', '--config', type=int, default=0,
                    choices=[0, 1, 2, 3, 4],
                    help=textwrap.dedent('''\
                    select one of these config files:
                    0 -- fcos_r50_fpn_multiscale_2x.py
                    1 -- fcos_rt_r50_fpn_4x.py
                    2 -- fcos_rt_dla34_fpn_4x.py
                    3 -- fcos_rt_r50vd_fpn_dcn_2x.py
                    4 -- aaaaaaaaaaaaaa.py'''))

训练
cd ~/w*
python train_fcos.py --config=0

cd ~/w*
python train_fcos.py --config=1

cd ~/w*
python train_fcos.py --config=2

cd ~/w*
python train_fcos.py --config=3

cd ~/w*
python train_fcos.py --config=4




预测
cd ~/w*
python demo_fcos.py --config=0

cd ~/w*
python demo_fcos.py --config=1

cd ~/w*
python demo_fcos.py --config=2

cd ~/w*
python demo_fcos.py --config=3

cd ~/w*
python demo_fcos.py --config=4



预测并打包图片
cd ~/w*
python demo.py --config=2
rm -f out.zip
zip -r out.zip images/res/*.jpg




验证
cd ~/w*
python eval_fcos.py --config=0

cd ~/w*
python eval_fcos.py --config=1

cd ~/w*
python eval_fcos.py --config=2

cd ~/w*
python eval_fcos.py --config=3

cd ~/w*
python eval_fcos.py --config=4




跑test_dev
cd ~/w*
python test_dev.py --config=0

cd ~/w*
python test_dev.py --config=1

cd ~/w*
python test_dev.py --config=2

cd ~/w*
python test_dev.py --config=3

cd ~/w*
python test_dev.py --config=4









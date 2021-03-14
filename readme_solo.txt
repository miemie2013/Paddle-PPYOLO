


安装paddle1.8.4
pip install paddlepaddle-gpu==1.8.4.post107 -i https://mirror.baidu.com/pypi/simple



安装paddle2.0
nvidia-smi
pip install pycocotools
python -m pip install paddlepaddle_gpu==2.0.0rc0 -f https://paddlepaddle.org.cn/whl/stable.html
cd ~/w*





下载预训练模型solov2_light_r50_vd_fpn_dcn_512_3x.pdparams
cd ~/w*
wget https://paddlemodels.bj.bcebos.com/object_detection/solov2_light_r50_vd_fpn_dcn_512_3x.pdparams
python 1_paddle_solov2_light_r50_vd_fpn_dcn_512_3x2paddle.py
rm -f solov2_light_r50_vd_fpn_dcn_512_3x.pdparams


下载预训练模型solov2_r50_fpn_3x.pdparams
cd ~/w*
wget https://paddlemodels.bj.bcebos.com/object_detection/solov2_r50_fpn_3x.pdparams
python 1_paddle_solov2_r50_fpn_8gpu_3x2paddle.py
rm -f solov2_r50_fpn_3x.pdparams






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
rm -rf log*.txt


cd ~/w*
unzip P*.zip


-------------------------------- SOLO --------------------------------
parser.add_argument('-c', '--config', type=int, default=0,
                    choices=[0, 1, 2, 3, 4],
                    help=textwrap.dedent('''\
                    select one of these config files:
                    0 -- solov2_r50_fpn_8gpu_3x.py
                    1 -- solov2_light_448_r50_fpn_8gpu_3x.py
                    2 -- solov2_light_r50_vd_fpn_dcn_512_3x.py'''))


训练
cd ~/w*
python train_solo.py --config=0

cd ~/w*
python train_solo.py --config=1

cd ~/w*
python train_solo.py --config=2



预测
cd ~/w*
python demo_solo.py --config=0

cd ~/w*
python demo_solo.py --config=1

cd ~/w*
python demo_solo.py --config=2




验证
cd ~/w*
python eval_solo.py --config=0

cd ~/w*
python eval_solo.py --config=1

cd ~/w*
python eval_solo.py --config=2



跑test_dev
cd ~/w*
python test_dev_solo.py --config=0

cd ~/w*
python test_dev_solo.py --config=1

cd ~/w*
python test_dev_solo.py --config=2

















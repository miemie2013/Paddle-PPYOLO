# PPYOLO

## 概述

请前往AIStudio体验：[Paddle2.0动态图版PPYOLO的简单实现](https://aistudio.baidu.com/aistudio/projectdetail/1156231)

Paddle2.0动态图版本的PPYOLO，训练速度约为PaddleDetection原版的92%（单卡，批大小24，没开ema的情况下。没开ema时100step / 105s = 0.95 step/s，开ema时100step / 135s = 0.74 step/s）。

## 未实现的部分

多卡训练：咩酱没有多卡+懒，故不实现。

## 快速开始
(1)获取预训练模型

```
! cd ~/work; wget https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams
! cd ~/work; python 1_ppyolo_2x_2paddle.py
```
(2)使用模型预测图片、获取FPS（预测images/test/里的图片，结果保存在images/res/）
```
! cd ~/work; python demo.py --config=0
```
--config=0表示使用了0号配置文件ppyolo_2x.py，配置文件代号与配置文件的对应关系在tools/argparser.py文件里：
parser.add_argument('-c', '--config', type=int, default=0,
                    choices=[0, 1, 2, 3, 4, 5],
                    help=textwrap.dedent('''\
                    select one of these config files:
                    0 -- ppyolo_2x.py
                    1 -- yolov4_2x.py
                    2 -- ppyolo_r18vd.py
                    3 -- ppyolo_mobilenet_v3_large.py
                    4 -- ppyolo_mobilenet_v3_small.py
                    5 -- ppyolo_mdf_2x.py'''))
train.py、eval.py、demo.py、test_dev.py都需要指定--config参数表示使用哪个配置文件，后面不再赘述。


## 数据集的放置位置
如果不是在AIStudio上训练，而是在个人电脑上训练，数据集应该和本项目位于同一级目录(同时需要修改一下配置文件中self.train_path、self.val_path这些参数使其指向数据集)。一个示例：
```
D://GitHub
     |------COCO
     |        |------annotations
     |        |------test2017
     |        |------train2017
     |        |------val2017
     |
     |------VOCdevkit
     |        |------VOC2007
     |        |        |------Annotations
     |        |        |------ImageSets
     |        |        |------JPEGImages
     |        |        |------SegmentationClass
     |        |        |------SegmentationObject
     |        |
     |        |------VOC2012
     |                 |------Annotations
     |                 |------ImageSets
     |                 |------JPEGImages
     |                 |------SegmentationClass
     |                 |------SegmentationObject
     |
     |------Paddle-PPYOLO-master
              |------annotation
              |------config
              |------data
              |------model
              |------...
```


## 训练

如果你需要训练COCO2017数据集，那么需要先解压数据集

```
! pip install pycocotools
! cd ~/data/data7122/; unzip ann*.zip
! cd ~/data/data7122/; unzip val*.zip
! cd ~/data/data7122/; unzip tes*.zip
! cd ~/data/data7122/; unzip image_info*.zip
! cd ~/data/data7122/; unzip train*.zip
```

获取预训练的resnet50vd_ssld

```
! cd ~/work; wget https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_ssld_pretrained.tar
! cd ~/work; tar -xf ResNet50_vd_ssld_pretrained.tar
! cd ~/work; python 1_r50vd_ssld_2paddle.py
! cd ~/work; rm -f ResNet50_vd_ssld_pretrained.tar
! cd ~/work; rm -rf ResNet50_vd_ssld_pretrained
```

再输入以下命令训练（所有的配置都在config/ppyolo_2x.py里，请查看代码注释做相应修改。如果你抢到32GB的V100，可以开batch_size=24，否则请调小batch_size。使用的预训练模型是config/ppyolo_2x.py里self.train_cfg -> model_path指定的模型）

```
! cd ~/work; python train.py --config=0
```


## 训练自定义数据集
自带的voc2012数据集是一个很好的例子。

将自己数据集的txt注解文件放到annotation目录下，txt注解文件的格式如下：
```
xxx.jpg 18.19,6.32,424.13,421.83,20 323.86,2.65,640.0,421.94,20
xxx.jpg 48,240,195,371,11 8,12,352,498,14
# 图片名 物体1左上角x坐标,物体1左上角y坐标,物体1右下角x坐标,物体1右下角y坐标,物体1类别id 物体2左上角x坐标,物体2左上角y坐标,物体2右下角x坐标,物体2右下角y坐标,物体2类别id ...
```
注意：xxx.jpg仅仅是文件名而不是文件的路径！xxx.jpg仅仅是文件名而不是文件的路径！xxx.jpg仅仅是文件名而不是文件的路径！

运行1_txt2json.py会在annotation_json目录下生成两个coco注解风格的json注解文件，这是train.py支持的注解文件格式。
在config/ppyolo_2x.py里修改train_path、val_path、classes_path、train_pre_path、val_pre_path、num_classes这6个变量（自带的voc2012数据集直接解除注释就ok了）,就可以开始训练自己的数据集了。
而且，直接加载dygraph_ppyolo_2x.pdparams的权重（即配置文件里修改train_cfg的model_path为'dygraph_ppyolo_2x.pdparams'）训练也是可以的，这时候也仅仅不加载3个输出卷积层的6个权重（因为类别数不同导致了输出通道数不同）。
如果需要跑demo.py、eval.py，与数据集有关的变量也需要修改一下，应该很容易看懂。

## 评估
运行以下命令。评测的模型是config/ppyolo_2x.py里self.eval_cfg -> model_path指定的模型

```
! cd ~/work; python eval.py --config=0
```

该mAP是val集的结果。

## 预测
运行以下命令。使用的模型是config/ppyolo_2x.py里self.test_cfg -> model_path指定的模型

```
! cd ~/work; python demo.py --config=0
```

喜欢的话点个喜欢或者关注我哦~
为了更方便地查看代码、克隆仓库、跟踪更新情况，该仓库不久之后也会登陆我的GitHub账户，对源码感兴趣的朋友可以提前关注我的GitHub账户鸭（求粉）~

AIStudio: [asasasaaawws](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/165135)

GitHub: [miemie2013](https://github.com/miemie2013)


## 传送门

咩酱重写过很多算法，比如PPYOLO、SOLOv2、FCOS、YOLOv4等，而且在多个深度学习框架（tensorflow、pytorch、paddlepaddle等）上都实现了一遍，你可以进我的GitHub主页看看，看到喜欢的仓库可以点个star呀！

cv算法交流q群：645796480
但是关于仓库的疑问尽量在Issues上提，避免重复解答。

本人微信公众号：miemie_2013

技术博客：https://blog.csdn.net/qq_27311165

AIStudio主页：[asasasaaawws](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/165135)

欢迎在GitHub或AIStudio上关注我（求粉）~

## 打赏

如果你觉得这个仓库对你很有帮助，可以给我打钱↓
![Example 0](weixin/sk.png)

咩酱爱你哟！另外，有偿接私活，可联系微信wer186259，金主快点来吧！

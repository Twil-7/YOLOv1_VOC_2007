# YOLOv1_VOC_2007

# 环境配置：


python == 3.8

keras == 2.4.3

tensorflow == 2.4.1

opencv-python == 4.5.3.56


# 文件介绍： 

先运行main.py搭建模型进行训练，再运行yolo_predict.py调用训练好的权重进行检测。

Annotations、JPEGImages文件夹：存储VOC2007目标检测数据集。

下载路径：https://blog.csdn.net/Twilight737?spm=1018.2226.3001.5343&type=download

demo文件夹：该模型对测试集的检测效果，大体使人满意，但轮廓的精确度不是十分的高。

Logs文件夹：存储训练过程中的权重文件。

raw_weights.hdf5：模型初始训练时加载进的权重文件，成功加载后，将网络的特征提取部分冰冻起来，此部分不再训练。

下载路径：https://blog.csdn.net/Twilight737?spm=1018.2226.3001.5343&type=download

tiny_yolov1_model.py：tiny_yolov1模型，其特征提取backbone比yolov1简单很多，在VOC2007数据集下仍效果不错。

yolo_loss.py：损失函数，对无目标物体置信度加权系数0.5，有目标物体置信度加权系数1，类别加权系数1，物体x、y、w、h坐标位置加权系数5。

train.py文件：导入初始权重raw_weights.hdf5，冰冻特征提取部分前31层，优化器adam = Adam(lr=1e-4, amsgrad=True)效果最佳。

predict.py：检测测试集的目标检测效果。

# 训练结果

最佳优化器：先使用adam = Adam(lr=1e-4, amsgrad=True)，后使用sgd = optimizers.SGD(lr=1e-5, momentum=0.9)，效果最佳。

最佳val loss = 67.05，后续达到瓶颈无法降低。

测试集的检测效果大体使人满意。

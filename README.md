
## 基于CNN和LSTM得人脸微表情识别



本设计实现一种基于深度学习的微表情识别系统，通过结合卷积神经网络（CNN）和长短时记忆网络（LSTM），有效捕捉微表情的空间特征和时间动态。采用了改进的CNN模型来提取面部图像的空间特征，并利用LSTM网络捕捉这些表情随时间的细微变化，以识别短暂出现的微表情。在CASMEⅡ数据集上的实验结果表明，本研究提出的深度学习模型在微表情识别上相比传统方法有显著提高，准确率可达78.90%。此外，本文还设计了一个GUI界面提供用户交互功能，不仅支持上传图片，上传视频和实时识别三种预测方法，还支持用户选择微表情还是宏观表情模型进行预测，还可支持对结果进行可视化操作。
![image](https://github.com/user-attachments/assets/cf555f23-b734-4aa4-8261-507fa33efd34)

***论文地址：[https://ieeexplore.ieee.org/abstract/document/10672980/]()***

**由于模型文件过大，LSTM,CNN,VGG16三个模型文件放在百度网盘中，请自行下载。 通过百度网盘分享的文件：
链接：https://pan.baidu.com/s/1YvWBrIxPF3cdUagZ0WCYyA?pwd=qlcl
提取码：qlcl**

++- - 本微表情模型的数据集采用的是中科院的CASMEⅡ数据集，可自行去申请。++

***
***文件说明：***
lstm train.ipynb:微表情识别模型定义和训练文件

cnn train.ipynb:卷积神经网络模型定义和训练，用于提取人脸特征

models：存放各模型的权重

GUI：交互文件，其中Emo_gui2.ui使用Qt Designer设计生成，Ui_Emo_gui2.py通过VSCode编译成py文件，GUI_main.py是逻辑交互的主文件

FaceDect：人脸预测文件，train.ipynb中包括了对数据集的一些预处理和训练，main.ipynb实现一个简单的预测

***
**==使用方法：直接运行GUI/GUI_main.py文件，如果有相关模型路径不对问题，请修改路径。==**
## **模型的基本结构**
![image](https://github.com/user-attachments/assets/f7ce2eba-de83-4fc3-b43d-17e0bae22147)

采用将预训练的卷积神经网络和长短期记忆网络相结合的模型架构。首先导入预训练的CNN模型权重进行特征提取，并将CNN模型的权重固定为不可训练,以避免特征提取部分的参数在训练中被改变。
对于输入的微表情图像序列,将每一帧图像通过上述特征提取过程获取对应的特征向量,并将这些特征向量按时序组合成特征序列,作为LSTM的输入。设计两层LSTM层，捕捉时序之间的关系。最终连接全连接层,输出每个表情类别的预测概率分布。
### CNN和LSTM模型定义如下：
**CNN:**
![image](https://github.com/user-attachments/assets/d6c89472-a408-42f3-ac00-62d5fe3220db)

![image](https://github.com/user-attachments/assets/baa56c34-b251-448b-a28b-766683774ad6)

**LSTM:**
![image](https://github.com/user-attachments/assets/6286f4c6-3386-43df-a6cf-6bee28333687)


## 微表情识别系统的设计
1. **人脸检测的实现**
在本研究中使用YOLOv8模型实现人脸检测功能。使用Ultralytics提供的YOLO接口加载预训练的YOLOv8n模型,并基于配置好的人脸数据集对此YOLO模型进行微调。
其中人脸检测数据集可见：[https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset]()
```python
from ultralytics import YOLO
# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='face.yaml', epochs=6, batch=16, workers=0, amp=False)  
```
2. **GUI界面的设计**
使用了Qt Designer设计GUI页面，提供上传图片，上传视频，实时识别等功能。同时还可选择不同的模型进行预测。左下角显示预测结果，同时绘制各表情概率柱状图，右下角显示实时图像。
![image](https://github.com/user-attachments/assets/3fe89bcb-67f2-4d44-8a47-8a13fb5461f2)



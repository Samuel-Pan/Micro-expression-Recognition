from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene, QGraphicsTextItem, QGraphicsRectItem
import sys
import cv2
from keras.models import load_model
import numpy as np
from ultralytics import YOLO
from Ui_Emo_gui2 import Ui_MainWindow
from PyQt5.QtGui import QColor, QFont, QBrush
from keras import backend as K
from collections import deque

LABEL_MAPS = {
    "微表情模型": {
        0: 'disgust',
        1: 'happiness',
        2: 'others',
        3: 'repression',
        4: 'surprise'
    },
    "表情模型 VGG16": {
        0: 'Angry',
        1: 'Contempt',
        2: 'Disgust',
        3: 'Fear',
        4: 'Happy',
        5: 'Neutral',
        6: 'Sadness',
        7: 'Surprise'
    },
    "表情模型 ResNet50": {
        0: 'Angry',
        1: 'Disgust',
        2: 'Fear',
        3: 'Happy',
        4: 'Neutral',
        5: 'Sadness',
        6: 'Surprise'
    }
}


# 加载yolo模型
global_yolo_model = YOLO('../models/best.pt')
model_shape=(48,48)
prev_box = None  # 全局变量，用于存储上一帧的人脸位置
THRESHOLD=10
showbar = True
seq_length = 32

# 初始化一个固定大小的队列来存储连续帧
frame_queue = deque(maxlen=seq_length)


# 人脸检测，判断是否识别到人脸，并返回人脸坐标
def face_dec(image, yolo_model = None):
    global prev_box
    
    if yolo_model is None:
        yolo_model = global_yolo_model

    # yolo_model = yolo_model
    # img = cv2.imread(image)
    results = yolo_model(image)
    for result in results:
        print("正在识别人脸")
        pred_box = result.boxes.xyxy.tolist()
        if len(pred_box) == 0:
            print("没有检测到人脸")
            return False
        print("识别到人脸")
        # return pred_box
        if prev_box is not None:  # 如果不是第一帧，比较当前帧与上一帧的位置
            # 计算两个矩形框的中心点
            prev_center = [(prev_box[0][0] + prev_box[0][2]) / 2, (prev_box[0][1] + prev_box[0][3]) / 2]
            current_center = [(pred_box[0][0] + pred_box[0][2]) / 2, (pred_box[0][1] + pred_box[0][3]) / 2]

            # 计算中心点之间的距离
            distance = ((prev_center[0] - current_center[0]) ** 2 + (prev_center[1] - current_center[1]) ** 2) ** 0.5

            if distance < THRESHOLD:  # 如果距离小于阈值
                pred_box = prev_box  # 保持上一帧的位置，但用于输出
            else:
                prev_box = pred_box  # 更新上一帧的位置为当前位置

            return pred_box  # 返回最终决定使用的位置

        else:
            prev_box = pred_box  # 如果是第一帧，直接更新位置
            return pred_box

# 图像预处理，返回预处理后的图像以及矩形坐标
def process_image(imagePath, is_dec, i, image=None):
    # 判断是直接使用图像还是从文件路径加载图像
    if image is None or not image.any():
        image = cv2.imread(imagePath)
        
    # 从is_dec中获取裁剪区域坐标
    x1, y1, x2, y2 = map(int, is_dec[i])
    # 裁剪图像
    crop = image[y1:y2, x1:x2]
    '''
    # cv2.imshow("or", crop)
    # 计算原始裁剪区域的高度
    original_crop_height = y2 - y1

    # 计算额外裁剪的高度 (上方15% 和 下方10%)
    additional_crop_top = int(0.15 * original_crop_height)
    additional_crop_bottom = int(0.10 * original_crop_height)

    # 更新y1和y2以进一步裁剪
    y1_new = y1 + additional_crop_top
    y2_new = y2 - additional_crop_bottom

    # 根据更新的坐标再次裁剪图像
    crop = crop[y1_new-y1:y2_new-y1, :]
    # cv2.imshow("ori", crop)
    '''
    # 将裁剪后的图像调整尺寸
    resized_image = cv2.resize(crop, model_shape )
    # cv2.imshow("ori2", crop)
    # cv2.imshow('resized', resized_image)
    # 对图像进行去噪处理
    denoised_image = cv2.medianBlur(resized_image, 3)  # 5 是中值滤波的内核大小

    # 对图像进行锐化处理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    sharpened_image = cv2.dilate(denoised_image, kernel)

    # 将图像转换为灰度并归一化
    img_gray = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2GRAY) / 255.0
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img_gray, x1, y1, x2, y2

# 宏观表情预测
def predict_emotion(model, image):
    # 返回预测的表情字符串
    gray = image[:,:,np.newaxis]
    gray = np.expand_dims(gray, axis=0) 
    # image = np.expand_dims(image, axis=0)
    prd = model.predict(gray, verbose=0) # verbose=0表示不输出日志信息
    return prd

# 微表情预测
def Micro_predict_emotion(sequence, model):
    img_array = np.array(sequence)
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    prd = model.predict(img_array, verbose=0)
    return prd

# 图像微表情预测
def Micro_Pic_predict_emotion(image, model):
    img_array = np.expand_dims(image, axis=0)  # 增加批处理维度

    # 正确地沿着序列长度的维度重复图片
    img_array = np.repeat(img_array, 80, axis=0)
    img_array = np.expand_dims(img_array, axis=0)  # 再次增加一个维度以匹配模型输入的期望形状
    prd = model.predict(img_array, verbose=0)
    return prd


# 显示预测结果和柱状图
def displayPrediction(self, prediction):
    # 清空之前的场景内容
    self.scene.clear()
    # 显示 "Emotion:" 文本
    emotion = self.label_map[np.argmax(prediction)]
    label_text_item = QtWidgets.QGraphicsTextItem("Predicted Emotion: ")
    label_text_item.setPos(-35, 0)  # 设置文本位置
    label_text_item.setFont(QFont("Arial", 14))
    self.scene.addItem(label_text_item)

    # 创建 QFont 对象设置字体和大小
    font = QFont("Arial", 20)  # 参数为字体类型和字体大小

    # 创建 QBrush 对象设置文本颜色
    brush = QBrush(QColor(255, 0, 0))  # 参数为一个 QColor 对象，这里设置为红色

    # 显示 predicted_emotion 文本
    emotion_text_item = QtWidgets.QGraphicsTextItem(emotion)
    emotion_text_item.setPos(-25, 0)  # 设置文本位置
    emotion_text_item.setFont(font)  # 应用字体设置
    emotion_text_item.setDefaultTextColor(brush.color())  # 应用颜色设置
    emotion_text_item.setPos(label_text_item.boundingRect().width(), 0)  # 设置文本位置紧跟 "Predicted Emotion:" 之后
    self.scene.addItem(emotion_text_item)

    if showbar:
        # 设置柱状图的起始位置（文本右侧）
        start_y = emotion_text_item.boundingRect().height() + 20

        # 定义柱状图的参数
        max_width = 180  # 柱状图的最大宽度
        bar_height = 20  # 柱状图的高度
        spacing = 30  # 柱状图之间的间距

        for i, score in enumerate(prediction[0]):
            # 绘制柱状图
            bar_width = score * max_width  # 根据得分计算柱状图的宽度
            bar_item = QtWidgets.QGraphicsRectItem(0, start_y + i * (bar_height + spacing), bar_width, bar_height)
            bar_item.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 255)))  # 设置柱状图的填充颜色为蓝色
            self.scene.addItem(bar_item)

            # 在柱状图右侧显示概率值
            probability_text = "{:.2%}".format(score)
            probability_text_item = QtWidgets.QGraphicsTextItem(probability_text)
            probability_text_item.setPos(bar_width + 5, start_y + i * (bar_height + spacing))
            self.scene.addItem(probability_text_item)

            # 在柱状图左侧显示表情标签
            label = self.label_map[i]
            label_item = QtWidgets.QGraphicsTextItem(label)
            label_item.setPos(-label_item.boundingRect().width() - 10, start_y + i * (bar_height + spacing))
            self.scene.addItem(label_item)
        
# 在这里添加自定义的应用逻辑类
class EmotionRecognitionApp(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(EmotionRecognitionApp, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.isMirco = False
        self.initUI()
        # 加载表情识别模型
         # 初始化模型为空
        self.model = None
    
        self.sequence_ready = False  # 标志位，指示是否有足够的帧进行预测
        
        # 连接QComboBox的信号到槽函数
        self.ui.comboBox.currentIndexChanged.connect(self.model_selected)

        self.timer = QtCore.QTimer()
        # 是否调用摄像头，默认None
        self.cap = None

        # 添加用于跟踪实时识别状态的变量
        self.isRealTimeRecognitionActive = False

         # 初始化 QGraphicsScene
        self.scene = QtWidgets.QGraphicsScene(self)
        self.ui.graphicsView.setScene(self.scene)


    def initUI(self):
        # 设置窗口标题和其他自定义UI初始化
        self.setWindowTitle("微表情识别系统")
        
        self.ui.pushButton_2.clicked.connect(self.uploadImage)
        self.ui.pushButton_3.clicked.connect(self.uploadVideo)
        self.ui.pushButton.clicked.connect(self.realTimeRecognition)

    #选择模型
    def model_selected(self, index):
        global model_shape
        # 根据选择的模型加载模型
        model_name = self.ui.comboBox.itemText(index)

        # 首先断开pushButton_2的所有现有连接
        self.ui.pushButton_2.clicked.disconnect()

        if model_name == "表情模型 VGG16":
            #释放模型
            K.clear_session()
            self.model = None  # 释放原模型对象，前提是没有其他变量引用它
            self.model = load_model('../models/V16_Plus_0228_1412_ACC8205.h5')
            model_shape=(48, 48)
            self.isMirco = False 
            self.sequence_ready = True
            self.ui.pushButton_2.clicked.connect(self.uploadImage)
        elif model_name == "表情模型 ResNet50":
            K.clear_session()
            self.model = False  
            self.model = load_model('../models/Resnet50_Ps+_0224_1630_acc8089.h5')
            model_shape=(48, 48)
            self.isMirco = False
            self.sequence_ready = True
            self.ui.pushButton_2.clicked.connect(self.uploadImage)
        elif model_name == "微表情模型":
            K.clear_session()
            self.model = None 
            self.model = load_model('../models/lstm_casme2_0418_1652ACC7890.h5')
            model_shape=(128, 128)
            self.isMirco = True
            self.sequence_ready = False
            self.ui.pushButton_2.clicked.connect(self.uploadImageSequence)
        self.label_map = LABEL_MAPS.get(model_name, {})
        print(f"已加载模型: {model_name}，标签映射：{self.label_map}")

    # 上传图片序列模块
    def uploadImageSequence(self):
        print("uploadImageSequence")
        queue=[]
        # 清空之前的场景内容
        self.scene.clear()

        # 图片上传逻辑
        # 打开文件对话框选择图片
        imagePaths, _ = QFileDialog.getOpenFileNames(self, "Open Image Files", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)")
        if imagePaths:
            for imagePath in imagePaths:
                # 判断是否检测到人脸
                is_dec = face_dec(imagePath)  
                if is_dec:
                    # img = cv2.imread(imagePath)
                    # 图像预处理
                    processed_image = process_image(imagePath, is_dec, i=0)
                    img_gray, x1, y1, x2, y2 = processed_image[0], processed_image[1], processed_image[2], processed_image[3], processed_image[4]
                    if len(queue) >= seq_length:
                        queue.pop(0)  # 如果序列已满，移除最早的图片
                    queue.append(np.array(img_gray))  # 添加处理后的图片到序列

            if queue:
                while len(queue) < seq_length:
                    queue.append(np.zeros((128,128)))
                prediction = Micro_predict_emotion(queue, self.model)
                emotion = self.label_map[np.argmax(prediction)]
                img = cv2.imread(imagePath)
                # 画出人脸矩形框和表情预测值
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, emotion, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # 将处理后的OpenCV图像转换为适用于QLabel显示的QPixmap
                height, width, channel = img.shape
                bytesPerLine = 3 * width
                qImg = QtGui.QImage(img.data, width, height, bytesPerLine, QtGui.QImage.Format_BGR888)
                pixmap = QtGui.QPixmap.fromImage(qImg)

                # 显示图像
                self.ui.showLabel.setPixmap(pixmap.scaled(self.ui.showLabel.size(), QtCore.Qt.KeepAspectRatio))
                
                # 显示预测结果
                displayPrediction(self, prediction)
            
            # 没有检测到人脸则直接显示原图像
            else:
                pixmap = QtGui.QPixmap(imagePath)
                self.ui.showLabel.setPixmap(pixmap.scaled(self.ui.showLabel.size(), QtCore.Qt.KeepAspectRatio)) 

                label_text_item = QtWidgets.QGraphicsTextItem("没有识别到相关表情，请重新上传")
                label_text_item.setPos(-40, 0)  # 设置文本位置
                label_text_item.setFont(QFont("Arial", 14))
                self.scene.addItem(label_text_item)

    # 上传图片模块
    def uploadImage(self):
        # 清空之前的场景内容
        self.scene.clear()

        # 图片上传逻辑
        # 打开文件对话框选择图片
        imagePath, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)")
        if imagePath:
            # 显示选择的图片
            # pixmap = QtGui.QPixmap(imagePath)
            # self.ui.showLabel.setPixmap(pixmap.scaled(self.ui.showLabel.size(), QtCore.Qt.KeepAspectRatio))  # 修改这里

            # 判断是否检测到人脸
            is_dec = face_dec(imagePath)  
            if is_dec:
                img = cv2.imread(imagePath)
                for i in range(len(is_dec)):
                    # 图像预处理
                    processed_image = process_image(imagePath, is_dec, i=i)
                    img_gray, x1, y1, x2, y2 = processed_image[0], processed_image[1], processed_image[2], processed_image[3], processed_image[4]
                    if self.isMirco:
                        # 调用微表情模型方法进行预测
                        prediction = Micro_Pic_predict_emotion(model=self.model, image=img_gray)
                    else:
                        # 调用宏观模型方法进行预测
                        prediction = predict_emotion(self.model, img_gray)
                    emotion = self.label_map[np.argmax(prediction)]
                    # emotion = predict_emotion(self.model, img_gray)

                    # 画出人脸矩形框和表情预测值
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, emotion, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    # 将处理后的OpenCV图像转换为适用于QLabel显示的QPixmap
                    height, width, channel = img.shape
                    bytesPerLine = 3 * width
                    qImg = QtGui.QImage(img.data, width, height, bytesPerLine, QtGui.QImage.Format_BGR888)
                    pixmap = QtGui.QPixmap.fromImage(qImg)

                # 显示图像
                self.ui.showLabel.setPixmap(pixmap.scaled(self.ui.showLabel.size(), QtCore.Qt.KeepAspectRatio))
                
                # 显示预测结果
                displayPrediction(self, prediction)
            
            # 没有检测到人脸则直接显示原图像
            else:
                pixmap = QtGui.QPixmap(imagePath)
                self.ui.showLabel.setPixmap(pixmap.scaled(self.ui.showLabel.size(), QtCore.Qt.KeepAspectRatio)) 

                label_text_item = QtWidgets.QGraphicsTextItem("没有识别到相关表情，请重新上传")
                label_text_item.setPos(-40, 0)  # 设置文本位置
                label_text_item.setFont(QFont("Arial", 14))
                self.scene.addItem(label_text_item)
                
    # 上传视频模块
    def uploadVideo(self):
        self.scene.clear()
        
        # 视频上传逻辑
        videoPath, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi);;All Files (*)")
        if videoPath:
            # 读取视频
            self.cap = cv2.VideoCapture(videoPath)
            self.timer.timeout.connect(self.displayVideoFrame)
            self.timer.start(20)  # 每20毫秒读取一帧

    #实时识别模块
    def realTimeRecognition(self):
        self.scene.clear()
        global frame_queue 
        frame_queue = deque(maxlen=seq_length)
        self.sequence_ready = False

        # 实时识别逻辑
        if not self.isRealTimeRecognitionActive:
            # 启动实时识别
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QtWidgets.QMessageBox.warning(self, "Error", "Cannot open camera.")
                return
            self.timer.timeout.connect(self.displayVideoFrame)
            self.timer.start(20)
            self.isRealTimeRecognitionActive = True
        # 再次点击实时识别则关闭摄像头
        else:
            # 停止实时识别
            self.timer.stop()
            if self.cap and self.cap.isOpened():
                self.cap.release()
            # self.sequence_ready = False
            # 清除 QLabel 的图像
            self.ui.showLabel.clear()
            self.ui.showLabel.setText("摄像头已关闭")  # 可选：在 QLabel 中显示提示文本
            self.scene.clear()  # 清除预测结果的文本

            self.isRealTimeRecognitionActive = False

    def displayVideoFrame(self):
        # 初始化一个固定大小的队列来存储连续帧
        global frame_queue

        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.cap.release()
            return

        frame = cv2.flip(frame, 1)
        # 在这里添加图像处理的代码，例如 cv2.rectangle 和 cv2.putText
        is_dec = face_dec(frame)
        if is_dec:
            # 清空之前的场景内容
            self.scene.clear()
            for i in range(len(is_dec)):
                processed_image = process_image(None , is_dec, image=frame, i=i)
                img_gray, x1, y1, x2, y2 = processed_image[0], processed_image[1], processed_image[2], processed_image[3], processed_image[4]
                #画出人脸区域
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # 添加处理后的帧到队列
                frame_queue.append(np.array(img_gray))
                if len(frame_queue) == seq_length:
                    self.sequence_ready = True

                if self.sequence_ready:
                    # 判断是否是微表情
                    if self.isMirco:
                        # 调用微表情模型方法进行预测
                        prediction = Micro_predict_emotion(frame_queue, self.model)
                    else:
                        # 调用宏观表情方法方法进行预测
                        prediction = predict_emotion(image=img_gray, model= self.model)
                    emotion = self.label_map[np.argmax(prediction)]
                    
                    cv2.putText(frame, emotion, (x1+30, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    prediction= [[0, 0, 0, 0, 0]]
            # 显示预测的表情结果
            displayPrediction(self, prediction)
        else:
            self.scene.clear()
            label_text_item = QtWidgets.QGraphicsTextItem("没有检测到人脸")
            label_text_item.setPos(-40, 0)  # 设置文本位置
            label_text_item.setFont(QFont("Arial", 14))
            self.scene.addItem(label_text_item)
        # 将OpenCV图像（BGR格式）转换为QImage
        height, width, channel = frame.shape
        bytesPerLine = 3 * width
        qImg = QtGui.QImage(frame.data, width, height, bytesPerLine, QtGui.QImage.Format_BGR888)

        # 将QImage转换为QPixmap并显示在QLabel中
        pixmap = QtGui.QPixmap.fromImage(qImg)
        self.ui.showLabel.setPixmap(pixmap.scaled(self.ui.showLabel.size(), QtCore.Qt.KeepAspectRatio))

    
    # 确保在应用关闭时停止定时器并释放资源
    def closeEvent(self, event):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        super(EmotionRecognitionApp, self).closeEvent(event)


# 使用EmotionRecognitionApp类
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = EmotionRecognitionApp()
    main_window.show()
    sys.exit(app.exec_())


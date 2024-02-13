from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.cm as cm
from IPython.display import Image
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.models import * 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
from PyQt5.uic import loadUi
class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1122, 837)
        self.widget = QtWidgets.QWidget(Dialog)
        self.widget.setGeometry(QtCore.QRect(0, 0, 1121, 841))
        self.widget.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.widget.setStyleSheet("background-color:rgb(245, 245, 245);\n"
"")
        self.widget.setObjectName("widget")
        self.label_4 = QtWidgets.QLabel(self.widget)
        self.label_4.setGeometry(QtCore.QRect(260, 70, 821, 111))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.Image = QtWidgets.QLabel(self.widget)
        self.Image.setGeometry(QtCore.QRect(330, 220, 281, 321))
        self.Image.setFrameShape(QtWidgets.QFrame.Box)
        self.Image.setFrameShadow(QtWidgets.QFrame.Plain)
        self.Image.setText("")
        self.Image.setScaledContents(True)
        self.Image.setObjectName("Image")
        self.label_6 = QtWidgets.QLabel(self.widget)
        self.label_6.setGeometry(QtCore.QRect(340, 550, 181, 41))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.cscore = QtWidgets.QLabel(self.widget)
        self.cscore.setGeometry(QtCore.QRect(340, 680, 231, 41))
        self.cscore.setFrameShape(QtWidgets.QFrame.Box)
        self.cscore.setFrameShadow(QtWidgets.QFrame.Plain)
        self.cscore.setText("")
        self.cscore.setObjectName("cscore")
        self.outcome = QtWidgets.QLabel(self.widget)
        self.outcome.setGeometry(QtCore.QRect(340, 590, 231, 41))
        self.outcome.setFrameShape(QtWidgets.QFrame.Box)
        self.outcome.setFrameShadow(QtWidgets.QFrame.Plain)
        self.outcome.setLineWidth(1)
        self.outcome.setText("")
        self.outcome.setObjectName("outcome")
        self.upload = QtWidgets.QPushButton(self.widget)
        self.upload.setGeometry(QtCore.QRect(130, 250, 181, 51))
        self.upload.setStyleSheet("font: 75 11pt \"MS Shell Dlg 2\";")
        self.upload.setDefault(False)
        self.upload.setFlat(False)
        self.upload.setObjectName("upload")
        self.detect = QtWidgets.QPushButton(self.widget)
        self.detect.setGeometry(QtCore.QRect(140, 360, 161, 51))
        self.detect.setStyleSheet("font: 75 11pt \"MS Shell Dlg 2\";")
        self.detect.setObjectName("detect")
        self.reset = QtWidgets.QPushButton(self.widget)
        self.reset.setGeometry(QtCore.QRect(170, 480, 111, 31))
        self.reset.setStyleSheet("font: 75 11pt \"MS Shell Dlg 2\";")
        self.reset.setObjectName("reset")
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setGeometry(QtCore.QRect(340, 640, 181, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setGeometry(QtCore.QRect(380, 770, 591, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setGeometry(QtCore.QRect(410, 790, 491, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_7 = QtWidgets.QLabel(self.widget)
        self.label_7.setGeometry(QtCore.QRect(90, 100, 61, 61))
        self.label_7.setText("")
        self.label_7.setPixmap(QtGui.QPixmap(r"C:\Users\biomedical research\Desktop\CoE Research Works\coe logo900.jpeg"))
        self.label_7.setScaledContents(True)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.widget)
        self.label_8.setGeometry(QtCore.QRect(350, 170, 291, 41))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.Image_2 = QtWidgets.QLabel(self.widget)
        self.Image_2.setGeometry(QtCore.QRect(650, 220, 281, 321))
        self.Image_2.setFrameShape(QtWidgets.QFrame.Box)
        self.Image_2.setFrameShadow(QtWidgets.QFrame.Plain)
        self.Image_2.setText("")
        self.Image_2.setScaledContents(True)
        self.Image_2.setObjectName("Image_2")
        self.label_9 = QtWidgets.QLabel(self.widget)
        self.label_9.setGeometry(QtCore.QRect(710, 170, 291, 41))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.line = QtWidgets.QFrame(self.widget)
        self.line.setGeometry(QtCore.QRect(40, 40, 1051, 16))
        self.line.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line.setLineWidth(2)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(self.widget)
        self.line_2.setGeometry(QtCore.QRect(30, 50, 20, 771))
        self.line_2.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line_2.setLineWidth(2)
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setObjectName("line_2")
        self.line_3 = QtWidgets.QFrame(self.widget)
        self.line_3.setGeometry(QtCore.QRect(40, 810, 1051, 16))
        self.line_3.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line_3.setLineWidth(2)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setObjectName("line_3")
        self.line_4 = QtWidgets.QFrame(self.widget)
        self.line_4.setGeometry(QtCore.QRect(1080, 50, 20, 771))
        self.line_4.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line_4.setLineWidth(2)
        self.line_4.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_4.setObjectName("line_4")
        self.label_10 = QtWidgets.QLabel(self.widget)
        self.label_10.setGeometry(QtCore.QRect(650, 550, 181, 41))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.about = QtWidgets.QLabel(self.widget)
        self.about.setGeometry(QtCore.QRect(650, 590, 281, 131))
        self.about.setFrameShape(QtWidgets.QFrame.Box)
        self.about.setFrameShadow(QtWidgets.QFrame.Plain)
        self.about.setText("")
        self.about.setObjectName("about")
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label_4.setText(_translate("Dialog", "        KNEE OSTEOARTHRITIS DETECTION \n"
"                 AND KL GRADING SYSTEM"))
        self.label_6.setText(_translate("Dialog", "Predicted grade"))
        self.upload.setText(_translate("Dialog", "UPLOAD KNEE \n"
" X-RAY IMAGE"))
        self.detect.setText(_translate("Dialog", "PREDICT \n"
" PATHOLOGY"))
        self.reset.setText(_translate("Dialog", "RESET"))
        self.label_3.setText(_translate("Dialog", "Severity"))
        self.label.setText(_translate("Dialog", "CENTER OF EXCELLENCE IN MEDICAL IMAGING"))
        self.label_2.setText(_translate("Dialog", "RAJALAKSHMI ENGINEERING COLLEGE"))
        self.label_8.setText(_translate("Dialog", "Input knee x-ray image"))
        self.label_9.setText(_translate("Dialog", "Generated heatmap "))
        self.label_10.setText(_translate("Dialog", "About the severity"))
        self.upload.clicked.connect(self.loadimage)
        self.detect.clicked.connect(self.predictimage)
        self.tmp=None
        self.ab=None
        self.out=None
        self.model=tf.keras.models.load_model(r"C:\Users\biomedical research\Desktop\CoE Research Works\project works\Knee osteoarthritis\kl grading model.h5")
    def loadimage(self):
            self.filename1 = QtWidgets.QFileDialog.getOpenFileName(filter='Image (*.*)')
            self.filename = self.filename1[0]
            print(self.filename)
            self.outcome.setText("Image Loaded!!")
            self.image = cv2.imread(self.filename)
            self.last_conv_layer_name='conv2d'
            self.setPhoto(self.image)

    def setPhoto(self, image):
            self.tmp = image
            image = cv2.resize(image, (256, 256), cv2.INTER_AREA)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.ab=image
            image = QtGui.QImage(image.data, image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888)
            self.Image.setPixmap(QPixmap.fromImage(image))

    def predictimage(self):
            class_names = ['0','1','2','3','4']
            img = self.image
            img=cv2.resize(img,(224,224))
            imag = image.img_to_array(img)
            imaga = np.expand_dims(imag,axis=0)
            self.tmp1 = cv2.resize(self.tmp, (224, 224))
            self.tmp1 = np.expand_dims(self.tmp1, axis=0)
            # self.tmp = self.tmp / 255

            self.predictions = self.model.predict(self.tmp1)
            pred_labels = np.argmax(self.predictions, axis=1)
            if pred_labels==0:
                a="Normal"
                b="Healthy knee image"
            elif pred_labels==1:
                a='Doubtful'
                b='Doubtful joint narrowing \n with possible osteophytic lipping'
            elif pred_labels==2:
                a='Mild'
                b='Definite presence of osteophytes \n and possible joint space narrowing'
            elif pred_labels==3:
                a="Moderate"
                b='Multiple osteophytes,\n definite joint space narrowing,\n with mild sclerosis.'
            else:
                a='Severe'
                b='Large osteophytes, \n significant joint narrowing, \n and severe sclerosis.'
            self.outcome.setText("Model predict: {} ".format(class_names[int(pred_labels)]))
            self.cscore.setText("{}".format(a))
            self.about.setText("{}".format(b))
            def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
                array = tf.keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
                array = np.expand_dims(array, axis=0)
                return array


            def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
                grad_model = tf.keras.models.Model(
                    [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
                )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
                with tf.GradientTape() as tape:
                    last_conv_layer_output, preds = grad_model(img_array)
                    if pred_index is None:
                        pred_index = tf.argmax(preds[0])
                    class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
                grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
                last_conv_layer_output = last_conv_layer_output[0]
                heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
                heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
                heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
                return heatmap.numpy()
            def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
                img = self.ab
                
    # Rescale heatmap to a range 0-255
                heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
                jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
                jet_colors = jet(np.arange(256))[:, :3]
                jet_heatmap = jet_colors[heatmap]
    # Create an image with RGB colorized heatmap
                jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
                jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
                jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
                #jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    # Superimpose the heatmap on original image
                superimposed_img = jet_heatmap * alpha + img
                superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
                self.out=superimposed_img
    # Save the superimposed image
                superimposed_img.save(cam_path)
    # Display Grad CAM
                display(Image(cam_path))
            heatmap = make_gradcam_heatmap(imaga, self.model, self.last_conv_layer_name)
            out_map=save_and_display_gradcam(imaga, heatmap)
            out_map=self.out
            out_map=np.array(out_map)
            out_map = QtGui.QImage(out_map.data,out_map.shape[1], out_map.shape[0], QtGui.QImage.Format_RGB888)
            self.Image_2.setPixmap(QPixmap.fromImage(out_map))
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

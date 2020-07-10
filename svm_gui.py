# encoding=utf8
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
from sklearn import metrics
#thuvienTkinter
import os
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
from tkinter import Label
from tkinter import *
from tkinter.messagebox import showinfo
from tkinter.filedialog import askopenfile
from PIL import Image
from PIL import ImageTk

#load du lieu tai ve trong thu muc mldata
(X_train,y_train),(X_test,y_test) = mnist.load_data()
print(X_train.shape, y_train.shape)

#cho x_train
X_train_feature = []
for i in range(len(X_train)):
    #so luong o co kich thuoc 14x14, vecto dinh huong 9, co 4 khoi tren o co kich thuoc 14x14, vecto dac trung hog co kich thuoc 4*9=36
    feature = hog(X_train[i],orientations=9,pixels_per_cell=(14,14),cells_per_block=(1,1),block_norm="L2")
    X_train_feature.append(feature)
X_train_feature = np.array(X_train_feature,dtype = np.float32)

#cho x_test
X_test_feature = []
for i in range(len(X_test)):
    feature = hog(X_test[i],orientations=9,pixels_per_cell=(14,14),cells_per_block=(1,1),block_norm="L2")
    X_test_feature.append(feature)
X_test_feature = np.array(X_test_feature,dtype=np.float32)

#default kernel='rbf', default gama='auto'
model = LinearSVC(C=5)
model.fit(X_train_feature,y_train)
y_pre = model.predict(X_test_feature)

#do chinh xac
print("Accuracy:")
print(accuracy_score(y_test,y_pre))

#cac chi so recall, precision, f1-score
print("Result:")
cl_report = metrics.classification_report(y_test, y_pre)
print(cl_report)

 
 
class Application(ttk.Frame):

    def __init__(self, master):
        ttk.Frame.__init__(self, master)
        self.pack()
        self.button_bonus = ttk.Button(self, text="Chọn ảnh nhận diện", command=display_selected)
        self.button_bonus.pack()
        self.button_Exit = ttk.Button(self, text="Thoát", command=root.destroy)
        self.button_Exit.pack()
 

def display_selected():
    root = Tk()
    #chon file anh
    root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Chọn file ảnh",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    image = cv2.imread(root.filename)
    #covert sang gray color
    im_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #giam nhieu bang GaussianBlur
    im_blur = cv2.GaussianBlur(im_gray,(5,5),0)
    #chuyen ve anh nhi phan
    im,thre = cv2.threshold(im_blur,90,255,cv2.THRESH_BINARY_INV)
    #tim contour va ve bouding box
    contours,hierachy = cv2.findContours(thre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(cnt) for cnt in contours]
    for i in contours:
        (x,y,w,h) = cv2.boundingRect(i)
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
        roi = thre[y:y+h,x:x+w]
        roi = np.pad(roi,(20,20),'constant',constant_values=(0,0))
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        
        # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1),block_norm="L2")
        nbr = model.predict(np.array([roi_hog_fd], np.float32))
        cv2.putText(image, str(int(nbr[0])), (x, y),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
        cv2.imshow("image",image)
    cv2.imwrite("image_pand.jpg",image)

#title gui
root = tk.Tk()
root.title("Nhận diện kí tự số với SVM")

#anh nen giao dien
image = Image.open("bg1.png")
photo = ImageTk.PhotoImage(image)
label_header = Label(image=photo,width=100,height=300)
label_header.pack(fill=BOTH, pady=5)

#Gioi thieu de tai
labelGT = Label(text="BÁO CÁO MÁY HỌC NÂNG CAO", fg="red", width=20, font=("Arial Bold", 15))
labelGT.pack(fill=BOTH, padx=20, pady=10)

labelGT = Label( text="NHẬN DIỆN KÍ TỰ SỐ VỚI SVM", width=20, font=("Arial Bold", 20))
labelGT.pack(fill=BOTH, padx=50, pady=10)

#Giang vien huong dan
'''
labelGV= Label(text="Giảng viên hướng dẫn", width=20, font=("Arial Bold",10))
labelGV.pack(fill=BOTH, padx=50, pady=10)
'''
labelGV= Label(text="PGS.TS Phạm Nguyên Khang", width=20, font=("Arial Bold",15))
labelGV.pack(fill=BOTH, padx=50, pady=10)

#so thu tu nhom
labelNhom= Label(text="NHÓM 05", width=20, font=("Arial Bold",15))
labelNhom.pack(fill=BOTH, padx=50, pady=10)

#thanh vien nhom 05
labelNhom= Label(text="B1609830 - Lê Thanh Lương", width=20, font=("Arial Bold",15))
labelNhom.pack(fill=BOTH, padx=50, pady=10)

labelNhom= Label(text="B1611128 - Lâm Thanh Hòa", width=20, font=("Arial Bold",15))
labelNhom.pack(fill=BOTH, padx=50, pady=10)

#pink background
#root.configure(background = 'AntiqueWhite1')
root.geometry ('1200x1200')
app = Application(root)
root.mainloop()
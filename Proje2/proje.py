# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 15:47:13 2021

@author: erent
"""
import cv2
import time
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.QtWidgets import *
from PyQt5.uic import *
from PyQt5.Qt import QApplication, QUrl, QDesktopServices
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt
from PyQt5 import uic
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout,QDesktopWidget, QWidget,QTableWidget,QTableView,QTableWidgetItem,QHeaderView,QGraphicsScene,QGraphicsPixmapItem,QFileDialog
from PyQt5 import QtCore, QtGui, QtWidgets
import xlwt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
import sys
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import random
import seaborn as sns
from pandas import DataFrame
from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier 
from skimage.feature import daisy
import os,shutil
from skimage.transform import resize
from sklearn.preprocessing import MinMaxScaler
import pathlib
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
import pickle
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from matplotlib import pyplot as plt
from keras.datasets import mnist
import joblib
import csv
from csv import writer
import random
from keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D, Dropout
from keras.models import Model, load_model, Sequential

class window(QMainWindow):
    
    def __init__(self):
        super(window, self).__init__()
        loadUi("projearayuz.ui", self)
        self.pushButton_2.clicked.connect(self.Ekle)
        self.pushButton_5.clicked.connect(self.VeriSetiGetir)
        self.pushButton.clicked.connect(self.ogrenimveriseti)
        self.cnnbtn.clicked.connect(self.derinOgrenme)
        self.arabtn.clicked.connect(self.ara)
        self.silbtn.clicked.connect(self.delete)
        self.duzenlebtn.clicked.connect(self.duzenle)
        self.sorgulabtn.clicked.connect(self.sorgula)
        self.pushButton_3.clicked.connect(self.sorguislem)
        
        self.show()
        
    def Ekle(self):
        #Nesne İçerikleri
        mid = self.textEdit_17.toPlainText()
        changer=self.CinsiyetCeviri(self.comboBox.currentText())
        cinsiyet = changer
        changer=self.numberdegis(self.comboBox_2.currentText())
        yasli = changer
        changer=self.YesNoDegistir(self.comboBox_5.currentText())
        evli=changer
        changer=self.YesNoDegistir(self.comboBox_6.currentText())
        ekonomikbagimli=changer
        text = self.textEdit_7.toPlainText()
        aboneliksure=text
        changer=self.YesNoDegistir(self.comboBox_7.currentText())
        telefonhizmeti=changer
        changer=self.YesNoDegistir(self.comboBox_8.currentText())
        birdenfazlahat=changer
        internetservisi=self.comboBox_3.currentText()
        changer=self.YesNoDegistir(self.comboBox_9.currentText())
        cevrimiciguvenlik=changer
        changer=self.YesNoDegistir(self.comboBox_10.currentText())
        cevrimiciyedekleme=changer
        changer=self.YesNoDegistir(self.comboBox_11.currentText())
        cihazkoruma=changer
        changer=self.YesNoDegistir(self.comboBox_12.currentText())
        teknikdestek=changer
        changer=self.YesNoDegistir(self.comboBox_13.currentText())
        onlinetele=changer
        changer=self.YesNoDegistir(self.comboBox_14.currentText())
        onlinefilm=changer
        changer=self.YesNoDegistir(self.comboBox_15.currentText())
        cevrimicifatura=changer  
        changer=self.OdemeYontem(self.comboBox_4.currentText())
        odemeyontemi=changer
        text = self.textEdit.toPlainText()
        sozlesmesure=text
        text = self.textEdit_2.toPlainText()
        toplamodeme=text
        text = self.textEdit_3.toPlainText()
        aylikodeme=text
        musterikaybi="Bilinmiyor"
        
        #Veri Girişi
        List=[mid,cinsiyet,yasli,evli,ekonomikbagimli,aboneliksure,telefonhizmeti,birdenfazlahat,internetservisi,cevrimiciguvenlik,cevrimiciyedekleme,cihazkoruma,teknikdestek ,onlinetele,onlinefilm,cevrimicifatura,odemeyontemi,sozlesmesure,toplamodeme,aylikodeme,musterikaybi]
             
        with open('veriseti.csv', 'a') as f_object:     
            writer_object = writer(f_object)
            writer_object.writerow(List)
            f_object.close()  
        # self.verisetigetir()    
        
        
        #Null verileri temizleme
        self.data = self.data.dropna(axis=0)
        c=len(self.data.columns)
        r=len(self.data.values)
        self.tableWidget.setColumnCount(c)
        self.tableWidget.setRowCount(r)
        colmnames=["id","Cinsiyet","Yaşlı","Evli","Ekonomik Bagımlı","Abonelik Süresi","Telefon Hizmeti","Birden Fazla Hat",
                   "İnternet Servisi","Çevrimiçi Güvenlik","Çevrimiçi Yedekleme","Cihaz Koruma","Teknik Destek","Televizyon",
                   "Film","Sözleşme Süresi","Çevrimiçi Fatura","Ödeme Yöntemi","Aylık Ödeme","Toplam Ödeme","Müşteri Kaybı"]
        self.tableWidget.setHorizontalHeaderLabels(colmnames)
        
        for i,row in enumerate(self.data):
            for j,cell in enumerate(self.data.values):
                      self.tableWidget.setItem(j,i, QtWidgets.QTableWidgetItem(str(cell[i])))
                      # self.data.info()
        print("bitti")   
        self.veriSetiGetirF('veriseti.csv')
        # print(self.data.isnull().sum())
    
        
    def OdemeYontem(self,deger):
        if (deger == "Kredi Kartı"):
            deger="Creditcard(automatic)"
        elif (deger == "Banka Transfer"):
            deger="Banktransfer(automatic)"
        elif (deger == "Elektronik Çek"):
            deger="Electroniccheck"    
        elif (deger == "Mail Çek"):
            deger="Mailedcheck"
        return deger
            
            
    def CinsiyetCeviri(self,deger):
        if (deger == "Erkek"):
            deger="Male"
        else:
            deger="Female"
        return deger
        
    def YesNoDegistir(self,deger):
        # deger1=self.YesNoDegistir(self.comboBox.currentText())
        # cinsiyet = deger1
        if(deger == "Evet"):
            deger = "Yes"
        else:
            deger = "No"
        return deger   
            
    def numberdegis(self,deger):
        # deger1=self.YesNoDegistir(self.comboBox.currentText())
        # cinsiyet = deger1
        if(deger == "Evet"):
            deger = "1"
        else:
            deger = "0"
        return deger
        
        
    def VeriSetiGetir(self):      
       file_name,_= QFileDialog.getOpenFileName(self, 'Open Image File', r".\Desktop")
       self.data = pd.read_csv(file_name)
       self.data = self.data.dropna(axis=0)
       c=len(self.data.columns)
       r=len(self.data.values)
       self.tableWidget.setColumnCount(c)
       self.tableWidget.setRowCount(r)
       colmnames=["id","Cinsiyet","Yaşlı","Evli","Ekonomik Bagımlı","Abonelik Süresi","Telefon Hizmeti","Birden Fazla Hat",
                   "İnternet Servisi","Çevrimiçi Güvenlik","Çevrimiçi Yedekleme","Cihaz Koruma","Teknik Destek","Televizyon",
                   "Film","Sözleşme Süresi","Çevrimiçi Fatura","Ödeme Yöntemi","Aylık Ödeme","Toplam Ödeme","Müşteri Kaybı"]
       self.tableWidget.setHorizontalHeaderLabels(colmnames)
       for i,row in enumerate(self.data):
             for j,cell in enumerate(self.data.values):
                  self.tableWidget.setItem(j,i, QtWidgets.QTableWidgetItem(str(cell[i])))
    def veriSetiGetirF(self,file_name):
       self.data = pd.read_csv(file_name)
       self.data = self.data.dropna(axis=0)
       c=len(self.data.columns)
       r=len(self.data.values)
       self.tableWidget.setColumnCount(c)
       self.tableWidget.setRowCount(r)
       colmnames=["id","Cinsiyet","Yaşlı","Evli","Ekonomik Bagımlı","Abonelik Süresi","Telefon Hizmeti","Birden Fazla Hat",
                   "İnternet Servisi","Çevrimiçi Güvenlik","Çevrimiçi Yedekleme","Cihaz Koruma","Teknik Destek","Televizyon",
                   "Film","Sözleşme Süresi","Çevrimiçi Fatura","Ödeme Yöntemi","Aylık Ödeme","Toplam Ödeme","Müşteri Kaybı"]
       self.tableWidget.setHorizontalHeaderLabels(colmnames)
       for i,row in enumerate(self.data):
             for j,cell in enumerate(self.data.values):
                  self.tableWidget.setItem(j,i, QtWidgets.QTableWidgetItem(str(cell[i])))
                  
    #Sayfa2 Eğitim          
    def ogrenimveriseti(self):              
          file_name,_= QFileDialog.getOpenFileName(self, 'Open Image File', r".\Desktop")
          self.data2 = pd.read_csv(file_name)
          self.data2 = self.data2.dropna(axis=0)  
          
                
    def onHazirlik(self):
        self.labeEnc()
        self.aykiriVeri()           
        x = self.encoded.drop('Churn', axis = 1)              
        y = self.encoded['Churn'] 
        return x,y   
          
    def labeEnc(self):
        # print("veri",len(self.veriler))
        from sklearn.preprocessing import LabelEncoder
        self.encoded = self.data2.apply(lambda x: LabelEncoder().fit_transform(x) if x.dtype == 'object' else x)     

    def aykiriVeri(self):
        Müşteri_Kaybı_Yaşandı=self.encoded.loc[self.encoded['Churn'].abs()>0]
        # print(Müşteri_Kaybı_Yaşandı)
        Q1 = Müşteri_Kaybı_Yaşandı['TotalCharges'].quantile(0.25)
        Q3 = Müşteri_Kaybı_Yaşandı['TotalCharges'].quantile(0.75)
        IQR = Q3 - Q1
        Q=Q3+(1.5*IQR)  
        encoded_out = self.encoded[~((self.encoded['TotalCharges'] < (Q3 + 1.5 * IQR)))&(self.encoded['Churn']>0)]
        # print(encoded_out.head(8000))
        # Aykırı veriler çağır
        self.encoded.drop(self.encoded[~((self.encoded['TotalCharges'] < (Q3 + 1.5 * IQR)))&(self.encoded['Churn']>0)].index, inplace=True)
        # print(self.encoded.head(8000))            
        Q1_A = Müşteri_Kaybı_Yaşandı['tenure'].quantile(0.25)
        Q3_A = Müşteri_Kaybı_Yaşandı['tenure'].quantile(0.75)
        IQR_A = Q3_A - Q1_A
        # print( IQR_A)
        Q_A=Q3_A+(1.5*IQR_A)
        # print(Q_A)
        encoded_A_out = self.encoded[~((self.encoded['tenure'] < (Q3_A + 1.5 * IQR_A)))&(self.encoded['Churn']>0)]
        # print(encoded_A_out.head(8000))
        self.encoded.drop(self.encoded[~((self.encoded['tenure'] < (Q3_A + 1.5 * IQR_A)))&(self.encoded['Churn']>0)].index, inplace=True)
        # print("enc",len(self.encoded))
   
    def onHazirlik2(self):
        self.labeEnc2()
        self.aykiriVeri2()           
        x2 = self.encoded2.drop('Musterikaybi', axis = 1)              
        y2 = self.encoded2['Musterikaybi'] 
        return x2,y2   
          
    def labeEnc2(self):
        # print("veri",len(self.veriler))
        from sklearn.preprocessing import LabelEncoder
        self.encoded2 = self.data.apply(lambda x: LabelEncoder().fit_transform(x) if x.dtype == 'object' else x)     

    def aykiriVeri2(self):
        Müşteri_Kaybı_Yaşandı=self.encoded2.loc[self.encoded2['Musterikaybi'].abs()>0]
        # print(Müşteri_Kaybı_Yaşandı)
        Q1 = Müşteri_Kaybı_Yaşandı['ToplamOdeme'].quantile(0.25)
        Q3 = Müşteri_Kaybı_Yaşandı['ToplamOdeme'].quantile(0.75)
        IQR = Q3 - Q1
        Q=Q3+(1.5*IQR)  
        encoded_out = self.encoded2[~((self.encoded2['ToplamOdeme'] < (Q3 + 1.5 * IQR)))&(self.encoded2['Musterikaybi']>0)]
        # print(encoded_out.head(8000))
        # Aykırı veriler çağır
        self.encoded2.drop(self.encoded2[~((self.encoded2['ToplamOdeme'] < (Q3 + 1.5 * IQR)))&(self.encoded2['Musterikaybi']>0)].index, inplace=True)
        # print(self.encoded.head(8000))            
        Q1_A = Müşteri_Kaybı_Yaşandı['SozlesmeSuresi'].quantile(0.25)
        Q3_A = Müşteri_Kaybı_Yaşandı['SozlesmeSuresi'].quantile(0.75)
        IQR_A = Q3_A - Q1_A
        # print( IQR_A)
        Q_A=Q3_A+(1.5*IQR_A)
        # print(Q_A)
        encoded_A_out = self.encoded2[~((self.encoded2['SozlesmeSuresi'] < (Q3_A + 1.5 * IQR_A)))&(self.encoded2['Musterikaybi']>0)]
        # print(encoded_A_out.head(8000))
        self.encoded2.drop(self.encoded2[~((self.encoded2['SozlesmeSuresi'] < (Q3_A + 1.5 * IQR_A)))&(self.encoded2['Musterikaybi']>0)].index, inplace=True)
        # print("enc",len(self.encoded))    
   
                
    def derinOgrenme(self):
        x,y=self.onHazirlik()
        x2,y2=self.onHazirlik2()
        # test_size1=float(self.lineEdit.text())
        x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, test_size = 0.1 , random_state = 42)

        x_train, x_test, y_train, y_test = train_test_split(x2, y2, test_size = 0.9 , random_state = 42)

        # import
        from keras.utils import to_categorical 
        y_train1 = to_categorical(y_train1, 2)
        y_test= to_categorical(y_test, 2)
        from keras.models import Sequential
        from keras.layers import Dense,Dropout,BatchNormalization,Activation
        
        #Model Oluşturma
        model = Sequential()
        
        #Öznitelik Saısı belirleme
        n_cols = x_train1.shape[1]
        
        #Model Katmanı Ekleme
        model.add(Dense(16, input_shape=(n_cols,)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(9))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(6))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        model.summary()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])       
        history =model.fit(x_train1, 
        y_train1,
        validation_data=(x_test, y_test),
        # bath kullanici seçicek
        batch_size=int(self.bstb_4.text()), 
        shuffle=True,
        verbose=1,
        epochs=int(self.eptb_3.text()))
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        model.save('model', overwrite=True)
        # Graflar
        plt.subplot(1, 2, 1)              
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel(str(round(score[1]*100,3)))
        plt.legend(['Train', 'Test'], loc='upper left')   
        plt.subplot(1, 2, 2)      
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel(str(round(score[0],3)))
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig("accloss.png")
        self.pixmap = QPixmap("accloss.png")
        self.label_37.setPixmap(self.pixmap)
        plt.show()  
        print('----Sonuç-----')  
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        y_pred = model.predict(x_test)
        y_test = y_test.reshape(-1, 1)
        print(y_test)
        self.textEdit_4.setText(str(y_test))
        y_pred=y_pred.reshape(-1, 1)
        y_pred=y_pred.round()
        self.textEdit_6.setText(str(y_pred))
        print(y_pred)
        self.confmat(y_test,y_pred,"cnfmat")
        self.textEdit_9.setText(str(history.history['accuracy']) + "\n" + str(history.history['loss']))
        self.textEdit_10.setText(str(score[1]))
        self.textEdit_5.setText(str(confusion_matrix(y_test, y_pred.round())))
        # y_pred2=y_pred.round()
        # self.Cmatrix(y_test,y_pred2,"Derin Öğrenme")
        # self.pltRoc2(y_test,y_pred,"Derin Öğrenme")          
        print("--------------------")
        
    def ara(self):
        data =[]
        with open("veriseti.csv") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                data.append(row)
        print(data)
        name=self.textEdit_8.toPlainText()
        col=[x[0] for x in data]
        if name in col:
            for x in range(0,len(data)):
                if name == data[x][0]:
                    print(data[x])
                    deger = data[x]
                    return deger
        else:
            print("Bulunamadı")
    
    
    def sorguara(self,deger):
        data =[]
        with open("veriseti.csv") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                data.append(row)
        print(data)
        name=self.textEdit_16.toPlainText()
        col=[x[0] for x in data]
        if name in col:
            for x in range(0,len(data)):
                if name == data[x][0]:
                    print(data[x])
                    deger = data[x]
                    return deger
        else:
            print("Bulunamadı")
    
        
    def duzenle(self):
        name=self.textEdit_17.toPlainText()
        self.deleteClick(name)
        self.Ekle()
        
    def delete(self):
        name=self.textEdit_17.toPlainText()
        self.deleteClick(name)
        
        
    def deleteClick(self,name):
        lines = list()
        with open('veriseti.csv', 'r') as readFile:
            reader = csv.reader(readFile)
            for row in reader:
                lines.append(row)
                for field in row:
                    if field == name:
                        lines.remove(row)
        with open('veriseti.csv', 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(lines)
        self.veriSetiGetirF('veriseti.csv')
        
    def confmat(self,y_test,y_pred,isim):
        cm = confusion_matrix(y_test, y_pred)
        cm_data = pd.DataFrame(cm)
        plt.figure(figsize = (5,5))
        sns.heatmap(cm_data, annot=True,fmt="d")
        plt.title(isim)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        pathlib.Path('./PCAs').mkdir(parents=True, exist_ok=True)
        plt.savefig("./cnfmat.png")
        self.pixmap = QPixmap("./cnfmat.png")
        self.label_38.setPixmap(self.pixmap)
        plt.show()       
  
    
    def sorgula(self):  
        self.sorgumain()
        self.label_24.setText(str(self.pred))
        print(self.pred)
        print(self.pred[0][0])
        self.sorgulasil(self.mid)
      
        
    def sorgumain(self):
        data =[]
        with open("veriseti.csv") as csvfile: 
            reader = csv.reader(csvfile)
            for row in reader:
                data.append(row)
        name=self.textEdit_16.toPlainText()
        col=[x[0] for x in data]
        if name in col:
            for x in range(0,len(data)):
                if name == data[x][0]:
                    print(data[x])
                    self.deger = data[x]
                    print (self.deger[0])
        else:
            print("Bulunamadı")
        self.mid=self.deger[0]
        print(self.mid)
        cinsiyet=self.deger[1]
        yasli=self.deger[2]
        evli=self.deger[3]
        ekonomikbagimli=self.deger[4]
        aboneliksure=self.deger[5]
        telefonhizmeti=self.deger[6]
        birdenfazlahat=self.deger[7]
        internetservisi=self.deger[8]
        cevrimiciguvenlik=self.deger[9]
        cevrimiciyedekleme=self.deger[10]
        cihazkoruma=self.deger[11]
        teknikdestek=self.deger[12]
        onlinetele=self.deger[13]
        onlinefilm=self.deger[14]
        cevrimicifatura=self.deger[15]
        odemeyontemi=self.deger[16]
        sozlesmesure=self.deger[17]
        toplamodeme=self.deger[18]
        aylikodeme=self.deger[19]
        musterikaybi=self.deger[20]
        List=[self.mid,cinsiyet,yasli,evli,ekonomikbagimli,aboneliksure,telefonhizmeti,birdenfazlahat,internetservisi,cevrimiciguvenlik,cevrimiciyedekleme,cihazkoruma,teknikdestek ,onlinetele,onlinefilm,cevrimicifatura,odemeyontemi,sozlesmesure,toplamodeme,aylikodeme,musterikaybi]
        with open('aranacak.csv', 'a') as f_object:     
            writer_object = writer(f_object)
            writer_object.writerow(List)
            f_object.close()
        self.aranacakdata = pd.read_csv('aranacak.csv')
        self.aranacakdata = self.aranacakdata.dropna(axis=0)
        x3=self.onHazirlik3()
        print(x3)
        modelRakam = load_model("model")
        self.pred = modelRakam.predict(x3)
        self.pred=self.pred.round()
        
    def sorguislem(self):
        self.sorgumain()
        while(self.pred < "1"):
            toplamEksilecek = 0
            aylikEksilecek = 0
            mid=self.deger[0]
            cinsiyet=self.deger[1]
            yasli=self.deger[2]
            evli=self.deger[3]
            ekonomikbagimli=self.deger[4]
            aboneliksure=self.deger[5]
            telefonhizmeti=self.deger[6]
            birdenfazlahat=self.deger[7]
            internetservisi=self.deger[8]
            cevrimiciguvenlik=self.deger[9]
            cevrimiciyedekleme=self.deger[10]
            cihazkoruma=self.deger[11]
            teknikdestek=self.deger[12]
            onlinetele=self.deger[13]
            onlinefilm=self.deger[14]
            cevrimicifatura=self.deger[15]
            odemeyontemi=self.deger[16]
            sozlesmesure=self.deger[17]
            toplamodeme=self.deger[18]
            degistoplam=int(toplamodeme)-10
            toplamEksilecek = toplamEksilecek + 10
            aylikodeme=self.deger[19]
            degisaylik=int(aylikodeme)-1
            aylikEksilecek = aylikEksilecek +1
            musterikaybi=self.deger[20]
            List=[mid,cinsiyet,yasli,evli,ekonomikbagimli,aboneliksure,telefonhizmeti,birdenfazlahat,internetservisi,cevrimiciguvenlik,cevrimiciyedekleme,cihazkoruma,teknikdestek ,onlinetele,onlinefilm,cevrimicifatura,odemeyontemi,sozlesmesure,degistoplam,degisaylik,musterikaybi]
            with open('aranacak.csv', 'a') as f_object:     
                writer_object = writer(f_object)
                writer_object.writerow(List)
                f_object.close()
            self.aranacakdata = pd.read_csv('aranacak.csv')
            self.aranacakdata = self.aranacakdata.dropna(axis=0)
            x3=self.onHazirlik3()
            print(x3)
            modelRakam = load_model("model")
            self.pred = modelRakam.predict(x3)
            self.pred= self.pred.round()
        print(str(self.pred))
        print(str(aylikEksilecek))
        print(str(toplamEksilecek))
        print("bitti")
        self.textEdit_11.setText(str(aylikEksilecek)+ "\n" + str(toplamEksilecek))
        
        
    def onHazirlik3(self):
        self.labeEnc3()
        self.aykiriVeri3()           
        x3 = self.encoded3.drop('Musterikaybi', axis = 1)              
        y3 = self.encoded3['Musterikaybi'] 
        return x3  
          
    def labeEnc3(self):
        # print("veri",len(self.veriler))
        from sklearn.preprocessing import LabelEncoder
        self.encoded3 = self.aranacakdata.apply(lambda x: LabelEncoder().fit_transform(x) if x.dtype == 'object' else x)     

    def aykiriVeri3(self):
        Müşteri_Kaybı_Yaşandı=self.encoded3.loc[self.encoded3['Musterikaybi'].abs()>0]
        # print(Müşteri_Kaybı_Yaşandı)
        Q1 = Müşteri_Kaybı_Yaşandı['ToplamOdeme'].quantile(0.25)
        Q3 = Müşteri_Kaybı_Yaşandı['ToplamOdeme'].quantile(0.75)
        IQR = Q3 - Q1
        Q=Q3+(1.5*IQR)  
        encoded_out = self.encoded3[~((self.encoded3['ToplamOdeme'] < (Q3 + 1.5 * IQR)))&(self.encoded3['Musterikaybi']>0)]
        # print(encoded_out.head(8000))
        # Aykırı veriler çağır
        self.encoded3.drop(self.encoded3[~((self.encoded3['ToplamOdeme'] < (Q3 + 1.5 * IQR)))&(self.encoded3['Musterikaybi']>0)].index, inplace=True)
        # print(self.encoded.head(8000))            
        Q1_A = Müşteri_Kaybı_Yaşandı['SozlesmeSuresi'].quantile(0.25)
        Q3_A = Müşteri_Kaybı_Yaşandı['SozlesmeSuresi'].quantile(0.75)
        IQR_A = Q3_A - Q1_A
        # print( IQR_A)
        Q_A=Q3_A+(1.5*IQR_A)
        # print(Q_A)
        encoded_A_out = self.encoded3[~((self.encoded3['SozlesmeSuresi'] < (Q3_A + 1.5 * IQR_A)))&(self.encoded3['Musterikaybi']>0)]
        # print(encoded_A_out.head(8000))
        self.encoded3.drop(self.encoded3[~((self.encoded3['SozlesmeSuresi'] < (Q3_A + 1.5 * IQR_A)))&(self.encoded3['Musterikaybi']>0)].index, inplace=True)
        # print("enc",len(self.encoded))
        
    def sorgulasil(self,name):
        lines = list()
        with open('aranacak.csv', 'r') as readFile:
            reader = csv.reader(readFile)
            for row in reader:
                lines.append(row)
                for field in row:
                    if field == name:
                        lines.remove(row)
        with open('aranacak.csv', 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(lines)
            
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = window()
    window.show()
    sys.exit(app.exec())
    
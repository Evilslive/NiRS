import re
import math
import sys

import numpy as np
import scipy
from scipy.signal import butter, lfilter, filtfilt, freqz, firwin #Band-Pass

import matplotlib.pyplot as plt

import tkinter as tk # GUI
from tkinter import ttk
from tkinter import messagebox as mb
from tkinter import simpledialog as sd # 輸入字串與數值
from tkinter import filedialog as fd # 讀取字串與數值, 以及開啟檔案路徑

import PySide2 as ps2  # <<< LGPL
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

class PreProcessing():
    def __init__(self):
        pass
    
    #---------
    # 參數提取
    #---------
    def references(self):
        '''選擇檔案, 並將數據轉為list,最後畫出圖'''
        # 開啟視窗
        loadFile = tk.Tk()
        loadFile.withdraw() # 隱藏tk視窗
        self.fileman = fd.askopenfilename() # 莫名的延遲
        self.fileman = self.fileman[self.fileman.find(":")-1:self.fileman.find(".")] # 檔案路徑, 去副檔名
        intro = open(self.fileman +'.hdr', 'r')
        info = intro.read()
        intro.close
        # 讀入所有的參數設定
        GeneralInfo = info[info.find("GeneralInfo") +12:info.find("ImagingParameters")-1]
        ImagingParameters = info[info.find("ImagingParameters")+18:info.find("Paradigm")-1]
        length1 = int(ImagingParameters[ImagingParameters.find("Wavelengths")+13:ImagingParameters.find("Wavelengths")+16])
        length2 = int(ImagingParameters[ImagingParameters.find("Wavelengths")+17:ImagingParameters.find("TrigIns")-2])
        self.Length = [length1, length2]
        self.Fs = float(ImagingParameters[ImagingParameters.find("SamplingRate")+13:ImagingParameters.find("Mod Amp")])
        Paradigm = info[info.find("Paradigm")+9:info.find("ExperimentNotes")-1]
        ExperimentNotes = info[info.find("ExperimentNotes")+16:info.find("GainSettings")-1]
        # Gains 矩陣
        GainSettings = info[info.find("GainSettings")+22:info.find("Markers")-5]
        self.GainSettings = list(map(int,re.findall(r"\d|-\d", GainSettings)))
        # triggers, Conditions (in frames)
        Markers = info[info.find("Markers")+19:info.find("DataStructure")-5]
        # 如果要產生str矩陣, Markers = Markers.strip("\n").strip("\t")
        Markers = Markers.split()
        # 拆分出frames、triggers、seconds
        self.seconds = []
        self.triggers = []
        self.frames = []
        for i in range(len(Markers)):
            if (i+1) % 3 == 0:
                self.frames.append(Markers[i])
            elif (i+2) % 3 == 0:
                self.triggers.append(Markers[i])
            else:
                self.seconds.append(Markers[i])
        # 資料結構
        DataStructure = info[info.find("DataStructure")+14:info.find("DarkNoise")-5]
        S_D_Key = re.split(r":|,", DataStructure[DataStructure.find("S-D-Key")+9:DataStructure.find("S-D-Mask")-3])  # 多重切割
        self.S_D_Keys = []
        S_D_Values = []
        for i in range(len(S_D_Key)):
            if i % 2 == 0:
                self.S_D_Keys.append('S'+S_D_Key[i].replace('-', "_"))
            else:
                S_D_Values.append('S'+S_D_Key[i].replace('-', "_"))  # 是否固定為序號?
        self.S_D_Mask = DataStructure[DataStructure.find("S-D-Mask")+11:len(DataStructure)].split()
        # Dark Noise
        DarkNoise = info[info.find("DarkNoise")+10:info.find("ChannelsDistance")-1]
        # Channels Distance
        ChannelsDistance = info[info.find("ChannelsDistance")+13:len(info)].split() # 為可用Channel的長度(d)
        # 讀取資料
        wl1 = open(self.fileman+'.wl1', 'r')
        wl2 = open(self.fileman+'.wl2', 'r')
        data1 = wl1.read().split()  # 總長度 = Channels(先) x frames(後)
        data2 = wl2.read().split()
        self.max_frames = int(len(data1) / len(self.S_D_Mask))  # 不是整數的防呆?
        self.total_channels = int(len(data1) / self.max_frames)
        wl1.close()
        wl2.close()
        # 維度轉換
        self.xLabel = []
        self.raw_wl1 = []
        self.raw_wl2 = []
        collect_1_DataPoint = []
        collect_2_DataPoint = []
        for i in range(self.total_channels):  # 512
            for j in range(self.max_frames): # 868
                if i == 0:
                    self.xLabel.append(j+1) # 產生x座標軸
                collect_1_DataPoint.append(data1[i + self.total_channels*j])
                collect_2_DataPoint.append(data2[i + self.total_channels*j])
                if j == self.max_frames - 1:
                    self.raw_wl1.append(list(map(float,collect_1_DataPoint))) # 結果, str轉為float
                    self.raw_wl2.append(list(map(float,collect_2_DataPoint)))
                    collect_1_DataPoint = []
                    collect_2_DataPoint = []
    #-------------
    # 全域線圖
    #-------------
    def raw_lineplot(self):
        self.raw = {
            "self.raw_wl1": ["self.Length[0]",1,"self.filtered_wl1"], 
            "self.raw_wl2": ["self.Length[1]",2,"self.filtered_wl2"]
            } # {原始線圖: [波長, 圖片分版的位置, 過濾線圖]}
        for key,value in self.raw.items():
            for i in range(len(self.S_D_Mask)):  # 0 1
                if self.S_D_Mask[i] == "1":
                    fig = plt.subplot(2,1,value[1])
                    plt.plot(self.xLabel, eval(key)[i])#, label=self.S_D_Keys[i]) # raw data
            fig.title.set_text("Available Channels ("+ str(eval(value[0])) +"nm)")
            plt.grid(True)
            plt.ylabel('Amplitude(V)')
        plt.xlabel('frames('+str(round(float(self.Fs),2))+'Hz)')
        plt.show()
    #----------------------------------
    # Discontinuities, Spike Artifacts
    #----------------------------------
    def head_displace(self):
        '''手動標示, 全頻道清除, 由最大的spike、兩側往中間開始處理, *不計算範圍內的triggers*'''
        reject_band = 0
        reject_area = []
        while reject_band != "z" or reject_band == None:
            temp0 = 0.00
            temp1 = 0.00
            temp2 = 0.00
            f0 = 0
            f1 = 0
            f2 = 0
            try:
                reject_band = input("輸入認定頭動的起始與結束點(frames, 為整數), 以斜線分隔如「28/95」, 結束輸入z：")
                reject1 = int(reject_band[0:reject_band.find("/")])
                reject2 = int(reject_band[reject_band.find("/")+1:len(reject_band)])
                reject_area.append(reject1) # <<<< 需要 排序 演算法
                reject_area.append(reject2)
                for raw in [self.raw_wl1,self.raw_wl2]:
                    for i in range(len(self.S_D_Mask)):
                        if reject1 < 2: # 防呆
                            reject1 = 2
                        if reject2 > self.max_frames:
                            reject2 = self.max_frames
                        if reject1 > reject2:
                            reject1, reject2 = reject2, reject1 # 數值互換
                        if reject2 > len(raw[i]):
                            reject1 = len(raw[i])
                        if self.S_D_Mask[i] == "1":
                            for j in range(0,reject1-1): # 前
                                temp1 += raw[i][j]
                                f1 += 1
                            for j in range(reject1-1,reject2-1): # 中
                                temp0 += raw[i][j]
                                f0 += 1
                            for j in range(reject2-1,len(raw[i])): # 後
                                temp2 += raw[i][j]
                                f2 += 1
                            rangeX = reject2 - reject1
                            rangeY = temp2/f2 - temp1/f1
                            slope = rangeY/rangeX # 結果論的斜率
                            if f1 >= f2 : # 後段適配前段
                                for j in range(reject2-1, len(raw[i])): 
                                    raw[i][j] = raw[i][j] * (temp1/f1)/(temp2/f2) # <<<< 訊息壓縮
                                for j in range(reject1-1, reject2-1): # 拒絕域中的修正 < 還沒找到好方法
                                    raw[i][j] = temp1/f1
                            elif f2 > f1: # 前段適配後段
                                for j in range(0, reject2-1): 
                                    raw[i][j] = raw[i][j] * (temp2/f2)/(temp1/f1)
                                for j in range(reject1-1, reject2-1): # 拒絕域中的修正
                                    raw[i][j] =temp2/f2
                            temp0 = 0.00
                            temp1 = 0.00
                            temp2 = 0.00
                            f0 = 0
                            f1 = 0
                            f2 = 0
                pp.raw_lineplot()
            except:
                print("try again, or ?")
    # 跳出輸入拒絕區間的視窗
    def reject_interval(self): 
        pass

    def semiauto_displace(self):
        '''自動去頭動, 用次數分配處理頭動 *未完成, 目前只有直方圖*'''
        for i in range(len(self.S_D_Mask)):  # 0 1
            if self.S_D_Mask[i] == "1":
                z = 1.96
                print(min(self.raw_wl1[i]),max(self.raw_wl1[i]))
                a = [round(i,3) for i in self.raw_wl1[i]]
                print(np.median(a)) #中位數
                print(scipy.stats.mode(a)[0][0]) #眾數
                print(np.mean(self.raw_wl1[i])) #平均數
                fig = plt.subplot(111)
                fig.title.set_text(self.S_D_Keys[i])
                plt.hist(self.raw_wl1[i])
                plt.axvline(x=scipy.stats.mode(a)[0][0]+z*np.std(self.raw_wl1[i]),ymin=0,ymax=1,linestyle="--",color="red")
                plt.axvline(x=scipy.stats.mode(a)[0][0]-z*np.std(self.raw_wl1[i]),ymin=0,ymax=1,linestyle="--",color="red")
                plt.show()
    #----------------------------------
    # Coeffiecient of Variation(CV)
    # 依照標準分辨 Good or Bad Channels
    #----------------------------------
    '''
    CV(變異係數) = S / Xbar

    Lu
    Channel C.V. > 15%
    Trial C.V. > 5%

    NirsLab 
    Gain Setting = 8
    CV = 7.5
    '''
    def cov(self):
        temp1 = 0.00 # Channel數值加總用
        temp2 = 0.00
        temp1sq = 0.00 # Channel數值平方和
        temp2sq = 0.00
        wl1_avg = [] # 產生每個Channel的平均數、標準差
        wl1_delta = []
        wl2_avg = []
        wl2_delta = []
        self.cv1 = [] # Coefficient of Variation
        self.cv2 = []
        # 先算整體的平均數標準差(母群)
        for i in range(len(self.S_D_Mask)):  # 0 1
            if self.S_D_Mask[i] == "1":
                for j in range(len(self.raw_wl1[i])):
                    w1 = float(self.raw_wl1[i][j])
                    w2 = float(self.raw_wl2[i][j])
                    temp1 += w1
                    temp2 += w2
                    temp1sq += w1**2
                    temp2sq += w2**2
                m1 = temp1/self.max_frames
                s1 = pow((temp1sq-temp1**2/self.max_frames)/self.max_frames, .5)
                m2 = temp2/self.max_frames
                s2 = pow((temp2sq-temp2**2/self.max_frames)/self.max_frames, .5)
                wl1_avg.append(m1)
                wl2_avg.append(m2)
                wl1_delta.append(s1)
                wl2_delta.append(s2)
                self.cv1.append( 100*s1/m1) # %
                self.cv2.append( 100*s2/m2)
                temp1 = 0.00
                temp2 = 0.00
                temp1sq = 0.00
                temp2sq = 0.00
        #return cv1, cv2
    # gain_Setting = int(input("Gain的標準："))
    # channel_CV = float(input("Channel的C.V.(%)："))

    #-----------------------
    # Bandpass Filter
    #-----------------------
    def butter_bandpass(self, data, lowcut, highcut, fs, order=5):
        '''
        Butterworth 濾波器
        . fs Sampling rate 取樣頻率
        . order 濾波器的階數, 階數越大效果越好, 但計算量也增加
        . Wn 正歸化的截止頻率, 介於0-1之間, 所能處理的最高頻率為 fs/2

        Nirslab > LPF = .01, HPF = .2
        Lu, Chia-Feng > LPF = .01, HPF = .1
        '''
        # The Nyquist rate of the signal.
        #nyq = .5 * fs 
        #low = lowcut / nyq # .01
        #high = highcut / nyq # .2

        b, a = butter(order, [lowcut,highcut], btype='band')
        #b, a = butter(order, lowcut, btype='low')
        #b, a = butter(order, highcut, btype='high')

        #w, h = freqz(b, a, worN=2000) # 新增
        #y = lfilter(b, a, data) # < 一次
        y = filtfilt(b, a, data) # < 向前、向後共兩次
        return y

    def filterLine(self):
        '''
        由原始線圖 > 過濾線圖(呈現) > 血氧濃度線圖(A)
            raw_        filtered_    deltaA_
        
        nirsLAB中 lf = .01 、 hf = .2
        但實際測試 lf = .005 、 hf = .12 結果比較接近
        '''
        temp1 = 0.00
        temp2 = 0.00
        self.A1 = []
        self.A2 = []
        self.filtered_wl1 = self.raw_wl1
        self.filtered_wl2 = self.raw_wl2
        for key,value in self.raw.items():
            fig = plt.figure() # 清空畫布
            fig = plt.subplot(111)
            for i in range(len(self.S_D_Mask)):  # 0 1
                if self.S_D_Mask[i] == "1": 
                    fig = plt.subplot(111)
                    if value[1] == 1:
                        for j in range(len(self.raw_wl1[i])):
                            temp1 += self.raw_wl1[i][j]
                        temp1 = temp1/len(eval(key)[i]) # 平均數
                        self.filtered_wl1[i] = pp.butter_bandpass(self.filtered_wl1[i], .005, .12, fs=self.Fs) #Band-Pass, 一條資料串
                        self.filtered_wl1[i] = self.filtered_wl1[i] + temp1 # 呈現用
                        plt.plot(self.xLabel, self.filtered_wl1[i], label=self.S_D_Keys[i]) # filtered data
                        temp1 = 0.00
                    elif value[1] == 2:
                        for j in range(len(self.raw_wl2[i])):
                            temp2 += self.raw_wl2[i][j]
                        temp2 = temp2/len(self.raw_wl2[i]) # 平均數
                        self.filtered_wl2[i] = pp.butter_bandpass(self.filtered_wl2[i], .005, .12, fs=self.Fs) #Band-Pass
                        self.filtered_wl2[i] = self.filtered_wl2[i] + temp2 # 呈現用
                        plt.plot(self.xLabel, self.filtered_wl2[i], label=self.S_D_Keys[i]) # filtered data
                        temp2 = 0.00
            #plt.ylim(-0.003, 0.003)
            fig.title.set_text("Filtered Data("+str(eval(value[0]))+"nm)")
            plt.grid(True)
            plt.xlabel('frames')
            plt.ylabel('Amplitude(V)')
            plt.legend(loc='best') # show label
            plt.pause(.1)
            #plt.show()
    #-------------------------
    # Beer-Lambert Law (mBLL)
    #-------------------------    
    '''
    Molar Extinction Coefficients [wavelength, for oxyH, for deoxyH]
    from W. B. Gratzer
    [760, 1486.5865, 3843.707] # 除以1000?
    [850, 2526.3910, 1798.643]
    form J. M. Schmitt
    [760, 1381.8, 3961.16]
    [850, 2441.18, 1842.4]
    from S. Takatani, M D. Graham
    [760, 1349.558, 3910.494]
    [850, 2436.574, 1888.460]

    NiRx Absorption Coefficients
    [760, 1.34956, 3.56624]
    [850, 2.43657, 1.59211]

    DPF in NiRx
    6yrs
    [760, 5.54363]
    [850, 4.48135]
    20yrs
    [760, 6.0022]
    [850, 4.93992]
    30yrs(預設)
    [760, 6.2966]
    [850, 5.23433]
    '''
    def concentration(self): # input filtered data
        #---------------------------------------------------
        # Coefficients form W. B. Gratzer and J. M. Schmitt
        #---------------------------------------------------
        wave_Lengths = 2 # 760 850
        baseline_inframes = [1, self.max_frames]
        totHb = 75 # uM
        mov2Sat = 70 # %

        pathLength = 3 # 測量距離, optical path(cm), 深度相當於距離的 1/4
        wl1_DPF = 6.4 # Differential Pathlength Factor (DPF)
        wl2_DPF = 5.75

        dp1 = wl1_DPF * pathLength # differential pathlength(DP)
        dp2 = wl2_DPF * pathLength

        sq = 10**(-3)
        εHbO_wl1 = 1486.5865*sq # 單位?
        εHbO_wl2 = 2526.3910*sq
        εHbR_wl1 = 3843.707*sq
        εHbR_wl2 = 1798.643*sq
        '''
        #----------------------------------------------
        # 光吸收度(A) = 吸收係數(α) * 光徑長(l) * 濃度(c)
        #----------------------------------------------
        α > 吸收係數(absorptivity), 或稱absorption coefficient, 亦可稱為消光係數(extinction coefficient, k)
            若 b > 光徑長以 cm為單位
               c > 濃度使用 M 為單位
               吸收係數以 (1/M)*(1/cm) or L*(1/g)*(1/cm) 為單位, 即為 莫耳吸收係數(molar absorptivity, ε)
        
            ### A = ε*b*c ### 為常見的表示法

        OD > 光密度(optical desity), 材料遮光能力的表徵, 是一個對數值沒有單位
            OD = log(I0/I1)
            透光度 > 10^(-OD)
            透光率 > 100*10^(-OD) = T%

        T > 穿透度(Transmittance), 100* I1 / I0 (%為單位)
            入射光的強度(I0), incident light intensity
            透射光的強度(I1), transmission light intensity  <<< *Nirx所給的數據*

            ### A = -logT = -log(I1/T0) = 2 - log(T%) = ε*c*d*DPF +G ### 為光的被吸收度
        
        A > 光吸收度(absorbance), 一般而言介於0-2之間 ### 機器產出的結果
            0, 完全無吸收
            2, 百分之99的光通過時被吸收
            ΔA = A_after - A_before = log10(I_before / I_after) = ε*Δc*d*DPF +G 

            ### ΔA =  ln( mean_I / I) ### 改變的A為I總和除以當前的I(這裡取ln)

        a > 吸收率(absorptance), 不考慮散射、反射
            a = (I0 - I1) / I0

        #-----------------------------------------------
        # 光密度的變化(ΔA) = (εHb*ΔHb) + εHbO*ΔHbO)*d*DPF
        #-----------------------------------------------

        > 正常情況下CoHb、MetHb濃度很少, 且在650nm之後吸收係數非常低
        
        MVO2Sat = 70%

        saO2 > 動脈血氧濃度, Arterial Oxygen Saturation, 一般人介於95%-98%, 低於90%為低氧血症
               totHb > 全血液 total hemoglobin, 假設為均質溶液(無散射) # nirsLab設定為 75 uM 
               oxyHb > 含氧血 oxy-hemoglobin
               deoHb > 去氧血 deoxy-hemoglobin
               co2Hb > 一氧化碳血紅素
               metHb > 變性血紅素

            ### ΔHb = (ε2HbO*ΔA1/DPF1 - ε1HbO*ΔA2/DPF2)/ ### 1, 2 表示不同波長的光

            ### ΔHbO ###
            
            ### saO2 = oxyHb / (oxyHb + deoHb + co2Hb + metHb) ###

        svO2 > 靜脈血氧濃度, Venous Oxygen Saturation (SvO2), 一般人介於60%-85%

        spO2 > 脈衝式血氧濃度, Saturation of Peripheral Oxygen
               
            ### spO2 = 100* oxyHb / (oxyHb + deoHb) ### 血氧飽和度(%), oxygen saturation
        '''
        #--------
        # I to A 
        #--------
        temp_A1 = []
        temp_A2 = []
        self.deltaA_wl1 = []
        self.deltaA_wl2 = []
        for i in range(len(self.S_D_Mask)):  # 0 1
            if self.S_D_Mask[i] == "1":
                for j in range(self.max_frames):
                    temp_A1.append(math.log(np.mean(self.filtered_wl1[i]) / self.filtered_wl1[i][j],math.e))
                    temp_A2.append(math.log(np.mean(self.filtered_wl2[i]) / self.filtered_wl2[i][j],math.e))
                self.deltaA_wl1.append(temp_A1)
                self.deltaA_wl2.append(temp_A2)
                temp_A1 = []
                temp_A2 = []
            else:
                self.deltaA_wl1.append([])
                self.deltaA_wl2.append([])
        #---------------
        # A to HbO & Hb
        #---------------
        b = float(pathLength*(εHbR_wl1*εHbO_wl2 - εHbO_wl1*εHbR_wl2))
        temp_HbO = []
        temp_HbR = []
        self.HbO = []
        self.HbR = []
        for i in range(len(self.S_D_Mask)):  # 0 1
            if self.S_D_Mask[i] == "1":
                for j in range(self.max_frames):
                    temp_HbO.append((εHbR_wl1*self.deltaA_wl2[i][j]/wl2_DPF - εHbR_wl2*self.deltaA_wl1[i][j]/wl1_DPF)/b)
                    temp_HbR.append((εHbO_wl2*self.deltaA_wl1[i][j]/wl1_DPF - εHbO_wl1*self.deltaA_wl2[i][j]/wl2_DPF)/b)
                self.HbO.append(temp_HbO)
                self.HbR.append(temp_HbR)
                temp_HbO = []
                temp_HbR = []
            else:
                self.HbO.append([])
                self.HbR.append([])

        return self.HbO, self.HbR

    def hemoglobin(self):
        pp.concentration()
        fig = plt.figure() # 清空畫布
        fig = plt.subplot(111)
        for i in range(0,5):  #len(self.S_D_Mask)):  # 0 1
            if self.S_D_Mask[i] == "1": 
                plt.plot(self.xLabel, self.HbO[i], label=self.S_D_Keys[i]) # filtered data
                plt.plot(self.xLabel, self.HbR[i], label=self.S_D_Keys[i]) # filtered data
        #plt.ylim(-0.003, 0.003)
        plt.grid(True)
        #plt.xlabel('frames')
        #plt.ylabel('Amplitude(V)')
        plt.legend(loc='best') # show label
        plt.pause(.1)

#---------------------
# GUI >>> 起始輸入介面
#---------------------

workshop = tk.Tk()
workshop.title("NiRx 數據分析")
workshop.resizable(0,0) # 鎖定視窗大小

def clickOK():
    mb.showinfo("分析開始","準備開始 ! \n 下好離手 !")

def mb_Option():
    mb.showinfo()
    mb.showerror()
    mb.askyesno()
    mb.askokcancel()

def cutpoint0():
    sd.askinteger("Cut Point", "請輸入起始切截點(frames, 如-10)：")

def cutpoint1():
    sd.askinteger("Cut Point", "請輸入結束切截點(frames, 如80)：")

# Introduction
preProcessing = tk.Label(workshop,bg="red",text="PreProcessing",fg="white", width=10,height=2,borderwidth=10).grid(column=0,row=0)
level1 = tk.Label(workshop,bg="green",text="LEVEL 1.",fg="white",width=10,height=2,borderwidth=10).grid(column=1,row=0)
level2 = tk.Label(workshop,bg="blue",text="LEVEL 2.",fg="white",width=10,height=2,borderwidth=10).grid(column=2,row=0)

buttons_func = {
    ## PreProcessing
    # Data & Conditions
    "load_Data":     ["讀取參數", 0, 1, pp.references],
    "set_Markers":   ["畫訊號圖", 0, 2, pp.raw_lineplot],
    # Data Preprocessing
    "truncate":      ["手去頭動", 0, 3, pp.head_displace],
    "check_Quality": ["共變分析", 0, 4, clickOK],
    "apply_Filter":  ["畫濾波圖", 0, 5, pp.filterLine],
    # Hemodynamic Staties
    "set_Parameters":["嘗試結果", 0, 6, pp.hemoglobin],
    "compute":       ["情境存檔", 0, 7, clickOK],
    ## First level
    "first_lv":      ["個人比較", 1, 1, clickOK],
    ## Second level
    "second_lv":     ["實驗比較", 2, 1, clickOK],
    }

for key in buttons_func:
    key = ttk.Button(workshop, 
        text=buttons_func[key][0], 
        command=buttons_func[key][3]).grid(column=buttons_func[key][1],
                                            row=buttons_func[key][2]) 

workshop.mainloop() # 呼叫並維持視窗

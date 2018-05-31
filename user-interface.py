# -*- coding: utf-8 -*-

from tkinter import *
import pandas as pd
import numpy as np


class App():

    
    """----------------------------------------------------------------------------
    CONSTRUCTOR
    """
    
    def __init__(self):
        self.df = (pd.read_csv('/Users/isabellepolizzi/Desktop/UPC/IDSS/PW3/idss_pw3/data/processed_cleveland_data.csv').replace('?',0))
        self.data = []
        
        self.window=Tk()
        self.p1=PanedWindow()
        self.p2=PanedWindow(self.p1, orient=VERTICAL)
        self.patient_list=Listbox(self.p1)
    
    """----------------------------------------------------------------------------
    LOAD DATA FROM PATIENT (TRIGGERED BY LISTBOX ELEMENT CLICK)
    """
    def loadData(self,event):
        s = (self.df.iloc[[event.widget.curselection()[0]]]).values[0]
        data=[]
        
        # 0 Age
        if s[0]<33:
            desc="Young"
        elif s[0]>=33 and s[0]<=40:
            desc="Medium"
        elif s[0]>40 and s[0]<=52:
            desc="Old"
        elif s[0]>52:
            desc="Very old"
        else:
            desc==""
        data.append(["Age", s[0], desc])
    
        # 1 Gender
        if s[1]==1:
            desc="Male"
        elif s[1]==0:
            desc="Female"
        else:
            desc=""
        data.append(["Gender", s[1], desc])
            
        # 2 Chain pain
        if s[2]==1:
            desc="Typical angina"
        elif s[2]==2:
            desc="Atypical angina"
        elif s[2]==3:
            desc="Non-angina pain"
        elif s[2]==4:
            desc="Asymptomatic"
        else:
            desc=""
        data.append(["Chain pain", s[2], desc])
            
        # 3 Resting blood pressure
        if s[3]<128:
            desc="Low"
        elif s[3]>=128 and s[3]<=142:
            desc="Medium"
        elif s[3]>142 and s[3]<=154:
            desc="High"
        elif s[3]>154:
            desc="Very high"
        else:
            desc=""
        data.append(["Resting blood pressure", s[3], desc])
    
        # 4 Serum cholestoral
        if s[3]<188:
            desc="Low"
        elif s[3]>=188 and s[3]<=217:
            desc="Medium"
        elif s[3]>217 and s[3]<=281:
            desc="High"
        elif s[3]>281:
            desc="Very high"
        else:
            desc=""
        data.append(["Serum cholestoral", s[3], desc])
            
        # 5 Fasting blood sugar
        if s[5]==1:
            desc="Over 129 mg/dl"
        elif s[5]==0:
            desc="Under 129 mg/dl"
        else:
            desc=""
        data.append(["Fasting blood sugar", s[5], desc])
            
        # 6 Resting electrocardiographic results
        if s[6]==0:
            desc="Normal"
        elif s[6]==1:
            desc="ST-T Abnormal"
        elif s[6]==2:
            desc="Hyperpotrophy"
        else:
            desc=""
        data.append(["Resting electrocardiographic results", s[6], desc])
            
        # 7 Maximum heart rate achieved
        if s[7]<112:
            desc="Low"
        elif s[7]>=112 and s[7]<=152:
            desc="Medium"
        elif s[7]>152:
            desc="High"
        else:
            desc=""
        data.append(["Maximum heart rate achieved", s[7], desc])
            
        # 8 Exercise induced angina
        if s[8]==1:
            desc="True"
        elif s[8]==0:
            desc="False"
        else:
            desc=""
        data.append(["Exercise induced angina", s[8], desc])
            
        # 9 ST depression induced by exercise relative to rest
        if s[9]<1.5:
            desc="Low"
        elif s[9]>=1.5 and s[9]<=2.55:
            desc="Risk"
        elif s[9]>2.55:
            desc="Terrible"
        else:
            desc=""
        data.append(["ST depression induced by exercise relative to rest", s[9], desc])
            
        # 10 Slop
        if s[10]==1:
            desc="Upsloping"
        elif s[10]==2:
           desc="Flat"
        elif s[10]==3:
            desc="Downsloping"
        else:
            desc=""
        data.append(["Slop", s[10], desc])
            
        # 11 Number of major vessels colored by fluoroscopy
        if s[11]==0:
            desc="Fluoroscopy-0"
        elif s[11]==1:
            desc="Fluoroscopy-1"
        elif s[11]==2:
            desc="Fluoroscopy-2"
        elif s[11]==3:
            desc="Fluoroscopy-3"
        else:
            desc=""
        data.append(["Number of major vessels colored by fluoroscopy", s[11], desc])
            
        # 12 Thalanium scan
        if s[12]==3:
            desc="Normal"
        elif s[12]==6:
            desc="Fixed defect"
        elif s[12]==7:
            desc="Reversible defect"
        else:
            desc=""
        data.append(["Thalanium scan", s[12], desc])
            
        # 13 Diagnosis of heart disease
        if s[13]==0:
            desc="<50% narrowing"
        elif s[13]==1:
            desc=">50% narrowing"
        else: 
            desc=""
        data.append(["Diagnosis of heart disease", s[12], desc])
        
        
        self.data=data
        
        self.p2.grid_forget()
        
        Label(self.p2, text="Characteristic", font=("Helvetica", 18)).grid(sticky = W, row=1,column=0)
        Label(self.p2, text="Value", font=("Helvetica", 18)).grid(sticky = W, row=1,column=1)
        Label(self.p2, text="Description", font=("Helvetica", 18)).grid(sticky = W, row=1,column=2)
        
        for r in range(len(self.data)):
            Label(self.p2, text=self.data[r][0]).grid(sticky = W, row=r+2,column=0)
            Label(self.p2, text=self.data[r][1]).grid(sticky = W, row=r+2,column=1)
            Label(self.p2, text=self.data[r][2]).grid(sticky = W, row=r+2,column=2)
        
        #print(self.data)
        
    
    """----------------------------------------------------------------------------
    SET WINDOW WITH PANELS
    """
    def launchApp(self):
        #window=Tk()
        self.window.geometry("800x500")
        #p1 = PanedWindow()
        self.p1.pack(fill=BOTH, expand=1)
    
        #patient_list=Listbox(p1)
        for i in range (len(self.df)):
            self.patient_list.insert(i, "Patient "+str(i))
            self.patient_list.bind("<<ListboxSelect>>", self.loadData)
    
        self.p1.add(self.patient_list)
    
        #p2 = PanedWindow(p1, orient=VERTICAL)
        self.p1.add(self.p2)
    
        Label(self.p2, text="Data Description", font=("Helvetica", 24)).grid(sticky = W, row=0,column=0)
        
        self.window.mainloop()
    
"""----------------------------------------------------------------------------
CREATE AND LAUNCH THE APP
"""
app = App()
app.launchApp()
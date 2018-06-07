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
        self.characteristics=['Age',
                              'Gender',
                              'Chest pain',
                              'Resting blood pressure', 
                              'Serum cholestoral',
                              'Fasting blood sugar',
                              'Resting electrocardiographic results',
                              'Maximum heart rate achieved',
                              'Exercise induced angina',
                              'ST depression induced by exercise relative to rest',
                              'Slop',
                              'Number of major vessels colored by fluoroscopy',
                              'Thalanium scan',
                              'Diagnosis of heart disease'
                              ]
        self.data = []
        self.index = 0
        
        self.window=Tk()
        self.window.geometry("800x500")
        self.window.title("Heart Failure Risk Prediction")
        
        #MAIN PANEL
        self.mainPane=PanedWindow()
        self.mainPane.pack(fill=BOTH, expand=1)
        
        #LEFT PANEL
        self.leftPane=PanedWindow(self.mainPane, orient=VERTICAL)
        self.patientList=Listbox(self.leftPane)
        self.addButton=Button(self.leftPane, text="Add patient", command=self.newPatient)
        self.editButton=Button(self.leftPane, text="Edit", command=self.editPatient)
        
        #RIGHT PANEL
        self.rightPane=PanedWindow(self.mainPane, orient=VERTICAL)
        self.frame=Frame(self.rightPane, relief=GROOVE)
        
    
    """----------------------------------------------------------------------------
    LOAD DATA FROM PATIENT (TRIGGERED BY LISTBOX ELEMENT CLICK)
    """
    def loadData(self,event):
        self.index = event.widget.curselection()[0]
        s = (self.df.iloc[[self.index]]).values[0]
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
        data.append([self.characteristics[0], s[0], desc])
    
        # 1 Gender
        if s[1]==1:
            desc="Male"
        elif s[1]==0:
            desc="Female"
        else:
            desc=""
        data.append([self.characteristics[1], s[1], desc])
            
        # 2 Chest pain
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
        data.append([self.characteristics[2], s[2], desc])
            
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
        data.append([self.characteristics[3], s[3], desc])
    
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
        data.append([self.characteristics[4], s[4], desc])
            
        # 5 Fasting blood sugar
        if s[5]==1:
            desc="Over 129 mg/dl"
        elif s[5]==0:
            desc="Under 129 mg/dl"
        else:
            desc=""
        data.append([self.characteristics[5], s[5], desc])
            
        # 6 Resting electrocardiographic results
        if s[6]==0:
            desc="Normal"
        elif s[6]==1:
            desc="ST-T Abnormal"
        elif s[6]==2:
            desc="Hyperpotrophy"
        else:
            desc=""
        data.append([self.characteristics[6], s[6], desc])
            
        # 7 Maximum heart rate achieved
        if s[7]<112:
            desc="Low"
        elif s[7]>=112 and s[7]<=152:
            desc="Medium"
        elif s[7]>152:
            desc="High"
        else:
            desc=""
        data.append([self.characteristics[7], s[7], desc])
            
        # 8 Exercise induced angina
        if s[8]==1:
            desc="True"
        elif s[8]==0:
            desc="False"
        else:
            desc=""
        data.append([self.characteristics[8], s[8], desc])
            
        # 9 ST depression induced by exercise relative to rest
        if s[9]<1.5:
            desc="Low"
        elif s[9]>=1.5 and s[9]<=2.55:
            desc="Risk"
        elif s[9]>2.55:
            desc="Terrible"
        else:
            desc=""
        data.append([self.characteristics[9], s[9], desc])
            
        # 10 Slop
        if s[10]==1:
            desc="Upsloping"
        elif s[10]==2:
           desc="Flat"
        elif s[10]==3:
            desc="Downsloping"
        else:
            desc=""
        data.append([self.characteristics[10], s[10], desc])
            
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
        data.append([self.characteristics[11], s[11], desc])
            
        # 12 Thalanium scan
        if s[12]==3:
            desc="Normal"
        elif s[12]==6:
            desc="Fixed defect"
        elif s[12]==7:
            desc="Reversible defect"
        else:
            desc=""
        data.append([self.characteristics[12], s[12], desc])
        
         # 13 Diagnosis of heart disease
        if s[13]==0:
            desc="<50% diameter narrowing"
        elif s[13]==1:
            desc=">50% diameter narrowing"
        else:
            desc=""
        data.append([self.characteristics[13], s[13], desc])
        
        # 14 PREDICTION 
        # prediction=predictHF(data)
        
        self.data=data
        
        list = self.frame.grid_slaves()
        for l in list:
            l.destroy()
        
        Label(self.frame, text="Characteristic", font=("Helvetica", 18)).grid(sticky = W, row=1,column=0)
        Label(self.frame, text="Value", font=("Helvetica", 18)).grid(sticky = W, row=1,column=1)
        Label(self.frame, text="Description", font=("Helvetica", 18)).grid(sticky = W, row=1,column=2)
        
        for r in range(len(self.data)):
            Label(self.frame, text=self.data[r][0]).grid(sticky = W, row=r+2,column=0)
            Label(self.frame, text=self.data[r][1]).grid(sticky = W, row=r+2,column=1)
            Label(self.frame, text=self.data[r][2]).grid(sticky = W, row=r+2,column=2)
        
    
    
    """----------------------------------------------------------------------------
    CREATE NEW PATIENT
    """
    def newPatient(self):
        
        form=Tk()
        form.geometry("520x450")
        form.title("New patient form")
        
        def addPatient():
            data=[]
            list = form.winfo_children()
            for l in list:
                if type(l) is Entry:
                    data.append(int(l.get()))
            
            self.df.loc[-1] = data
            self.patientList.insert(END,"New patient "+str(self.patientList.index(END)))
            
            form.destroy() #Close form window
        
        for i in range (len(self.characteristics)):
            Label(form, text=self.characteristics[i]).grid(sticky=W, row=i, column=0)
            Entry(form).grid(row=i, column=1)
        
        Button(form, text="Add", command=addPatient).grid(column=1)
        Button(form, text="Cancel", command=form.destroy).grid(column=0)
                
        form.mainloop()
        
    """----------------------------------------------------------------------------
    CREATE NEW PATIENT
    """
    def editPatient(self):
        
        form=Tk()
        form.geometry("520x450")
        form.title("Edit patient data")
        
        def modifyData():
            data=[]
            list = form.winfo_children()
            for l in list:
                if type(l) is Entry:
                    data.append(int(l.get()))
            
            self.df.iloc[self.index] = data
            self.patientList.activate(self.index)
        
            form.destroy() #Close form window
        
        for i in range (len(self.characteristics)):
            Label(form, text=self.characteristics[i]).grid(sticky=W, row=i, column=0)
            v = StringVar(form, value= self.data[i][1])
            Entry(form, textvariable=v).grid(row=i, column=1)
        i
        Button(form, text="Modify", command=modifyData).grid(column=1)
        Button(form, text="Cancel", command=form.destroy).grid(column=0)
                
        form.mainloop()
    
    
    """----------------------------------------------------------------------------
    SET WINDOW WITH PANELS
    """
    def launchApp(self):
    
        for i in range (len(self.df)):
            self.patientList.insert(i, "Patient "+str(i))
            self.patientList.bind("<<ListboxSelect>>", self.loadData)
    
        self.leftPane.add(self.patientList, height=430)
        self.leftPane.add(self.editButton, height=30)
        self.leftPane.add(self.addButton, height=30)
        self.rightPane.add(self.frame)
        self.mainPane.add(self.leftPane)
        self.mainPane.add(self.rightPane)
        
        Label(self.rightPane, text="Data Description", font=("Helvetica", 24)).grid(sticky = W, row=0,column=0)
        
        
        self.window.mainloop()
    
"""----------------------------------------------------------------------------
CREATE AND LAUNCH THE APP
"""
app = App()
app.launchApp()
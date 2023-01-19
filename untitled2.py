#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 22:30:04 2023

@author: tavo
"""
import numpy as np
import pandas as pd

import multiprocessing as mp

import differint.differint as df

###############################################################################
# Day length functions
###############################################################################

def GetDayLenght(J,lat):
    #CERES model  Ecological Modelling 80 (1995) 87-95
    phi = 0.4093*np.sin(0.0172*(J-82.2))
    coef = (-np.sin(np.pi*lat/180)*np.sin(phi)-0.1047)/(np.cos(np.pi*lat/180)*np.cos(phi))
    ha =7.639*np.arccos(np.max([-0.87,coef]))
    return ha

def GetYearLengths(lat):
    return np.array([GetDayLenght(j, lat) for j in range(368)])

###############################################################################
# Solar flux functions
###############################################################################

def SolarFluxCoefs(J,lat):
    
    I0 = 1367
    fact = np.pi/180
    
    delta = 23.45*np.sin(0.986*(J+284))
    a = np.sin(lat*fact)*np.sin(delta*fact)
    b0 = np.cos(lat*fact)*np.cos(delta*fact)
    Ct = 1+0.034*np.cos((J-2)*fact)
    Gamma = 0.796-0.01*np.sin((0.986*(J+284))*fact)
    
    return [a,b0,Ct*I0*Gamma,120*Gamma]

def SolarFlux(T,coefs):
    #New model to estimate and evaluate the solar radiation
    #Y.El Mghouchi A.El BouardiZ.ChoulliT.Ajzoul
    #ETEE, Faculty of Sciences, Abdelmalek Essaadi University, Tetouan, Morocco
    
    fact = np.pi/180
    w = 15*(12-T)
    
    a,b0,A0,A1 = coefs
    
    b = b0*np.cos(w*fact)
    h = np.arcsin((a+b))
    sinh = np.sin(h)
    
    I = A0*sinh*np.e**(-0.13/sinh)
    coeff = 0.4511+sinh
    Dh = A1*np.e**(coeff)
    
    return I+Dh

def GetFluxByDay(J,lat):
    
    if lat>65:
        localLat = 65
    else:
        localLat = lat
    
    coefs = SolarFluxCoefs(J,localLat)
    def LocalFlux(T):
        return SolarFlux(T,coefs)
    
    vflux = np.vectorize(LocalFlux)
    flux = vflux(np.linspace(0,24,num=2000))
    flux = flux[flux>0]
    return flux

def GetSFByLat(lat):
    
    container = []
    
    for k in range(368):
        
        localflux = GetFluxByDay(k,lat)
        length = np.mean(localflux)/np.std(localflux)
        container.append(length)
        
    return np.array(container)

###############################################################################
# Get dicts
###############################################################################

def GetDictsBylat(lat,function):
    
    Dict = {}
    Dict_n = {}
    
    days = function(lat)
    days_n = (days - days.min())/(days.max() - days.min())
    
    inDict = {}
    inDict_n = {}
    
    for k,_ in enumerate(days):
        inDict[k] = days[k]
        inDict_n[k] = days_n[k]
    
    Dict[0] = inDict
    Dict_n[0] = inDict_n
    
    for j in range(1,4):
        
        localdf = df.GL(j/3,days,num_points=len(days))
        localdf_n = (localdf - localdf.min())/(localdf.max() - localdf.min())
        
        inDict = {}
        inDict_n = {}
        
        for i,_ in enumerate(localdf):
            inDict[i] = localdf[i]
            inDict_n[i] = localdf_n[i]
        
        Dict[j] = inDict
        Dict_n[j] = inDict_n
    
    return Dict,Dict_n

###############################################################################
# Get dicts
###############################################################################

MaxCPUCount=int(0.85*mp.cpu_count())

def GetDictsBylatDL(lat):
    return GetDictsBylat(lat,GetYearLengths)

def GetDictsBylatSF(lat):
    return GetDictsBylat(lat,GetSFByLat)

#Wraper function for parallelization 
def GetDataParallel(data,Function):
    
    localPool=mp.Pool(MaxCPUCount)
    mData=localPool.map(Function, [ val for val in data])
    localPool.close()
    
    return mData

def MakeNestedDicts(qrys,lats,function):
    
    qryToDict= {}
    qryToDict_n = {}
    
    localdicts = GetDataParallel(lats,function)
    
    for val,sal in zip(qrys,localdicts):
        qryToDict[val] = sal[0]
        qryToDict_n[val] = sal[1]

    return qryToDict,qryToDict_n,localdicts

###############################################################################
# MetaData Features
###############################################################################

data = pd.read_csv('/media/tavo/storage/urgencias/urgenciasfinal.csv')

data['date_st'] = pd.to_datetime(data['date_st'], format='%Y-%m-%d')
data['year'] = data['date_st'].dt.year
data['dayofyear'] = data['date_st'].dt.dayofyear

data['qry'] = ['lat=='+str(val)+' & long=='+str(sal) for val,sal in zip(data['lat'],data['long'])]
qrylats = data.groupby('qry')['lat'].mean()

###############################################################################
# MetaData Features
###############################################################################

localdicts = GetDataParallel(qrylats,GetDictsBylatDL)

qryToDL= {}
qryToDL_n = {}

for val,sal in zip(qrylats.index,localdicts):
    qryToDL[val] = sal[0]
    qryToDL_n[val] = sal[1]

data['lengthofday'] = [qryToDL[val][0][sal] for val,sal in zip(data['qry'],data['dayofyear'])]
data['lengthofdayd03'] = [qryToDL[val][1][sal] for val,sal in zip(data['qry'],data['dayofyear'])]
data['lengthofdayd06'] = [qryToDL[val][2][sal] for val,sal in zip(data['qry'],data['dayofyear'])]
data['lengthofdayd10'] = [qryToDL[val][3][sal] for val,sal in zip(data['qry'],data['dayofyear'])]

###############################################################################
# Time Features
###############################################################################

localdictsSF = GetDataParallel(qrylats,GetDictsBylatSF)

qryToSF= {}
qryToSF_n = {}

for val,sal in zip(qrylats.index,localdictsSF):
    qryToSF[val] = sal[0]
    qryToSF_n[val] = sal[1]
  
data['sf_msd'] = [qryToSF[val][0][sal] for val,sal in zip(data['qry'],data['dayofyear'])]
data['sf_msd03'] = [qryToSF[val][1][sal] for val,sal in zip(data['qry'],data['dayofyear'])]
data['sf_msd06'] = [qryToSF[val][2][sal] for val,sal in zip(data['qry'],data['dayofyear'])]
data['sf_msd10'] = [qryToSF[val][3][sal] for val,sal in zip(data['qry'],data['dayofyear'])]

data.to_csv('/media/tavo/storage/urgencias/urgenciasfinalv2.csv',index=False)
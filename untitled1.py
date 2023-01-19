#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 22:25:24 2023

@author: tavo
"""

import numpy as np
import pandas as pd

###############################################################################
# Network encoder
###############################################################################

def ProcessAge(val,sal,lst):
    age = 0
    if val==lst[0]:
        age = sal/(365*24)
    elif val==lst[1]:
        age = sal/365
    elif val==lst[2]:
        age = sal/12
    elif val==lst[3]:
        age = sal
    return age

def MakeDF(path):
    
    with open(path) as f:
        lines = f.readlines()
        lines = [val.strip().split('|') for val in lines]
        
    headerSize = len(lines[0])
    finalLines = [val for val in lines if len(val)==headerSize]
    
    localDF = pd.DataFrame(np.array(finalLines[1::]),columns=finalLines[0])
    
    return localDF

headers = ['ID','CLUES','date_st','HORASESTANCIA','age','SEXO','AFECPRIN','desc','toem_norm','lat','long','alt']

###############################################################################
# Network encoder
###############################################################################

catloc = pd.read_csv('/media/tavo/storage/urgencias/catloc.csv')
newdata = catloc.groupby(['CVE_ENT','CVE_MUN'])['LAT_DECIMAL','LON_DECIMAL','ALTITUD'].mean()

###############################################################################
# Network encoder
###############################################################################

data2018 = pd.read_csv('/media/tavo/storage/urgencias/URGENCIAS2018.csv')

data2018['AFECPRIN'] = data2018['AFECPRIN'].astype(str)
data2018['AFECPRIN'] = [val.upper().strip() for val in data2018['AFECPRIN']]
data2018['correct'] = [1 if len(val)==4 else 0 for val in data2018['AFECPRIN']]

data2018 = data2018.drop(data2018[data2018.correct == 0].index)
data2018 = data2018.drop(data2018[data2018.ENTRESIDENCIA > 50].index)
data2018 = data2018.drop(data2018[data2018.MUNRESIDENCIA > 990].index)

inx = [tuple(val) for val in np.array(data2018[['ENTRESIDENCIA','MUNRESIDENCIA']])]
locdata = np.array(newdata.loc[inx])

data2018['lat'] = locdata[:,0]
data2018['long'] = locdata[:,1]
data2018['alt'] = locdata[:,2]

affecdata = pd.read_csv('/media/tavo/storage/urgencias/Catalogos_de_Urgencias_2018/Catálogos Urgencias 2018/CATCIE10.csv',encoding='latin-1',usecols=['Clave','Nombre'])
affecdata = affecdata.set_index('Clave')

data2018['desc'] = np.array(affecdata.loc[data2018['AFECPRIN']]).ravel().astype(str)

data2019 = pd.read_csv('/media/tavo/storage/urgencias/URGENCIAS2019.csv')

data2019['AFECPRIN'] = data2019['AFECPRIN'].astype(str)
data2019['AFECPRIN'] = [val.upper().strip() for val in data2019['AFECPRIN']]
data2019['correct'] = [1 if len(val)==4 else 0 for val in data2019['AFECPRIN']]

data2019 = data2019.drop(data2019[data2019.correct == 0].index)

data2019 = data2019.drop(data2019[data2019.ENTRESIDENCIA > 50].index)
data2019 = data2019.drop(data2019[data2019.MUNRESIDENCIA > 990].index)

inx = [tuple(val) for val in np.array(data2019[['ENTRESIDENCIA','MUNRESIDENCIA']])]
locdata = np.array(newdata.loc[inx])

data2019['lat'] = locdata[:,0]
data2019['long'] = locdata[:,1]
data2019['alt'] = locdata[:,2]

affecdata = pd.read_csv('/media/tavo/storage/urgencias/Catalogos_de_Urgencias_2019/CATCIE10.csv',encoding='latin-1',usecols=['Clave','Nombre'])
affecdata = affecdata.set_index('Clave')

data2019['desc'] = np.array(affecdata.loc[data2019['AFECPRIN']]).ravel().astype(str)

data = pd.concat([data2018,data2019])

data = data.drop(data[data.HORAINIATE > 50].index)
data = data.drop(data[data.MININIATE > 80].index)
data = data.dropna(subset=['HORAINIATE','MININIATE','FECHAINGRESO','FECHAALTA'])

data['refdate'] = ['1900-01-01 00:00:00' for val in data['AFECPRIN']]
data['refdate'] = pd.to_datetime(data['refdate'])

data['HORAINIATE'] = data['HORAINIATE'].astype(int)
data['MININIATE'] = data['MININIATE'].astype(int)

data['date_st'] = pd.to_datetime(data['FECHAINGRESO'], format='%Y-%m-%d',errors='coerce')
data['date_end'] = pd.to_datetime(data['FECHAALTA'], format='%Y-%m-%d',errors='coerce')

data = data.dropna(subset=['date_st','date_end'])

data['toem'] = (pd.to_datetime(data['HORAINIATE'].astype(str) + ':' + data['MININIATE'].astype(str), format='%H:%M')
          .dt.time)

data['toem'] =  pd.to_datetime(data['toem'],format='%H:%M:%S')
data['toem_float'] = (data['refdate'] - data['toem']).dt.seconds
data['toem_norm'] = [val/86400 for val in data['toem_float']]

data['CVEEDAD'] = data['CVEEDAD'].astype(int)
data = data.drop(data[data.CVEEDAD > 6].index)

data['SEXO'] = data['SEXO'].astype(int)
data = data.drop(data[data.SEXO > 6].index)

data['EDAD'] = data['EDAD'].astype(int)

data['age'] = [ProcessAge(val,sal ,[0,1,2,3]) for val,sal in zip(data['CVEEDAD'],data['EDAD'])]

data = data[headers]

###############################################################################
# Network encoder
###############################################################################

data2020 = MakeDF('/media/tavo/storage/urgencias/URGENCIAS2020.txt')

data2020['AFECPRIN'] = data2020['AFECPRIN'].astype(str)
data2020['AFECPRIN'] = [val.upper().strip() for val in data2020['AFECPRIN']]
data2020['correct'] = [1 if len(val)==4 else 0 for val in data2020['AFECPRIN']]

data2020 = data2020.drop(data2020[data2020.correct == 0].index)

data2020['ENTRESIDENCIA'] = data2020['ENTRESIDENCIA'].astype(int)
data2020['MUNRESIDENCIA'] = data2020['MUNRESIDENCIA'].astype(int)

data2020 = data2020.drop(data2020[data2020.ENTRESIDENCIA > 50].index)
data2020 = data2020.drop(data2020[data2020.MUNRESIDENCIA > 990].index)

inx = [tuple(val) for val in np.array(data2020[['ENTRESIDENCIA','MUNRESIDENCIA']])]
locdata = np.array(newdata.loc[inx])

data2020['lat'] = locdata[:,0]
data2020['long'] = locdata[:,1]
data2020['alt'] = locdata[:,2]

affecdata = pd.read_csv('/media/tavo/storage/urgencias/Catalogos_de_Urgencias_2020/CAT_CIE-10_2020.csv',encoding='latin-1',usecols=['Clave','Nombre'])
affecdata = affecdata.set_index('Clave')

data2020['desc'] = np.array(affecdata.loc[data2020['AFECPRIN']]).ravel().astype(str)

data2020['HORASESTANCIA'] = data2020['HORASESTANCIA'].astype(str)
data2020['hrs'] = [val[0:val.find(':')] for val in data2020['HORASESTANCIA']]

data2020 = data2020.drop(data2020[data2020.hrs == '99999.9'].index)

data2020['HORASESTANCIA'] = [int(val) for val in data2020['hrs']]

data2020['ID'] = data2020[list(data2020)[0]]

data2020['refdate'] = ['1900-01-01 00:00:00' for val in data2020['CLUES']]
data2020['refdate'] = pd.to_datetime(data2020['refdate'])

data2020['date_st'] = pd.to_datetime(data2020['fechaingreso'], format='%Y-%m-%d',errors='coerce')
data2020['date_end'] = pd.to_datetime(data2020['fechaalta'], format='%Y-%m-%d',errors='coerce')

data2020 = data2020.dropna(subset=['date_st','date_end'])

data2020['toem'] = (pd.to_datetime(data2020['hora_ingreso'], format='%H:%M',errors='coerce').dt.time)
data2020 = data2020.dropna(subset=['toem'])

data2020['toem'] =  pd.to_datetime(data2020['toem'],format='%H:%M:%S')
data2020['toem_float'] = (data2020['refdate'] - data2020['toem']).dt.seconds
data2020['toem_norm'] = [val/86400 for val in data2020['toem_float']]

data2020['CVEEDAD'] = data2020['CVEEDAD'].astype(int)
data2020 = data2020.drop(data2020[data2020.CVEEDAD > 6].index)

data2020['SEXO'] = data2020['SEXO'].astype(int)
data2020 = data2020.drop(data2020[data2020.SEXO > 6].index)

data2020['EDAD'] = data2020['EDAD'].astype(int)

data2020['age'] = [ProcessAge(val,sal ,[2,3,4,5]) for val,sal in zip(data2020['CVEEDAD'],data2020['EDAD'])]

data2020 = data2020[headers]

data = pd.concat([data,data2020])

###############################################################################
# Network encoder
###############################################################################

data2021 = MakeDF('/media/tavo/storage/urgencias/URGENCIAS2021.txt')

data2021['AFECPRIN'] = data2021['AFECPRIN'].astype(str)
data2021['AFECPRIN'] = [val.upper().strip() for val in data2021['AFECPRIN']]
data2021['correct'] = [1 if len(val)==4 else 0 for val in data2021['AFECPRIN']]

data2021 = data2021.drop(data2021[data2021.correct == 0].index)

data2021['ENTRESIDENCIA'] = data2021['ENTRESIDENCIA'].astype(int)
data2021['MUNRESIDENCIA'] = data2021['MUNRESIDENCIA'].astype(int)

data2021 = data2021.drop(data2021[data2021.ENTRESIDENCIA > 50].index)
data2021 = data2021.drop(data2021[data2021.MUNRESIDENCIA > 990].index)

inx = [tuple(val) for val in np.array(data2021[['ENTRESIDENCIA','MUNRESIDENCIA']])]
locdata = np.array(newdata.loc[inx])

data2021['lat'] = locdata[:,0]
data2021['long'] = locdata[:,1]
data2021['alt'] = locdata[:,2]

affecdata = pd.read_csv('/media/tavo/storage/urgencias/Catalogos_de_Urgencias_2021/CAT_CIE_10_2021.csv',encoding='latin-1',usecols=['CLAVE','NOMBRE'])
affecdata['CLAVE'] = affecdata['CLAVE'].astype(str)
affecdata['CLAVE'] = [val.upper().strip() for val in affecdata['CLAVE']]
affecdata['CLAVE'] = [val.upper().strip().replace('"','') for val in affecdata['CLAVE']]

affecdata = affecdata.set_index('CLAVE')

data2021['desc'] = np.array(affecdata.loc[data2021['AFECPRIN']]).ravel().astype(str)

data2021['HORASESTANCIA'] = data2021['TIEMPO_ESTANCIA']

data2021['HORASESTANCIA'] = data2021['HORASESTANCIA'].astype(str)
data2021['HORASESTANCIA'] = [int(val[0:val.find(':')]) for val in data2021['HORASESTANCIA']]

data2021['ID'] = data2021[list(data2021)[0]]

data2021['refdate'] = ['1900-01-01 00:00:00' for val in data2021['CLUES']]
data2021['refdate'] = pd.to_datetime(data2021['refdate'])

data2021['date_st'] = pd.to_datetime(data2021['FECHAINGRESO'], format='%Y-%m-%d',errors='coerce')
data2021['date_end'] = pd.to_datetime(data2021['FECHAALTA'], format='%Y-%m-%d',errors='coerce')

data2021 = data2021.dropna(subset=['date_st','date_end'])

data2021['toem'] = (pd.to_datetime(data2021['HORA_INGRESO'], format='%H:%M',errors='coerce').dt.time)
data2021 = data2021.dropna(subset=['toem'])

data2021['toem'] =  pd.to_datetime(data2021['toem'],format='%H:%M:%S')
data2021['toem_float'] = (data2021['refdate'] - data2021['toem']).dt.seconds
data2021['toem_norm'] = [val/86400 for val in data2021['toem_float']]

data2021['CVEEDAD'] = data2021['CVEEDAD'].astype(int)
data2021 = data2021.drop(data2021[data2021.CVEEDAD > 6].index)

data2021['SEXO'] = data2021['SEXO'].astype(int)
data2021 = data2021.drop(data2021[data2021.SEXO > 6].index)

data2021['EDAD'] = data2021['EDAD'].astype(int)

data2021['age'] = [ProcessAge(val,sal ,[2,3,4,5]) for val,sal in zip(data2021['CVEEDAD'],data2021['EDAD'])]

data2021 = data2021[headers]

data = pd.concat([data,data2021])

###############################################################################
# Network encoder
###############################################################################

data2022 = MakeDF('/media/tavo/storage/urgencias/URGENCIAS2022.txt')

data2022['AFECPRIN'] = data2022['AFECPRIN'].astype(str)
data2022['AFECPRIN'] = [val.upper().strip() for val in data2022['AFECPRIN']]
data2022['correct'] = [1 if len(val)==4 else 0 for val in data2022['AFECPRIN']]

data2022 = data2022.drop(data2022[data2022.correct == 0].index)

data2022['ENTRESIDENCIA'] = data2022['ENTRESIDENCIA'].astype(int)
data2022['MUNRESIDENCIA'] = data2022['MUNRESIDENCIA'].astype(int)

data2022 = data2022.drop(data2022[data2022.ENTRESIDENCIA > 50].index)
data2022 = data2022.drop(data2022[data2022.MUNRESIDENCIA > 990].index)

inx = [tuple(val) for val in np.array(data2022[['ENTRESIDENCIA','MUNRESIDENCIA']])]
locdata = np.array(newdata.loc[inx])

data2022['lat'] = locdata[:,0]
data2022['long'] = locdata[:,1]
data2022['alt'] = locdata[:,2]

affecdata = pd.read_csv('/media/tavo/storage/urgencias/Catalogos_de_Urgencias_2022/CAT_CIE_10_2021.csv',encoding='latin-1',usecols=['CLAVE','NOMBRE'])
affecdata['CLAVE'] = affecdata['CLAVE'].astype(str)
affecdata['CLAVE'] = [val.upper().strip() for val in affecdata['CLAVE']]
affecdata['CLAVE'] = [val.upper().strip().replace('"','') for val in affecdata['CLAVE']]

affecdata = affecdata.set_index('CLAVE')

data2022['desc'] = np.array(affecdata.loc[data2022['AFECPRIN']]).ravel().astype(str)

data2022['HORASESTANCIA'] = data2022['HORASESTANCIA'].astype(str)
data2022['HORASESTANCIA'] = [int(val[0:val.find(':')]) for val in data2022['HORASESTANCIA']]

data2022['ID'] = data2022[list(data2022)[0]]

data2022['refdate'] = ['1900-01-01 00:00:00' for val in data2022['CLUES']]
data2022['refdate'] = pd.to_datetime(data2022['refdate'])

data2022['date_st'] = pd.to_datetime(data2022['fechaingreso'], format='%Y-%m-%d',errors='coerce')
data2022['date_end'] = pd.to_datetime(data2022['fechaalta'], format='%Y-%m-%d',errors='coerce')

data2022 = data2022.dropna(subset=['date_st','date_end'])

data2022['toem'] = (pd.to_datetime(data2022['hora_ingreso'], format='%H:%M',errors='coerce').dt.time)
data2022 = data2022.dropna(subset=['toem'])

data2022['toem'] =  pd.to_datetime(data2022['toem'],format='%H:%M:%S')
data2022['toem_float'] = (data2022['refdate'] - data2022['toem']).dt.seconds
data2022['toem_norm'] = [val/86400 for val in data2022['toem_float']]

data2022['CVEEDAD'] = data2022['CVEEDAD'].astype(int)
data2022 = data2022.drop(data2022[data2022.CVEEDAD > 6].index)

data2022['SEXO'] = data2022['SEXO'].astype(int)
data2022 = data2022.drop(data2022[data2022.SEXO > 6].index)

data2022['EDAD'] = data2022['EDAD'].astype(int)

data2022['age'] = [ProcessAge(val,sal ,[2,3,4,5]) for val,sal in zip(data2022['CVEEDAD'],data2022['EDAD'])]

data2022 = data2022[headers]

data = pd.concat([data,data2022])

###############################################################################
# Network encoder
###############################################################################

data['desc'] = [val.lower().replace('"','') for val in data['desc']]

data.to_csv('/media/tavo/storage/urgencias/urgenciasfinal.csv',index=False)

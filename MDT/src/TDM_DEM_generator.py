#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 14:40:23 2020

@author: rgrimson
"""
#%%
# básicas
import numpy as np
import matplotlib.pyplot as plt
import rasterio

# las cuatro operaciones morfológicas elementales para máscaras
from skimage.morphology import binary_closing
from skimage.morphology import binary_dilation
from skimage.morphology import binary_erosion
from skimage.morphology import binary_opening
from skimage.morphology import disk

# la función que calcula la entropía de una imágen
from skimage.filters.rank import entropy

#la función para completar datos faltantes
from rasterio.fill import fillnodata

path_in='./raw_data/'
path_out='./proc_data/'
f_in="TDM1_30_Delta_DEM.tif"
f_out="TDM1_30_Delta_DEM_clean.tif"
f_out2="TDM1_30_Delta_DEM_clean_smooth.tif"
raster = rasterio.open(path_in+f_in)

dem = raster.read(1)

#usar la siguiente definición para hacer una prueba pequeña
#dem = raster.read(1)[2200:3200,1000:2000] 

#usar la siguiente definición para hacer una prueba pequeña del bajo delta
#dem = raster.read(1)[3200:4200,2200:3200] 

#%%
# IDEA: hacer una máscara de puntos válidos e interpolar el resto a partir de estos
# se realiza en dos etapas
# Etapa 1 (máscara). La más cara se genera como intersección de dos máscaras:
#          1-a máscara de outliers
#              se consigue con herramientas de "morphology" a partir de umbrales
#          1-b máscara de ruido
#              se consigue usando la entropía como medida de ruido
# Etapa 2 (interpolación). Se usa
#
#
#
#%% Etapa 1-a
#defino rango de valores y outliers
hmin=17
hmax=32
plotear=False 


if plotear:
    plt.imshow(dem,vmin=hmin,vmax=hmax)

#% Armo una máscara de outliers
mask1=(dem>hmin)*(dem<hmax)
if plotear:
    plt.imshow(mask1)

#% Si un punto está cerca de uno inválido, es inválido
mask2 = binary_erosion(mask1, disk(3))
if plotear:
    plt.imshow(mask2)
#% y solo son válidos los puntos que están en un disco completamente contenido en los validos
mask3 = binary_opening(mask2, disk(3))
if plotear:
    plt.imshow(mask3*4+mask2*2+mask1*1) #plotear las tres máscaras
    plt.imshow(dem+30*(1-mask3),vmin=hmin,vmax=52) #plotear "lo que saco"

#%% Etapa 1-b
ndem=dem.copy()
ndem[ndem<-4]=-4.0
ndem[ndem>40]=40
ndem+=4
ndem/=22.0
ndem-=1
#dem
#plt.imshow(dem)
ndem=np.array((ndem+1)*128,dtype=np.uint8)
#plt.imshow(img)
ent = entropy(ndem, disk(2))
maske=ent<1.5
if plotear:
    plt.imshow(maske)

#%% Etapa 1. Cierre: generación de la máscara como intersección de ambas máscaras
mask=maske*mask3
if plotear:
    plt.imshow(mask)

#%% Etapa 2. Interpolación

# borro los datos fuera de la máscara
# y completo los faltantes con un peso inverso a la distancia y radio de 100
mdt=fillnodata(dem*mask,mask*1, max_search_distance=100) 

if plotear:
    plt.imshow(mdt)


#%%
#diferentes cosas para ver
if plotear:
    plt.imshow(mdt)
    
    plt.imshow(mask3)
    plt.imshow(maske)
    plt.imshow(mask)
    plt.figure()
    plt.imshow(mdt,vmin=hmin,vmax=30)
    plt.imshow(dem,vmin=hmin,vmax=30)
    
    
#plt.imshow(dem*mask)

#%% 3. Computo de minimos - def

def min_over_window(img,d,p=5):
    """ computa el percentil p (mas robusto que minimo) 
    de la ventana de diametro d
    de cada pixels de la imagen img"""
    b=img.copy()
    r=(d-1)//2
    for i in range(r,img.shape[0]-r):
        for j in range(r,img.shape[1]-r):
            b[i,j]=np.percentile(img[i-r:i+r,j-r:j+r],p)
    return b
#%% 3. Computo de minimos - tarda mucho en la imagen entera
#min_img=min_over_window(mdt,5)        
#plt.imshow(mdt,vmin=17,vmax=22)
#plt.imshow(min_img,vmin=17,vmax=22)


#%% 4. Recorte Final
#cargar máscara del delta
raster = rasterio.open(path_in+"delta.tif")
delta_mask = raster.read(1)!=0

mdt_delta=mdt*delta_mask
if plotear:
    plt.imshow(delta_mask)
    plt.imshow(mdt_delta)
    plt.imshow(mdt)

#%% 5. Guardado en GTiff el MDT obtenido

with rasterio.open(
    path_out+f_out,
    'w',
    driver='GTiff',
    height=mdt.shape[0],
    width=mdt.shape[1],
    count=1,
    dtype=mdt.dtype,
    crs=raster.crs,
    transform=raster.transform,
) as dst:
    dst.write(mdt_delta, 1)

#%% 6. Blurring final
from scipy.ndimage import gaussian_filter

gauss=gaussian_filter(mdt,15)*delta_mask
if plotear:
    plt.imshow(gauss,vmin=17,vmax=22)

#%% 7. Guardado en GTiff el blureado

with rasterio.open(
    path_out+f_out2,
    'w',
    driver='GTiff',
    height=mdt.shape[0],
    width=mdt.shape[1],
    count=1,
    dtype=mdt.dtype,
    crs=raster.crs,
    transform=raster.transform,
) as dst:
    dst.write(gauss, 1)



#%%  8. Grabar GTiff con entreopia a diferentes escalas
entropy_n = np.zeros([5,dem.shape[0],dem.shape[1]])
entropy_n [0] = entropy(ndem, disk(1))
entropy_n [1] = entropy(ndem, disk(2))
entropy_n [2] = entropy(ndem, disk(3))
entropy_n [3] = entropy(ndem, disk(4))
entropy_n [4] = entropy(ndem, disk(5))

#%
if True:
  with rasterio.open(
    path_out+"Entropy_n.tif",
    'w',
    driver='GTiff',
    height=mdt.shape[0],
    width=mdt.shape[1],
    count=5,
    dtype=mdt.dtype,
    crs=raster.crs,
    transform=raster.transform) as dst:
    for band in range(5):
          dst.write(entropy_n[band].astype(np.float32), band+1)

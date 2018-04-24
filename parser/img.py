#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import shutil
import os
import time
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

<<<<<<< HEAD
df=pd.read_csv('AUDUSD_MKV.csv_cnt')
=======
df=pd.read_csv('prop_cnt')
>>>>>>> 883936747d99c6bfaf557db60b993dead6d4894e
da=df.values

da=np.delete(da,0,axis=1)
plt.imshow(da)
plt.show()


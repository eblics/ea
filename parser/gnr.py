#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import shutil
import os
import time
import tensorflow as tf
import pandas as pd
import numpy as np

FLAGS=tf.flags.FLAGS
tf.flags.DEFINE_string('rawpath','','')
tf.flags.DEFINE_string('outpath','','')

def to_prop_file():
    df=pd.read_csv(FLAGS.rawpath)
    cnt,d,data=0,{},[]
    for index,row in df.iterrows():
        k=row[0]
        if d.get(k):d[k]+=1
        else: d[k]=1
        cnt+=1
    for k,x in d.items():
        data.append([k,x,float(x)/cnt])

    outdf=pd.DataFrame(data)
    outdf.to_csv(FLAGS.outpath,header=False,index=False,sep='\t',mode='a')

def to_markov_file():
    df=pd.read_csv(FLAGS.rawpath,header=None)
    cols=[]
    for index,row in df.iterrows():
        if row[0] in cols:
            continue
        cols.append(row[0])
    cols=sorted(cols)
    outdf1=pd.DataFrame(columns=cols)
    nr=0
    lastrow=None
    for col in outdf1.columns:
        od=pd.DataFrame(data=[[0 for i in xrange(len(cols))]],columns=cols)
        outdf1=outdf1.append(od,ignore_index=True)
        for index,row in df.iterrows():
            if lastrow is None:
                lastrow=row
                continue
            if lastrow[0]==col:
                outdf1.loc[nr,row[0]]+=1
            lastrow=row
        nr+=1
    outdf1.to_csv(FLAGS.outpath+'_cnt')

<<<<<<< HEAD
def to_expv_file():
=======
def to_prop_file2():
>>>>>>> 883936747d99c6bfaf557db60b993dead6d4894e
    pd.set_option('display.max_rows',None)
    df=pd.read_csv(FLAGS.rawpath,index_col=0)
    df['rs']=df.sum(axis=1)
    df=df.apply(lambda x:x/df['rs'])
    df=df.drop(['rs'])
<<<<<<< HEAD
    df.columns=df.columns.map(float)
    df.index=df.columns
    df['e']=df.apply(lambda x:(x*df.columns).sum(),axis=1)
    df.to_csv(FLAGS.outpath)

if sys.argv[1]=='prop':
    to_prop_file()
if sys.argv[1]=='markov':
    to_markov_file()
if sys.argv[1]=='expv':
    to_expv_file()
=======
    df.to_csv(FLAGS.outpath)


to_markov_file()
>>>>>>> 883936747d99c6bfaf557db60b993dead6d4894e

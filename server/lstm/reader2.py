#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
import numpy as np

import tensorflow as tf

Py3 = sys.version_info[0] == 3

#col:high,low,open,close
def ea_raw_data(symbol='AUDUSD',col=None):
    train_data,valid_data,test_data=[],[],[]
    #文件总长度：5626870
    # train_index,valid_index,test_index=0,4000000,5000000
    # max_len=1000000000;
    train_index,valid_index,test_index=0,80000,90000
    max_len=100000;
    index= 6
    if   col=='open':index=3;
    elif col=='high':index=4;
    elif col=='low':index=5;
    else:index=6;

    filename = './data/'+symbol+".txt";
    f=tf.gfile.GFile(filename, "r")
    f.readline()
    i=0;
    #for line in f.readlines()[1:]:
    while True:
        line=f.readline()
        arr=line.split(',')
        if(len(arr)<6): continue
        if(i>max_len):  break;
        v=float(arr[index])
        if i<valid_index:
            train_data.append(v);i+=1;
        elif i<test_index:
            valid_data.append(v);i+=1;
        else:
            test_data.append(v);i+=1;

    f.close()
    return train_data,valid_data,test_data

def ea_input_from_raw(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.float32)

    data_len = len(raw_data)
    batch_len = (data_len-1) // (batch_size*num_steps)
    # epoch_size = (batch_len - 1) // num_steps

    for i in range(batch_len):
        x=np.zeros([batch_size,num_steps,1],dtype=np.float32);
        y=np.zeros([batch_size,num_steps],dtype=np.float32);
        for j in range(batch_size):
            for k in range(num_steps):
                x[j][k][0]=raw_data[batch_size*num_steps*i+batch_size*j+k]
                y[j][k]=raw_data[batch_size*num_steps*i+batch_size*j+k+1]
        yield(x,y)
  # batch_len = data_len // batch_size data = np.zeros([batch_size, batch_len], dtype=np.float32)
  # for i in range(batch_size):
    # data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

  # epoch_size = (batch_len - 1) // num_steps

  # if epoch_size == 0:
    # raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  # for i in range(epoch_size):
    # x = data[:, i*num_steps:(i+1)*num_steps]
    # y = data[:, i*num_steps+1:(i+1)*num_steps+1]
    # yield (x, y)


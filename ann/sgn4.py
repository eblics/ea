#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import os
import time
import random
import pickle
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True,precision=6)
np.set_printoptions(threshold=np.inf)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# tf.reset_default_graph()
NTRADERS=5
NHIDDEN=13
INITIAL_FOUNDS=1000.0
PERIOD=55
LOTS=1
GAP=10.0
GAP_FLOAT=0.00001*GAP
# FACTOR=1000000
FACTOR=np.exp(8)
EPOCH=100



def move():
    with tf.variable_scope('sgn',reuse=True):
        ov=tf.get_variable(name='order',dtype=tf.int32)
        opv=tf.get_variable(name='oop')
        bv=tf.get_variable(name='blances')
    x=tf.size(close_ph)-1
    price=close_ph[x]
    bmask=tf.cast(tf.equal(ov,0),tf.float32)
    bop=opv*bmask
    profit_ovs=tf.cast(tf.less_equal(bop,price-GAP_FLOAT),tf.int32)
    profit_ovs=profit_ovs*tf.cast(tf.greater(bop,0),tf.int32)
    bv=bv+tf.cast(profit_ovs,tf.float32)*LOTS*GAP
    ov=ov-profit_ovs
    opv=opv*tf.cast((1-profit_ovs),tf.float32)
    loss_ovs=tf.cast(tf.greater_equal(bop,price+GAP_FLOAT),tf.int32)
    bv=bv-tf.cast(loss_ovs,tf.float32)*LOTS*GAP
    ov=ov-loss_ovs
    opv=opv*tf.cast((1-loss_ovs),tf.float32)
    smask=tf.cast(tf.equal(ov,1),tf.float32)
    sop=opv*smask
    profit_ovs=tf.cast(tf.greater_equal(sop,price+GAP_FLOAT),tf.int32)
    bv=bv+tf.cast(profit_ovs,tf.float32)*LOTS*GAP
    ov=ov-2*profit_ovs
    opv=opv*tf.cast((1-profit_ovs),tf.float32)
    loss_ovs=tf.cast(tf.less_equal(sop,price-GAP_FLOAT),tf.int32)
    loss_ovs=loss_ovs*tf.cast(tf.greater(sop,0),tf.int32)
    ov=ov-2*loss_ovs
    opv=opv*tf.cast((1-loss_ovs),tf.float32)
    bv=bv-tf.cast(loss_ovs,tf.float32)*LOTS*GAP

    s=close_ph[x-PERIOD+1:x+1]-open_ph[x-PERIOD+1:x+1]
    y=decide4(s)
    ov=ov+tf.cast(tf.equal(ov,-1),tf.int32)*tf.argmax(y,1,output_type=tf.int32)
    cp=tf.cast(tf.equal(opv,0),tf.float32)*price
    opv=opv+cp
    opv=tf.cast(tf.greater(bv,0),tf.float32)*opv
    # opv=tf.Print(opv,[opv],message='opv',summarize=10)
    # ov=tf.Print(ov,[ov],message='ov',summarize=10)
    # bv=tf.Print(bv,[bv],message='bv',summarize=10)
    opv=tf.assign(oop,opv)
    ov=tf.assign(order,ov)
    bv=tf.assign(blances,bv)
    return ov,opv,bv

def decide(s):
    y=w1*s*FACTOR+b1
    y=tf.reduce_sum(y,axis=2,keepdims=True)
    y=tf.nn.elu(y)
    y=w2*y+b2_var
    y=tf.reduce_sum(y,axis=1,keepdims=True)
    y=tf.nn.tanh(y)
    y=w3*y
    y=tf.reduce_sum(y,axis=2)
    y=tf.nn.sigmoid(y)
    y=tf.nn.softmax(y)
    return y

def decide2(s):
    return np.random.random((NTRADERS,3))

def decide3(s):
    y=w1*s*FACTOR+b1
    y=tf.reduce_sum(y,axis=2)
    y=tf.reshape(y,shape=[NTRADERS,1,NHIDDEN])
    y=w3*y
    y=tf.reduce_sum(y,axis=2)
    y=tf.nn.sigmoid(y)
    y=tf.nn.softmax(y)
    return y

def decide4(s):
    y=w1*s*FACTOR+b1
    y=tf.reduce_sum(y,axis=2)
    y=tf.reshape(y,shape=[NTRADERS,1,NHIDDEN])
    y=tf.nn.sigmoid(y)
    y=w3*y
    y=tf.nn.sigmoid(y)
    y=tf.reduce_sum(y,axis=2)
    y=tf.nn.softmax(y)
    return y

def decide5(s):
    # y=tf.nn.elu(s)
    y=w1*s*FACTOR
    y=tf.reduce_sum(y,axis=2)
    y=tf.reshape(y,shape=[NTRADERS,1,NHIDDEN])
    y=w3*y
    y=tf.reduce_sum(y,axis=2)
    # y=tf.nn.sigmoid(y)
    y=tf.nn.softmax(y)
    return y

def cross(var,m):
    v2=tf.ones_like(var)
    v2=v2*var[m]
    v2=var*0.5+v2*0.5
    v2=tf.nn.dropout(v2,0.96)
    v2=tf.assign(var,v2)
    return v2


#def decide6(s):
#    # y=tf.nn.elu(s)
#    y=w4_var*s*FACTOR
#    y=tf.nn.sigmoid(y)
#    y=tf.reduce_sum(y,axis=2)
#    y=tf.nn.softmax(y)
#    return y

with tf.variable_scope('sgn',reuse=tf.AUTO_REUSE):
    initializer=tf.random_normal_initializer(mean=0, stddev=1)
    w1=tf.get_variable(name='w1',shape=[NTRADERS,NHIDDEN,PERIOD],initializer=initializer)
    b1=tf.get_variable(name='b1',shape=[NTRADERS,NHIDDEN,PERIOD],initializer=initializer)
    w2=tf.get_variable(name='w2',shape=[NTRADERS,NHIDDEN,NHIDDEN],initializer=initializer)
    b2=tf.get_variable(name='b2',shape=[NTRADERS,NHIDDEN,NHIDDEN],initializer=initializer)
    w3=tf.get_variable(name='w3',shape=[NTRADERS,3,NHIDDEN],initializer=initializer)
    blances=tf.get_variable(name='blances',shape=[NTRADERS],initializer=tf.constant_initializer(INITIAL_FOUNDS))
    order=tf.get_variable(name='order',shape=[NTRADERS],dtype=tf.int32,initializer=tf.constant_initializer(-1))
    oop=tf.get_variable(name='oop',shape=[NTRADERS],initializer=tf.constant_initializer(0))

open_ph=tf.placeholder(shape=[None],dtype=tf.float32,name='open')
close_ph=tf.placeholder(shape=[None],dtype=tf.float32,name='close')

def reinit():
    b=tf.assign(blances,tf.constant(INITIAL_FOUNDS,shape=[NTRADERS]))
    o=tf.assign(order,tf.constant(-1,shape=[NTRADERS],dtype=tf.int32))
    op=tf.assign(oop,tf.zeros(shape=[NTRADERS]))
    return [o,op,b]

def adapt():
    m=tf.argmax(blances)
    return [cross(v,m) for v in [w1,b1,w2,b2,w3]]

DUMP_PATH='winner.sgn'
def train(fname):
    # if os.path.exists(DUMP_PATH):
        # with open(DUMP_PATH,'rb') as f:
            # bm,w1,b1,w2,b2,w3=pickle.load(f)
    # df=pd.read_csv('test.csv.bak',header=None)
    df=pd.read_csv(fname,header=None)
    df.columns=['date','time','open','high','low','close','volume']
    xo=np.array(df['open'],dtype=np.float32)
    xc=np.array(df['close'],dtype=np.float32)
    sess=tf.Session()
    for e in range(0,EPOCH):
        sess.run(tf.global_variables_initializer())
        for i in range(PERIOD,len(df)):
            ixo=xo[i-PERIOD:i]
            ixc=xc[i-PERIOD:i]
            ov,opv,bv=sess.run(move(),feed_dict={open_ph:ixo,close_ph:ixc})
            print(bv)
        vars_adapt=sess.run(adapt())
        ove,opve,bve=sess.run(reinit())
        print('epoch:%d max:%f argmax:%d'%(e,bve.max(),bve.argmax()))
        # print('epoch:%d max:%f argmax:%d m:%d'%(e,bv.max(),bv.argmax(),m))
        # with open(DUMP_PATH,'wb') as f:
            # pickle.dump((m,w1,b1,w2,b2,w3),f)

def pred(fname):
    global loop,blances,norder,noop,w1,w2,b2,w3,blances_var,order,oop,w1,w2,b1,b2_var,w3
    if os.path.exists(DUMP_PATH):
        with open(DUMP_PATH,'rb') as f:
            bm,w1,b1,w2,b2,w3=pickle.load(f)
    df=pd.read_csv(fname,header=None)
    df.columns=['date','time','open','high','low','close','volume']
    xo=np.array(df['open'],dtype=np.float32)
    xc=np.array(df['close'],dtype=np.float32)
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    x,opv,ov,bv=sess.run(loop,feed_dict={open_ph:xo,close_ph:xc})
    print('argmax:%f maxv:%f m:%d mv:%f'%(bv.argmax(),bv.max(),bm,bv[bm]))

if sys.argv[1]=='train':
    train(sys.argv[2])
if sys.argv[1]=='pred':
    pred(sys.argv[2])

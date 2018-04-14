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
NTRADERS=100
NHIDDEN=13
INITIAL_FOUNDS=1000.0
PERIOD=55
LOTS=1
GAP=100.0
GAP_FLOAT=0.00001*GAP
# FACTOR=1000000
FACTOR=np.exp(8)
EPOCH=100



def succ(x,opv,ov,bv):
    price=dclose[x]
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
    # bv=tf.Print(bv,[bv])

    s=dclose[x-PERIOD+1:x+1]-dopen[x-PERIOD+1:x+1]
    y=decide5(s)
    ov=ov+tf.cast(tf.equal(ov,-1),tf.int32)*tf.argmax(y,1,output_type=tf.int32)
    cp=tf.cast(tf.equal(opv,0),tf.float32)*price
    opv=opv+cp
    opv=tf.cast(tf.greater(bv,0),tf.float32)*opv
    # bv=tf.Print(bv,[bv],message='bv:',summarize=20)
    return x,opv,ov,bv

def decide(s):
    y=w1_var*s*FACTOR+b1_var
    y=tf.reduce_sum(y,axis=2,keepdims=True)
    y=tf.nn.elu(y)
    y=w2_var*y+b2_var
    y=tf.reduce_sum(y,axis=1,keepdims=True)
    y=tf.nn.tanh(y)
    y=w3_var*y
    y=tf.reduce_sum(y,axis=2)
    y=tf.nn.sigmoid(y)
    y=tf.nn.softmax(y)
    return y

def decide2(s):
    return np.random.random((NTRADERS,3))

def decide3(s):
    y=w1_var*s*FACTOR+b1_var
    y=tf.reduce_sum(y,axis=2)
    y=tf.reshape(y,shape=[NTRADERS,1,NHIDDEN])
    y=w3_var*y
    y=tf.reduce_sum(y,axis=2)
    y=tf.nn.sigmoid(y)
    y=tf.nn.softmax(y)
    return y

def decide4(s):
    y=w1_var*s*FACTOR+b1_var
    y=tf.reduce_sum(y,axis=2)
    y=tf.reshape(y,shape=[NTRADERS,1,NHIDDEN])
    y=tf.nn.sigmoid(y)
    y=w3_var*y
    y=tf.nn.sigmoid(y)
    y=tf.reduce_sum(y,axis=2)
    y=tf.nn.softmax(y)
    return y

def decide5(s):
    # y=tf.nn.elu(s)
    y=w1_var*s*FACTOR
    y=tf.reduce_sum(y,axis=2)
    y=tf.reshape(y,shape=[NTRADERS,1,NHIDDEN])
    y=w3_var*y
    y=tf.reduce_sum(y,axis=2)
    # y=tf.nn.sigmoid(y)
    y=tf.nn.softmax(y)
    return y

def decide6(s):
    # y=tf.nn.elu(s)
    y=w4_var*s*FACTOR
    y=tf.nn.sigmoid(y)
    y=tf.reduce_sum(y,axis=2)
    y=tf.nn.softmax(y)
    return y

def fail(x,opv,ov,bv):
    return x,opv,ov,bv

def cond(x,opv,ov,bv):
    return tf.logical_and(x<tf.size(dclose),tf.reduce_any(tf.greater(bv,0)))

def body(x,opv,ov,bv):
    x,opv,ov,bv=tf.cond(x>=PERIOD-1,
        lambda:succ(x,opv,ov,bv),
        lambda:fail(x,opv,ov,bv))
    x+=1;#x=tf.Print(x,[x])
    return x,opv,ov,bv
# mean=tf.reduce_mean(blances_var)
blances=np.full((NTRADERS,),INITIAL_FOUNDS,dtype=np.float32)
w1=np.random.randn(NTRADERS,NHIDDEN,PERIOD)
b1=np.random.randn(NTRADERS,NHIDDEN,PERIOD)
w2=np.random.randn(NTRADERS,NHIDDEN,NHIDDEN)
b2=np.random.randn(NTRADERS,NHIDDEN,NHIDDEN)
w3=np.random.randn(NTRADERS,3,NHIDDEN)
w4=np.random.randn(NTRADERS,3,PERIOD)
#w1=w1*(w1<0.5)/np.sqrt(NTRADERS*PERIOD)
#b1=b1*(b1<0.5)/np.sqrt(NTRADERS*PERIOD)
#w2=w2*(w2<0.5)/np.sqrt(NTRADERS*PERIOD)
#b2=b2*(b2<0.5)/np.sqrt(NTRADERS*PERIOD)
#w3=w3*(w3<0.5)/np.sqrt(NTRADERS*PERIOD)
norder=np.random.randint(-1,0,(NTRADERS))
noop=np.zeros((NTRADERS),dtype=np.float32)

blances_var=tf.placeholder(shape=blances.shape,dtype=np.float32,name='blances')
w1_var=tf.placeholder(shape=w1.shape,dtype=tf.float32,name='w1')
b1_var=tf.placeholder(shape=b1.shape,dtype=tf.float32,name='b1')
w2_var=tf.placeholder(shape=w2.shape,dtype=tf.float32,name='w2')
b2_var=tf.placeholder(shape=b2.shape,dtype=tf.float32,name='b2')
w3_var=tf.placeholder(shape=w3.shape,dtype=tf.float32,name='w3')
w4_var=tf.placeholder(shape=w4.shape,dtype=tf.float32,name='w4')

dopen=tf.placeholder(shape=[None],dtype=tf.float32,name='open')
dclose=tf.placeholder(shape=[None],dtype=tf.float32,name='close')
order=tf.placeholder(shape=norder.shape,dtype=tf.int32,name='order')
oop=tf.placeholder(shape=noop.shape,dtype=np.float32,name='oop')
loop_i=tf.constant(0)
loop=tf.while_loop(cond,body,[loop_i,oop,order,blances_var])

DUMP_PATH='winner.sgn'
def train(fname):
    global loop,blances,norder,noop,w1,w2,b1,b2,w3,blances_var,order,oop,w1_var,w2_var,b1_var,b2_var,w3_var
    if os.path.exists(DUMP_PATH):
        with open(DUMP_PATH,'rb') as f:
            bm,w1,b1,w2,b2,w3=pickle.load(f)
    # df=pd.read_csv('test.csv.bak',header=None)
    df=pd.read_csv(fname,header=None)
    df.columns=['date','time','open','high','low','close','volume']
    xo=np.array(df['open'],dtype=np.float32)
    xc=np.array(df['close'],dtype=np.float32)
    sess=tf.Session()
    for e in range(0,EPOCH):
        sess.run(tf.global_variables_initializer())
        x,opv,ov,bv=sess.run(loop,feed_dict={dopen:xo,dclose:xc,w1_var:w1,b1_var:b1,w2_var:w2,b2_var:b2,w3_var:w3,w4_var:w4,oop:noop,order:norder,blances_var:blances})
        prop=0.5
        m=np.argmax(bv)
        print('epoch:%d max:%f argmax:%d m:%d'%(e,bv.max(),bv.argmax(),m))
        with open(DUMP_PATH,'wb') as f:
            pickle.dump((m,w1,b1,w2,b2,w3),f)
        for i in range(0,NTRADERS):
            if i==m:continue
            if random.random()>prop:
                w1[i]=np.random.randn(w1[i].shape[0],w1[i].shape[1])
                b1[i]=np.random.randn(b1[i].shape[0],b1[i].shape[1])
                w2[i]=np.random.randn(w2[i].shape[0],w2[i].shape[1])
                b2[i]=np.random.randn(b2[i].shape[0],b2[i].shape[1])
                w3[i]=np.random.randn(w3[i].shape[0],w3[i].shape[1])
                w4[i]=np.random.randn(w4[i].shape[0],w4[i].shape[1])
                if prop<0.99:prop+=0.001
            w1[i]=w1[m]*0.5+w1[i]*0.5
            b1[i]=b1[m]*0.5+b1[i]*0.5
            w2[i]=w2[m]*0.5+w2[i]*0.5
            b2[i]=b2[m]*0.5+b2[i]*0.5
            w3[i]=w3[m]*0.5+w3[i]*0.5
            w4[i]=w4[m]*0.5+w4[i]*0.5
        blances=np.full((NTRADERS,),INITIAL_FOUNDS,dtype=np.float32)
        norder=np.random.randint(-1,0,(NTRADERS))
        noop=np.zeros((NTRADERS),dtype=np.float32)
        # x,opv,ov,bv=sess.run(loop,feed_dict={open:xo,close:xc,w1_var:w1,b1_var:b1,w2_var:w2,b2_var:b2,w3_var:w3,oop:noop,order:norder,blances_var:blances})

def pred(fname):
    global loop,blances,norder,noop,w1,w2,b2,w3,blances_var,order,oop,w1_var,w2_var,b1_var,b2_var,w3_var
    if os.path.exists(DUMP_PATH):
        with open(DUMP_PATH,'rb') as f:
            bm,w1,w3=pickle.load(f)
    df=pd.read_csv(fname,header=None)
    df.columns=['date','time','open','high','low','close','volume']
    xo=np.array(df['open'],dtype=np.float32)
    xc=np.array(df['close'],dtype=np.float32)
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    x,opv,ov,bv=sess.run(loop,feed_dict={dopen:xo,dclose:xc,w1_var:w1,b1_var:b1,w2_var:w2,b2_var:b2,w3_var:w3,w4_var:w4,oop:noop,order:norder,blances_var:blances})
    print('argmax:%f maxv:%f m:%d mv:%f'%(bv.argmax(),bv.max(),bm,bv[bm]))

if sys.argv[1]=='train':
    train(sys.argv[2])
if sys.argv[1]=='pred':
    pred(sys.argv[2])

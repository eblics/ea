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
NTRADERS=3
NHIDDEN=3
INITIAL_FOUNDS=1000.0
PERIOD=5
LOTS=1
GAP=10.0
GAP_FLOAT=0.00001*GAP
# FACTOR=1000000
FACTOR=np.exp(8)
EPOCH=100


with tf.variable_scope('sgn',reuse=tf.AUTO_REUSE):
    initializer=tf.random_normal_initializer(mean=0, stddev=1)
    w1_var=tf.get_variable(name='w1',shape=[NTRADERS,NHIDDEN,PERIOD],initializer=initializer)
    b1_var=tf.get_variable(name='b1',shape=[NTRADERS,NHIDDEN,PERIOD],initializer=initializer)
    w2_var=tf.get_variable(name='w2',shape=[NTRADERS,NHIDDEN,NHIDDEN],initializer=initializer)
    b2_var=tf.get_variable(name='b2',shape=[NTRADERS,NHIDDEN,NHIDDEN],initializer=initializer)
    w3_var=tf.get_variable(name='w3',shape=[NTRADERS,3,NHIDDEN],initializer=initializer)
    blances_var=tf.get_variable(name='blances',shape=[NTRADERS],initializer=tf.constant_initializer(INITIAL_FOUNDS))
    order_var=tf.get_variable(name='order',shape=[NTRADERS],dtype=tf.int32,initializer=tf.constant_initializer(-1))
    oop_var=tf.get_variable(name='oop',shape=[NTRADERS],initializer=tf.constant_initializer(0))

open_ph=tf.placeholder(shape=[None],dtype=tf.float32,name='open')
close_ph=tf.placeholder(shape=[None],dtype=tf.float32,name='close')

def succ(x,bv,ov,opv):
    # with tf.variable_scope('sgn',reuse=True):
        # ov=tf.get_variable(name='order',dtype=tf.int32)
        # opv=tf.get_variable(name='oop')
        # bv=tf.get_variable(name='blances')

    # bv=tf.Print(bv,[bv],message='succ:')
    # x=tf.Print(x,[x],message="body:")
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
    # bv=tf.Print(bv,[bv])

    s=close_ph[x-PERIOD+1:x+1]-open_ph[x-PERIOD+1:x+1]
    y=decide(s)
    ov=ov+tf.cast(tf.equal(ov,-1),tf.int32)*tf.argmax(y,1,output_type=tf.int32)
    cp=tf.cast(tf.equal(opv,0),tf.float32)*price
    opv=opv+cp
    opv=tf.cast(tf.greater(bv,0),tf.float32)*opv
    # with tf.variable_scope('sgn',reuse=True):
        # bv=tf.assign(tf.get_variable(name='blances'),bv)
    return x,bv,ov,opv

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

#def decide6(s):
#    # y=tf.nn.elu(s)
#    y=w4_var*s*FACTOR
#    y=tf.nn.sigmoid(y)
#    y=tf.reduce_sum(y,axis=2)
#    y=tf.nn.softmax(y)
#    return y

def fail(x,bv,ov,opv):
    return x,bv,ov,opv

def cond(i,j,bv,ov,opv):
    with tf.variable_scope('sgn',reuse=tf.AUTO_REUSE):
        bv=tf.get_variable('blances')
    # return tf.logical_and(x<tf.size(close_ph),tf.reduce_any(tf.greater(bv,0)))
    # x=tf.Print(x,[x],message='body')
    return j<tf.size(close_ph)

def body(i,j,bv,ov,opv):
    j,bv,ov,opv=tf.cond(j>=PERIOD-1,
        lambda:succ(j,bv,ov,opv),
        lambda:fail(j,bv,ov,opv))
    j+=1;
    # j=tf.Print(j,[j],message='body')
    return i,j,bv,ov,opv

def ajust_body(k,m,prob):
    k=tf.cond(tf.not_equal(k,m),lambda:ajust(k,m,prob),lambda:k)
    k+=1
    return k,m,prob

def ajust(k,m,prob):
    with tf.variable_scope('sgn',reuse=tf.AUTO_REUSE):
        w1v=tf.get_variable(name='w1')
        b1v=tf.get_variable(name='b1')
        w2v=tf.get_variable(name='w2')
        b2v=tf.get_variable(name='b2')
        w3v=tf.get_variable(name='w3')

    tf.assign(w1v[k],w1v[k]*0.5+w1v[m]*0.5)
    tf.assign(w1v[k],tf.nn.dropout(w1v[k],prob))
    tf.assign(b1v[k],b1v[k]*0.5+b1v[m]*0.5,prob)
    tf.assign(b1v[k],tf.nn.dropout(b1v[k],prob))
    tf.assign(w2v[k],w2v[k]*0.5+w2v[m]*0.5,prob)
    tf.assign(w2v[k],tf.nn.dropout(w2v[k],prob))
    tf.assign(b2v[k],b2v[k]*0.5+b2v[m]*0.5)
    tf.assign(b2v[k],tf.nn.dropout(b2v[k],prob))
    tf.assign(w3v[k],w3v[k]*0.5+w3v[m]*0.5)
    tf.assign(w3v[k],tf.nn.dropout(w3v[k],prob))
    return k

def main_body(i,prob):
    with tf.variable_scope('sgn',reuse=True):
        ov=tf.get_variable(name='order',dtype=tf.int32)
        opv=tf.get_variable(name='oop')
        bv=tf.get_variable(name='blances')
        # bv=tf.assign(tf.get_variable('blances'),tf.ones([NTRADERS])*INITIAL_FOUNDS)
        # ov=tf.assign(tf.get_variable('order',dtype=tf.int32),tf.ones([NTRADERS],dtype=tf.int32)*-1)
        # opv=tf.assign(tf.get_variable('oop'),tf.zeros([NTRADERS],dtype=tf.float32))
    j=tf.constant(0)
    i,j,bv,ov,opv=tf.while_loop(cond,body,[i,j,bv,ov,opv])
    # bv=tf.Print(bv,[bv],'main:')
    # with tf.variable_scope('sgn',reuse=True):
        # bv=tf.assign(tf.get_variable('blances'),tf.ones_like(bv)*INITIAL_FOUNDS)
        # ov=tf.assign(tf.get_variable('order',dtype=tf.int32),tf.ones_like(ov,dtype=tf.int32)*-1)
        # opv=tf.assign(tf.get_variable('oop'),tf.zeros_like(ov,dtype=tf.float32))
    m=tf.argmax(bv,output_type=tf.int32)
    k=tf.constant(0)
    k,m,prob=tf.while_loop(lambda k,m,p:k<NTRADERS,ajust_body,[k,m,prob])
    i+=1
    # i=tf.Print(i,[i],message='main:')
    prob=tf.cond(prob<0.99,lambda:prob+0.01,lambda:prob)
    return i,prob


loop_i=tf.constant(0)
prob=tf.constant(0.5)
main_loop=tf.while_loop(lambda x,p:x<EPOCH,main_body,[loop_i,prob])
# loop=tf.while_loop(cond,body,[loop_i])

DUMP_PATH='winner.sgn'
def train(fname):
    if os.path.exists(DUMP_PATH):
        with open(DUMP_PATH,'rb') as f:
            bm,w1,b1,w2,b2,w3=pickle.load(f)
    # df=pd.read_csv('test.csv.bak',header=None)
    df=pd.read_csv(fname,header=None)
    df.columns=['date','time','open','high','low','close','volume']
    xo=np.array(df['open'],dtype=np.float32)
    xc=np.array(df['close'],dtype=np.float32)
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(main_loop,feed_dict={open_ph:xo,close_ph:xc})
    print(sess.run(blances_var))
    # with open(DUMP_PATH,'wb') as f:
        # pickle.dump((m,w1,b1,w2,b2,w3),f)

def pred(fname):
    global loop,blances,norder,noop,w1,w2,b2,w3,blances_var,order,oop,w1_var,w2_var,b1_var,b2_var,w3_var
    if os.path.exists(DUMP_PATH):
        with open(DUMP_PATH,'rb') as f:
            bm,w1,b1,w2,b2,w3=pickle.load(f)
    df=pd.read_csv(fname,header=None)
    df.columns=['date','time','open','high','low','close','volume']
    xo=np.array(df['open'],dtype=np.float32)
    xc=np.array(df['close'],dtype=np.float32)
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    x,opv,ov,bv=sess.run(loop,feed_dict={open_ph:xo,close_ph:xc,w1_var:w1,b1_var:b1,w2_var:w2,b2_var:b2,w3_var:w3,w4_var:w4,oop:noop,order:norder,blances_var:blances})
    print('argmax:%f maxv:%f m:%d mv:%f'%(bv.argmax(),bv.max(),bm,bv[bm]))

if sys.argv[1]=='train':
    train(sys.argv[2])
if sys.argv[1]=='pred':
    pred(sys.argv[2])

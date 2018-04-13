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
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True,precision=6)
np.set_printoptions(threshold=np.inf)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# tf.reset_default_graph()
NTRADERS=200
NHIDDEN=30
INITIAL_FOUNDS=1000.0
PERIOD=55
LOTS=1
GAP=100.0
GAP_FLOAT=0.001000
FACTOR=100000

blances=np.full((NTRADERS,),INITIAL_FOUNDS)
order_ops=np.full((NTRADERS,),-1)
order_open_prices=np.full((NTRADERS,),0.0)
order_takeprofits=np.full((NTRADERS,),0.0)
order_stoploss=np.full((NTRADERS,),0.0)
order_profit=np.full((NTRADERS,),0.0)

blances_var=tf.Variable(blances)
w1=np.random.random((NTRADERS,NHIDDEN,PERIOD))
b1=np.random.random((NTRADERS,NHIDDEN,PERIOD))
w2=np.random.random((NTRADERS,NHIDDEN,NHIDDEN))
b2=np.random.random((NTRADERS,NHIDDEN,NHIDDEN))
w3=np.random.random((NTRADERS,3,NHIDDEN))
# w1=w1/np.sqrt(NTRADERS)
# w2=w2/np.sqrt(NTRADERS)
# b1=b1/np.sqrt(NTRADERS)
w1=w1*(w1<0.5)/np.sqrt(NTRADERS*PERIOD)
b1=b1*(b1<0.5)/np.sqrt(NTRADERS*PERIOD)
w2=w2*(w2<0.5)/np.sqrt(NTRADERS*PERIOD)
b2=b2*(b2<0.5)/np.sqrt(NTRADERS*PERIOD)
w3=w3*(w3<0.5)/np.sqrt(NTRADERS*PERIOD)
# print(w1)
# print(w2)
w1_var=tf.Variable(w1)
b1_var=tf.Variable(b1)
w2_var=tf.Variable(w2)
b2_var=tf.Variable(b2)
w3_var=tf.Variable(w3)
#order_var=tf.Variable(np.random.random((3,NTRADERS)))
open=tf.placeholder(shape=[None],dtype=tf.float64)
close=tf.placeholder(shape=[None],dtype=tf.float64)
order=tf.Variable(np.random.randint(-1,0,(NTRADERS)))
oop=tf.Variable(np.zeros((NTRADERS),dtype=np.float64))

def succ(x,oop,order,open,close,blances_var,w1_var,w2_var,b1_var):
    price=close[x]
    bmask=tf.cast(tf.equal(order,0),tf.float64)
    bop=oop*bmask
    profit_orders=tf.cast(tf.less_equal(bop,price-GAP_FLOAT),tf.int64)
    profit_orders=profit_orders*tf.cast(tf.greater(bop,0),tf.int64)
    blances_var=blances_var+tf.cast(profit_orders,tf.float64)*LOTS*GAP
    order=order-profit_orders
    oop=oop*tf.cast((1-profit_orders),tf.float64)
    loss_orders=tf.cast(tf.greater_equal(bop,price+GAP_FLOAT),tf.int64)
    blances_var=blances_var-tf.cast(loss_orders,tf.float64)*LOTS*GAP
    order=order-loss_orders
    oop=oop*tf.cast((1-loss_orders),tf.float64)
    smask=tf.cast(tf.equal(order,1),tf.float64)
    sop=oop*smask
    profit_orders=tf.cast(tf.greater_equal(sop,price+GAP_FLOAT),tf.int64)
    blances_var=blances_var+tf.cast(profit_orders,tf.float64)*LOTS*GAP
    order=order-2*profit_orders
    oop=oop*tf.cast((1-profit_orders),tf.float64)
    loss_orders=tf.cast(tf.less_equal(sop,price-GAP_FLOAT),tf.int64)
    loss_orders=loss_orders*tf.cast(tf.greater(sop,0),tf.int64)
    order=order-2*loss_orders
    oop=oop*tf.cast((1-loss_orders),tf.float64)
    blances_var=blances_var-tf.cast(loss_orders,tf.float64)*LOTS*GAP

    s=close[x-PERIOD+1:x+1]-open[x-PERIOD+1:x+1]
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
    order=order+tf.cast(tf.equal(order,-1),tf.int64)*tf.argmax(y,1)
    cp=tf.cast(tf.equal(oop,0),tf.float64)*price
    oop=oop+cp
    oop=tf.cast(tf.greater(blances_var,0),tf.float64)*oop
    return x,oop,order,open,close,blances_var,w1_var,w2_var,b1_var
def fail(x,oop,order,open,close,blances_var,w1_var,w2_var,b1_var):
    return x,oop,order,open,close,blances_var,w1_var,w2_var,b1_var

i=tf.constant(0)
def cond(x,oop,order,open,close,blances_var,w1_var,w2_var,b1_var,mean):
    return tf.logical_and(x<tf.size(close),tf.reduce_any(tf.greater(blances_var,0)))

def body(x,oop,order,open,close,blances_var,w1_var,w2_var,b1_var,mean):
    x,oop,order,open,close,blances_var,w1_var,w2_var,b1_var=tf.cond(x>=PERIOD-1,
    lambda:succ(x,oop,order,open,close,blances_var,w1_var,w2_var,b1_var),
    lambda:fail(x,oop,order,open,close,blances_var,w1_var,w2_var,b1_var))
    x+=1
    # mean=tf.reduce_mean(blances_var)
    # mean=tf.Print(mean,[mean])
    x=tf.Print(x,[x])
    return x,oop,order,open,close,blances_var,w1_var,w2_var,b1_var,mean

mean=tf.reduce_mean(blances_var)
loop=tf.while_loop(cond,body,[i,oop,order,open,close,blances_var,w1_var,w2_var,b1_var,mean])

# x=np.array([0.9,0.8,0.7,0.9,0.6,0.4,1.2],dtype=np.float64)
# x=np.random.random((1000))
df=pd.read_csv('test.csv.bak',header=None)
# df=pd.read_csv('test.csv',header=None)
df.columns=['date','time','open','high','low','close','volume']
xo=np.array(df['open'])
xc=np.array(df['close'])
sess=tf.Session()
sess.run(tf.global_variables_initializer())
loop=sess.run(loop,feed_dict={open:xo,close:xc})
traders=loop[5]
print(traders)
# print(x)
# print('2')
# print(loop[2])
# print(loop[1])
# print(loop[4])

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
from socket import *
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True,precision=6)
np.set_printoptions(threshold=np.inf)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

NTRADERS=100
NHIDDEN=1
INITIAL_FOUNDS=1000.0
PERIOD=55
LOTS=0.5
GAP=40.0
GAP_FLOAT=0.00001*GAP
# FACTOR=1000000
FACTOR=np.exp(9)
EPOCH=20


# w1=np.random.randn(NTRADERS,NHIDDEN,PERIOD)
# w2=np.random.randn(NTRADERS,2,NHIDDEN)
stddev=1.0/tf.sqrt(PERIOD*NHIDDEN*1.0)
w1=tf.Variable(tf.random_normal(shape=[NTRADERS,NHIDDEN,PERIOD],mean=0.0,stddev=stddev))
b1=tf.Variable(tf.random_normal(shape=[NTRADERS,1,NHIDDEN],mean=0.0,stddev=stddev))
w2=tf.Variable(tf.random_normal(shape=[NTRADERS,2,NHIDDEN],mean=0.0,stddev=stddev))
with tf.variable_scope('temp',reuse=tf.AUTO_REUSE):
    init=tf.constant_initializer()
    w1_t=tf.get_variable(name='w1',shape=w1.shape,initializer=init)
    b1_t=tf.get_variable(name='b1',shape=b1.shape,initializer=init)
    w2_t=tf.get_variable(name='w2',shape=w2.shape,initializer=init)
# w2=np.random.randn(NTRADERS,2,NHIDDEN)
open_ph=tf.placeholder(shape=[None],dtype=tf.float32,name='open')
close_ph=tf.placeholder(shape=[None],dtype=tf.float32,name='close')

def decide(s):
    global w1,w2
    # return np.full((NTRADERS),1,dtype=np.int32)
    y=w1*s*FACTOR
    y=tf.reduce_sum(y,axis=2)
    y=tf.reshape(y,shape=[NTRADERS,1,NHIDDEN])
    y=y+b1
    y=tf.nn.tanh(y)
    y=w2*y
    y=tf.reduce_sum(y,axis=2)
    y=tf.nn.softmax(y)
    y=tf.argmax(y,axis=1,output_type=tf.int32)
    return y

DUMP_PATH='winner.sgn'
def checkorders(blances,orders,oop,iopen,ihigh,ilow,iclose):
    buy_orders=tf.cast(tf.equal(orders,1),tf.float32)
    buy_prices=oop*tf.cast(buy_orders,tf.float32)
    profit_buy_orders=tf.cast(tf.less_equal(buy_prices,iopen-GAP_FLOAT),tf.int32)
    profit_buy_orders=profit_buy_orders*tf.cast(tf.not_equal(buy_prices,0),tf.int32)
    lose_buy_orders=tf.cast(tf.greater_equal(buy_prices,iopen+GAP_FLOAT),tf.int32)
    blances+=tf.cast(profit_buy_orders,tf.float32)*GAP*LOTS
    blances-=tf.cast(lose_buy_orders,tf.float32)*GAP*LOTS

    sell_orders=tf.cast(tf.equal(orders,-1),tf.int32)
    sell_prices=oop*tf.cast(sell_orders,tf.float32)
    profit_sell_orders=tf.cast(tf.greater_equal(sell_prices,iopen-GAP_FLOAT),tf.int32)
    lose_sell_orders=tf.cast(tf.less_equal(sell_prices,iopen+GAP_FLOAT),tf.int32)
    lose_sell_orders=lose_sell_orders*tf.cast(tf.not_equal(sell_prices,0),tf.int32)
    blances+=tf.cast(profit_sell_orders,tf.float32)*LOTS*GAP
    blances-=tf.cast(lose_sell_orders,tf.float32)*LOTS*GAP

    orders=orders-profit_buy_orders
    orders=orders-lose_buy_orders
    orders=orders+profit_sell_orders
    orders=orders+lose_sell_orders

    oop=oop*tf.cast(tf.equal(profit_buy_orders,0),tf.float32)
    oop=oop*tf.cast(tf.equal(lose_buy_orders,0),tf.float32)
    oop=oop*tf.cast(tf.equal(profit_sell_orders,0),tf.float32)
    oop=oop*tf.cast(tf.equal(lose_sell_orders,0),tf.float32)
    return blances,orders,oop

def move(i,blances,orders,oop):
    iopen=open_ph[i]
    iclose=close_ph[i]
    # ihigh=df.iloc[i]['high']
    # ilow=df.iloc[i]['low']

    blances,orders,oop=checkorders(blances,orders,oop,iopen,iclose,None,None)
    xs=close_ph[i-PERIOD:i]-open_ph[i-PERIOD]
    decision=decide(xs)
    oop_now=decision*tf.cast(tf.equal(orders,0),tf.int32)
    oop=oop+tf.cast(oop_now,tf.float32)*iopen
    orders+=(decision)*tf.cast(tf.equal(orders,0),tf.int32)
    orders*=tf.cast(tf.greater(blances,0),tf.int32)
    i+=1
    return i,blances,orders,oop

def try_in_market():
    blances=tf.constant(INITIAL_FOUNDS,tf.float32,[NTRADERS])
    orders=tf.constant(0,tf.int32,[NTRADERS])
    oop=tf.constant(0,tf.float32,[NTRADERS])
    i=tf.constant(PERIOD)
    i,blances,orders,oop=tf.while_loop(lambda x,b,o,op:x<tf.size(close_ph),move,[i,blances,orders,oop])
    return blances,orders,oop

def cross(bit,biv):
    global w1,b1,w2
    w1_n=w1[bit]*0.5+w1[biv]*0.5
    b1_n=b1[bit]*0.5+b1[biv]*0.5
    w2_n=w2[bit]*0.5+w2[biv]*0.5
    w1_n=tf.assign(w1[bit],w1_n)
    b1_n=tf.assign(b1[bit],b1_n)
    w2_n=tf.assign(w2[bit],w2_n)
    return w1_n,w2_n

def reproduce(be):
    global w1,b1,w2
    with tf.variable_scope('temp',reuse=tf.AUTO_REUSE):
        w1_t=tf.get_variable(name='w1',shape=w1.shape)
        b1_t=tf.get_variable(name='b1',shape=b1.shape)
        w2_t=tf.get_variable(name='w2',shape=w2.shape)

    i=tf.constant(0)
    def body(x,w1_t,b1_t,w2_t):
        w1_t=tf.assign(w1[x],w1[x]*0.4+w1[be]*0.5+tf.random_normal(shape=w1[be].shape,stddev=stddev)*0.1)
        b1_t=tf.assign(b1[x],b1[x]*0.4+b1[be]*0.5+tf.random_normal(shape=b1[be].shape,stddev=stddev)*0.1)
        w2_t=tf.assign(w2[x],w2[x]*0.4+w2[be]*0.5+tf.random_normal(shape=w2[be].shape,stddev=stddev)*0.1)
        return x+1,w1_t,b1_t,w2_t
    i,w1_t,b1_t,w2_t=tf.while_loop(lambda x,w1,b1,w2:x<NTRADERS,body,[i,w1_t,b1_t,w2_t])
    return w1_t,b1_t,w2_t

def init(w1_value,b1_value,w2_value):
    global w1,b1,w2
    y=tf.assign(w1,w1_value)
    y=tf.assign(b1,b1_value)
    y=tf.assign(w2,w2_value)
    return y

def train(fname,vfname):
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    if os.path.exists(DUMP_PATH):
        with open(DUMP_PATH,'rb') as f:
            m,w1_value,b1_value,w2_value=pickle.load(f)
            sess.run(init(w1_value,b1_value,w2_value))

    columns=['date','time','open','high','low','close','volume']
    train_data=pd.read_csv(fname,header=None)
    train_data.columns=columns
    valid_data=pd.read_csv(vfname,header=None)
    valid_data.columns=columns
    # sess=tf.Session()
    for e in range(0,EPOCH):
        starttime=time.time()
        bt,ot,opt=sess.run(try_in_market(),feed_dict={open_ph:train_data['open'],close_ph:train_data['close']})
        bv,ov,opv=sess.run(try_in_market(),feed_dict={open_ph:valid_data['open'],close_ph:valid_data['close']})
        bit=bt.argmax()
        mit=bt[bit]
        biv=bv.argmax()
        miv=bv[biv]
        tiv=bv[bit]
        vit=bt[biv]
        best=bit
        sess.run(cross(bit,biv))
        sess.run(reproduce(best))
        # if tiv>INITIAL_FOUNDS and vit>INITIAL_FOUNDS:
            # best=bit
        # if tiv>INITIAL_FOUNDS and vit<INITIAL_FOUNDS:
            # best=bit
        # if tiv<INITIAL_FOUNDS and vit>INITIAL_FOUNDS:
            # best=biv
        with open(DUMP_PATH,'wb') as f:
            w1_value,b1_value,w2_value=sess.run([w1,b1,w2])
            pickle.dump((best,w1_value,b1_value,w2_value),f)
        endtime=time.time()
        # print('epoch:',e,' time:',endtime-starttime,'bit:',bit,' mit:',mit,'tiv:',tiv,'vit:',vit,'biv:',biv,'miv:',miv)
        print('epoch:%2d time:%3d bit:%3d biv:%3d mit:%5f miv:%5f tiv:%5f vit:%5f'%
              (e,endtime-starttime,bit,biv,mit,miv,tiv,vit))

HOST='0.0.0.0'
PORT=8899
ADDR=(HOST, PORT)
BUFSIZE=64

def listen():
    global loop,blances,norder,noop,w1,w2,b2,w3,blances_var,order,oop,w1,w2_var,b1_var,b2_var,w3_var
    if not os.path.exists(DUMP_PATH):
        print('there is no traders availeble');return
    with open(DUMP_PATH,'rb') as f:
        m,w1,b1,w2,b2,w3=pickle.load(f)
    columns=['date','time','open','high','low','close','volume']
    df=pd.DataFrame(columns=columns)
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    bar=0
    sock=socket(AF_INET, SOCK_STREAM)
    sock.setsockopt(SOL_SOCKET,SO_REUSEADDR, 1)
    sock.bind(ADDR)
    sock.listen(5)
    print('waiting for connection')
    while True:
        tcpClientSock, addr=sock.accept()
        try:
            data=tcpClientSock.recv(BUFSIZE)
            strs=data.decode().strip('\x00').split(',')
            # d=[strs[0],strs[1],float(strs[3]),float(strs[4]),float(str[5]),float(str[6])]
            date=strs[0]
            minute=strs[1]
            topen=float(strs[2])
            thigh=float(strs[3])
            tlow=float(strs[4])
            tclose=float(strs[5])
            d=[[date,minute,topen,thigh,tlow,tclose,0]]
            # print(d)
            item=pd.DataFrame(d,columns=columns)
            df=pd.concat([df,item],ignore_index=True)
            if len(df)<PERIOD:
                print("df is %d"%(len(df)))
                tcpClientSock.send('-1'.encode())
                continue
            index=len(df)
            xo=np.array(df['open'][index-PERIOD:index],dtype=np.float32)
            xc=np.array(df['close'][index-PERIOD:index],dtype=np.float32)
            x,opv,ov,bv=sess.run(loop,feed_dict={dopen:xo,dclose:xc,w1:w1,b1_var:b1,w2_var:w2,b2_var:b2,w3_var:w3,w4_var:w4,oop:noop,order:norder,blances_var:blances})
            print('df_len:%d m:%d bvm:%f bv.max:%f ovm:%d'%(index,m,bv[m],bv.max(),ov[m]))
            tcpClientSock.send(str(ov[m]).encode())
            if bar>30:
                df.to_csv(time.strftime('%Y_%m_%d',time.gmtime(time.time()))+'.csv',header=None,index=None)
                bar=0
            bar+=1
        except KeyboardInterrupt:
            tcpClientSock.close()
        except:
            tcpClientSock.close()
            raise()
    tcpClientSock.close()

if sys.argv[1]=='train':
    train(sys.argv[2],sys.argv[3])
if sys.argv[1]=='pred':
    pred(sys.argv[2])
if sys.argv[1]=='listen':
    listen()

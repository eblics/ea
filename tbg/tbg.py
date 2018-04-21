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
EPOCH=50


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
high_ph=tf.placeholder(shape=[None],dtype=tf.float32,name='high')
low_ph=tf.placeholder(shape=[None],dtype=tf.float32,name='low')

blances_ph=tf.placeholder(shape=[NTRADERS],dtype=tf.float32,name='blances')
orders_ph=tf.placeholder(shape=[NTRADERS],dtype=tf.int32,name='orders')
oop_ph=tf.placeholder(shape=[NTRADERS],dtype=tf.float32,name='oop')
best_ph=tf.placeholder(dtype=tf.int32)

def decide(s):
    # global w1,w2
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
    # y=tf.Print(y,[y],message='decision1:')
    y=y+(tf.cast(tf.equal(y,0),tf.int32)*-1)
    # y=tf.Print(y,[y],message='decision2:')
    return y

DUMP_PATH='winner.sgn'
def checkorders(blances,orders,oop,iopen,ihigh,ilow,iclose):
    buy_orders=tf.cast(tf.equal(orders,1),tf.float32)
    buy_prices=oop*tf.cast(buy_orders,tf.float32)
    profit_buy_orders=tf.cast(tf.less_equal(buy_prices,ihigh-GAP_FLOAT),tf.int32)
    profit_buy_orders=profit_buy_orders*tf.cast(tf.not_equal(buy_prices,0),tf.int32)
    lose_buy_orders=tf.cast(tf.greater_equal(buy_prices,ilow+GAP_FLOAT),tf.int32)
    blances+=tf.cast(profit_buy_orders,tf.float32)*GAP*LOTS
    blances-=tf.cast(lose_buy_orders,tf.float32)*GAP*LOTS

    sell_orders=tf.cast(tf.equal(orders,-1),tf.int32)
    sell_prices=oop*tf.cast(sell_orders,tf.float32)
    profit_sell_orders=tf.cast(tf.greater_equal(sell_prices,ilow+GAP_FLOAT),tf.int32)
    lose_sell_orders=tf.cast(tf.less_equal(sell_prices,ihigh-GAP_FLOAT),tf.int32)
    lose_sell_orders=lose_sell_orders*tf.cast(tf.not_equal(sell_prices,0),tf.int32)
    blances+=tf.cast(profit_sell_orders,tf.float32)*LOTS*GAP
    blances-=tf.cast(lose_sell_orders,tf.float32)*LOTS*GAP

    # orders=tf.Print(orders,[orders],message='corders1:')
    # profit_buy_orders=tf.Print(profit_buy_orders,[profit_buy_orders],message='cprofit_buy_orders1:')
    # lose_buy_orders=tf.Print(lose_buy_orders,[lose_buy_orders],message='close_buy_orders1:')
    # profit_sell_orders=tf.Print(profit_sell_orders,[profit_sell_orders],message='cprofit_sell_orders1:')
    # lose_sell_orders=tf.Print(lose_sell_orders,[lose_sell_orders],message='close_sell_orders1:')

    orders=orders-profit_buy_orders
    # orders=tf.Print(orders,[orders],message='corders2')
    orders=orders-lose_buy_orders
    # orders=tf.Print(orders,[orders],message='corders3')
    orders=orders+profit_sell_orders
    # orders=tf.Print(orders,[orders],message='corders4')
    orders=orders+lose_sell_orders
    # orders=tf.Print(orders,[orders],message='corders5')

    oop=oop*tf.cast(tf.equal(profit_buy_orders,0),tf.float32)
    oop=oop*tf.cast(tf.equal(lose_buy_orders,0),tf.float32)
    oop=oop*tf.cast(tf.equal(profit_sell_orders,0),tf.float32)
    oop=oop*tf.cast(tf.equal(lose_sell_orders,0),tf.float32)
    return blances,orders,oop

def move(i,blances,orders,oop):
    iopen=open_ph[i-1]
    iclose=close_ph[i-1]
    ihigh=high_ph[i-1]['high']
    ilow=low_ph[i-1]['low']

    blances,orders,oop=checkorders(blances,orders,oop,iopen,None,None,iclose)
    xs=close_ph[i-PERIOD:i]-open_ph[i-PERIOD]
    decision=decide(xs)
    oop_now=tf.not_equal(decision*tf.cast(tf.equal(orders,0),tf.int32),0)
    # orders=tf.Print(orders,[orders],message='orders1:')
    # oop_now=tf.Print(oop_now,[oop_now],message='oop_now:')
    # decision=tf.Print(decision,[decision],message='decision:')
    oop=oop+tf.cast(oop_now,tf.float32)*iclose
    # oop=tf.Print(oop,[oop],message='oop:')
    orders+=(decision)*tf.cast(tf.equal(orders,0),tf.int32)
    orders*=tf.cast(tf.greater(blances,0),tf.int32)
    # orders=tf.Print(orders,[orders],message='orders2:')
    i+=1
    # i=tf.Print(i,[i],message='i')
    # blances=tf.Print(blances,[blances],message='blances:')
    # orders=tf.Print(orders,[orders],message='orders:')
    # oop=tf.Print(oop,[oop],message='oop:')
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

def init(sess,w1_value,b1_value,w2_value):
    sess.run(tf.assign(w1,w1_value))
    sess.run(tf.assign(b1,b1_value))
    sess.run(tf.assign(w2,w2_value))

def train(fname):
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    if os.path.exists(DUMP_PATH):
        with open(DUMP_PATH,'rb') as f:
            m,w1_value,b1_value,w2_value=pickle.load(f)
            init(sess,w1_value,b1_value,w2_value)

    columns=['date','time','open','high','low','close','volume']
    train_data=pd.read_csv(fname,header=None)
    train_data.columns=columns
    blances,orders,oop=try_in_market()
    reproduce_op=reproduce(best_ph)
    for e in range(0,EPOCH):
        starttime=time.time()
        bt,ot,opt=sess.run([blances,orders,oop],feed_dict={open_ph:train_data['open'],close_ph:train_data['close'],high_ph:train_data['high'],low_ph:train_data['low']})
        bit=bt.argmax()
        mit=bt[bit]
        best=bit
        # if os.path.exists(DUMP_PATH):
            # os.rename(DUMP_PATH,DUMP_PATH+'_'+time.strftime('%Y_%m_%d_%H_%M_%S',time.gmtime(time.time())))
        with open(DUMP_PATH,'wb') as f:
            w1_value,b1_value,w2_value=sess.run([w1,b1,w2])
            pickle.dump((best,w1_value,b1_value,w2_value),f)
        sess.run(reproduce_op,feed_dict={best_ph:best})
        endtime=time.time()
        print('epoch:%2d time:%3d bit:%3d mit:%5f'%
              (e,endtime-starttime,bit,mit))

def train2(fname):
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    if os.path.exists(DUMP_PATH):
        with open(DUMP_PATH,'rb') as f:
            bp,w1_value,b1_value,w2_value=pickle.load(f)
            init(sess,w1_value,b1_value,w2_value)

    columns=['date','time','open','high','low','close','volume']
    data=pd.read_csv(fname,header=None)
    data.columns=columns
    _,op_b,op_o,op_op=move(PERIOD,blances_ph,orders_ph,oop_ph)
    reproduce_op=reproduce(best_ph)
    for e in range(0,EPOCH):
        blances=np.full((NTRADERS),INITIAL_FOUNDS,dtype=np.float32)
        orders=np.full((NTRADERS),0,dtype=np.int32)
        oop=np.full((NTRADERS),0,dtype=np.float32)
        b,o,op=blances,orders,oop
        bt=0
        starttime=time.time()
        for i in range(PERIOD,len(data)-1):
            b,o,op=sess.run([op_b,op_o,op_op],
                feed_dict={open_ph:data[i-PERIOD:i]['open'],
                close_ph:data[i-PERIOD:i]['close'],
                high_ph:data[i-PERIOD:i]['high'],
                low_ph:data[i-PERIOD:i]['low'],
                blances_ph:b,
                orders_ph:o,
                oop_ph:op})
            # print(b,o,op,op-data.iloc[i]['open'])
        bit=b.argmax()
        mit=b[bit]
        with open(DUMP_PATH,'wb') as f:
            w1_value,b1_value,w2_value=sess.run([w1,b1,w2])
            pickle.dump((bit,w1_value,b1_value,w2_value),f)
        sess.run(reproduce_op,feed_dict={best_ph:bit})
        print('e:%5d time:%3d bit:%d mit:%5d'%(e,time.time()-starttime,bit,mit))


def online(fname):
    blances=np.full((NTRADERS),INITIAL_FOUNDS,dtype=np.float32)
    orders=np.full((NTRADERS),0,dtype=np.int32)
    oop=np.full((NTRADERS),0,dtype=np.float32)
    bp_ph=tf.placeholder(tf.int32)
    bn_ph=tf.placeholder(tf.int32)
    _,op_b,op_o,op_op=move(PERIOD,blances_ph,orders_ph,oop_ph)
    cross_op=cross(bp_ph,bn_ph)
    reproduce_op=reproduce(best_ph)

    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    if os.path.exists(DUMP_PATH):
        with open(DUMP_PATH,'rb') as f:
            m,w1_value,b1_value,w2_value=pickle.load(f)
            init(sess,w1_value,b1_value,w2_value)

    columns=['date','time','open','high','low','close','volume']
    data=pd.read_csv(fname,header=None)
    data.columns=columns
    mybalance1=INITIAL_FOUNDS
    mybalance2=INITIAL_FOUNDS
    b,o,op=blances,orders,oop
    starttime=time.time()
    blance_t=INITIAL_FOUNDS
    bp=m
    for i in range(PERIOD,len(data)-1):
        b,o,op=sess.run([op_b,op_o,op_op],
            feed_dict={
            open_ph:data[i-PERIOD:i]['open'],
            close_ph:data[i-PERIOD:i]['close'],
            high_ph:data[i-PERIOD:i]['high'],
            low_ph:data[i-PERIOD:i]['low'],
            blances_ph:b,
            orders_ph:o,
            oop_ph:op})
        # if i%PERIOD==0:
            # bn=np.argmax(b)
            # bp=bn
            # sess.run(cross_op,feed_dict={bp_ph:bp,bn_ph:bn})
            # sess.run(reproduce_op,feed_dict={best_ph:bn})
        if blance_t==b[bp]:continue
        mybalance1=b[m]
        mybalance2+=b[bp]-blance_t
        bn=np.argmax(b)
        bp=bn
        blance_t=b[bp]
        print('i:%5d time:%3d best:%2d max:%5d mybalance1:%5d  mybalance2:%5d'%
              (i,time.time()-starttime,bn,b[bn],mybalance1,mybalance2))
    # with open(DUMP_PATH,'wb') as f:
        # w1_value,b1_value,w2_value=sess.run([w1,b1,w2])
        # pickle.dump((bp,w1_value,b1_value,w2_value),f)

HOST='0.0.0.0'
PORT=8899
ADDR=(HOST, PORT)
BUFSIZE=64

def listen():
    blances=np.full((NTRADERS),INITIAL_FOUNDS,dtype=np.float32)
    orders=np.full((NTRADERS),0,dtype=np.int32)
    oop=np.full((NTRADERS),0,dtype=np.float32)
    _,op_b,op_o,op_op=move(PERIOD,blances_ph,orders_ph,oop_ph)

    b,o,op=blances,orders,oop
    columns=['date','time','open','high','low','close','volume']
    df=pd.DataFrame(columns=columns)
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())

    if not os.path.exists(DUMP_PATH):
        print('there is no traders availeble');return
    if os.path.exists(DUMP_PATH):
        with open(DUMP_PATH,'rb') as f:
            m,w1_value,b1_value,w2_value=pickle.load(f)
            init(sess,w1_value,b1_value,w2_value)

    bar=0
    sock=socket(AF_INET, SOCK_STREAM)
    sock.setsockopt(SOL_SOCKET,SO_REUSEADDR, 1)
    sock.bind(ADDR)
    sock.listen(5)
    print('waiting for connection')
    bt,open_price,close_price=0.0,0.0,0.0
    open_time=''
    d=0
    while True:
        tcpClientSock, addr=sock.accept()
        try:
            data=tcpClientSock.recv(BUFSIZE)
            strs=data.decode().strip('\x00').split(',')
            # d=[strs[0],strs[1],float(strs[3]),float(strs[4]),float(str[5]),float(str[6])]
            date=strs[0];minute=strs[1];topen=float(strs[2]);thigh=float(strs[3]);tlow=float(strs[4]);tclose=float(strs[5])
            d=[[date,minute,topen,thigh,tlow,tclose,0]]
            item=pd.DataFrame(d,columns=columns)
            df=pd.concat([df,item],ignore_index=True)
            if len(df)<PERIOD:
                tcpClientSock.send('0'.encode())
                continue
            index=len(df)
            xo=np.array(df['open'][index-PERIOD:index],dtype=np.float32)
            xc=np.array(df['close'][index-PERIOD:index],dtype=np.float32)
            b,o,op=sess.run([op_b,op_o,op_op],feed_dict={open_ph:xo,close_ph:xc,blances_ph:b,orders_ph:o,oop_ph:op})

            if open_price==0:open_price=df.iloc[index-1]['close'];open_time=df.iloc[index-1]['date']+' '+df.iloc[index-1]['time']
            if abs(b[m]-bt)>=LOTS*GAP:
                close_price=df.iloc[index-1]['close']
                d=o[m]
                print('%s open at %f close at %f dir:%d profit:%f'%(open_time,open_price,close_price,d,(b[m]-bt)))
                open_price=close_price
                open_time=df.iloc[index-1]['date']+' '+df.iloc[index-1]['time']
                bt=b[m]
                am=b.argmax()
                nmax=b[am]
                print('oopm:%5f bm:%5d om:%1d am:%2d max:%5d'%(op[m],b[m],o[m],am,nmax))
            tcpClientSock.send(str(o[m]).encode())
            if bar>30:
                df.to_csv(time.strftime('%Y_%m_%d',time.gmtime(time.time()))+'.csv',header=None,index=None);bar=0;
            bar+=1
        except KeyboardInterrupt:
            tcpClientSock.close()
        except:
            tcpClientSock.close()
            raise()
    tcpClientSock.close()


if sys.argv[1]=='train':
    train(sys.argv[2])
if sys.argv[1]=='train2':
    train2(sys.argv[2])
# if sys.argv[1]=='pred':
    # pred(sys.argv[2])
if sys.argv[1]=='listen':
    listen()
if sys.argv[1]=='online':
    online(sys.argv[2])

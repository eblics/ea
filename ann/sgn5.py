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
NHIDDEN=30
INITIAL_FOUNDS=1000.0
PERIOD=55
LOTS=0.5
GAP=50.0
GAP_FLOAT=0.00001*GAP
# FACTOR=1000000
FACTOR=np.exp(8)
EPOCH=20


def decide(s,w1,w2):
    # return np.full((NTRADERS),1,dtype=np.int32)
    y=w1*s*FACTOR
    y=np.sum(y,axis=2)
    y=y.reshape((NTRADERS,1,NHIDDEN))
    y=w2*y
    y=np.sum(y,axis=2)
    y=softmax(y)
    y=np.argmax(y,axis=1)
    return y

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

DUMP_PATH='winner.sgn'
def checkorders(blances,orders,oop,iopen,ihigh,ilow,iclose):
    # print('checkprice:',iopen)
    # print('iopen:%f ihigh:%f ilow:%f iclose:%f'%(iopen,ihigh,ilow,iclose))
    buy_orders=np.equal(orders,1)
    buy_prices=oop*buy_orders
    profit_buy_orders=buy_prices<=(iopen-GAP_FLOAT)
    profit_buy_orders=profit_buy_orders*np.not_equal(buy_prices,0)
    lose_buy_orders=buy_prices>=(iopen+GAP_FLOAT)
    blances+=profit_buy_orders*LOTS*GAP
    blances-=lose_buy_orders*LOTS*GAP

    sell_orders=np.equal(orders,-1)
    sell_prices=oop*sell_orders
    profit_sell_orders=sell_prices>=(iopen-GAP_FLOAT)
    lose_sell_orders=sell_prices<=(iopen+GAP_FLOAT)
    lose_sell_orders=lose_sell_orders*np.not_equal(sell_prices,0)
    blances+=profit_sell_orders*LOTS*GAP
    blances-=lose_sell_orders*LOTS*GAP
    # print('orders before:',orders)
    orders=orders-profit_buy_orders
    # print('orders after profit buy:',orders)
    orders=orders-lose_buy_orders
    orders=orders+profit_sell_orders
    orders=orders+lose_sell_orders
    # print('orders after:',orders)

    # print("buy profit:",profit_buy_orders,'buy loss:',lose_buy_orders,'sell profit:',profit_sell_orders,'sell lose:',lose_sell_orders)
    # print('oop before:',oop)
    oop=oop*np.equal(profit_buy_orders,0)
    oop=oop*np.equal(lose_buy_orders,0)
    oop=oop*np.equal(profit_sell_orders,0)
    oop=oop*np.equal(lose_sell_orders,0)
    # print('oop after:',oop)
    return blances,orders,oop

def try_in_market(df,w1,w2):
    blances=np.full((NTRADERS),INITIAL_FOUNDS,dtype=np.float32)
    orders=np.full((NTRADERS),0,dtype=np.int32)
    oop=np.full((NTRADERS),0,dtype=np.float32)
    for i in range(PERIOD,len(df)):
        iopen=df.iloc[i]['open']
        iclose=df.iloc[i]['close']
        ihigh=df.iloc[i]['high']
        ilow=df.iloc[i]['low']

        blances,orders,oop=checkorders(blances,orders,oop,iopen,ihigh,ilow,iclose)
        period_data=df[i-PERIOD:i]
        xs=period_data['close']-period_data['open']
        decision=decide(np.array(xs),w1,w2)
        # print('decision:',decision)
        # print('oop before desision:',oop)
        oop_now=decision*np.equal(orders,0)
        oop=oop+oop_now*iopen
        # print('oop after desision:',oop)
        orders+=(decision)*np.equal(orders,0)
        # print('orderrs after decision:',orders)
        # print('open:%f high:%f low:%f close:%f'%(iopen,ihigh,ilow,iclose))
        # print('blances:',blances)
        # print('orders:',orders)
        # print('oop:',oop)
        # print('=============================')
    return blances,orders,oop

def train(fname,vfname):
    w1=np.random.randn(NTRADERS,NHIDDEN,PERIOD)
    w2=np.random.randn(NTRADERS,2,NHIDDEN)
    if os.path.exists(DUMP_PATH):
        with open(DUMP_PATH,'rb') as f:
            m,w1,w2=pickle.load(f)
    columns=['date','time','open','high','low','close','volume']
    train_data=pd.read_csv(fname,header=None)
    train_data.columns=columns
    valid_data=pd.read_csv(vfname,header=None)
    valid_data.columns=columns
    # sess=tf.Session()
    for e in range(0,EPOCH):
        starttime=time.time()
        # sess.run(tf.global_variables_initializer())
        bt,ot,opt=try_in_market(train_data,w1,w2)
        bv,ov,opv=try_in_market(valid_data,w1,w2)
        bit=bt.argmax()
        mit=bt[bit]
        biv=bv.argmax()
        miv=bv[biv]
        tiv=bv[bit]
        vit=bt[biv]
        best=bit
        if tiv>INITIAL_FOUNDS and vit>INITIAL_FOUNDS:
            w1[bit]=w1[bit]*0.5+w1[biv]*0.5
        if tiv>INITIAL_FOUNDS and vit<INITIAL_FOUNDS:
            best=bit
        if tiv<INITIAL_FOUNDS and vit>INITIAL_FOUNDS:
            best=biv
        for i in range(0,NTRADERS):
            w1[i]=w1[i]*0.5+w1[best]*0.5#+0.01*np.random.randn(NHIDDEN,PERIOD)
            w2[i]=w2[i]*0.5+w2[best]*0.5#+0.01*np.random.randn(2,NHIDDEN)
        with open(DUMP_PATH,'wb') as f:
            pickle.dump((best,w1,w2),f)
        endtime=time.time()
        print('epoch:',e,' time:',endtime-starttime,'bit:',bit,' mit:',mit,'biv:',biv,'miv:',miv)

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

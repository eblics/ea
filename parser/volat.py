#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import shutil
import os
import time
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python import debug as tfd
from tensorflow.contrib.cudnn_rnn.python.ops.cudnn_rnn_ops import CudnnLSTM
import json
import BaseHTTPServer
import random
# from tensorflow.contrib.cudnn_rnn.python.layers.cudnn_rnn import CudnnLSTM
flags = tf.flags
flags.DEFINE_string('symbol', 'AUDUSD', '货币对')
flags.DEFINE_string('model_path', './model', '模型地址')
flags.DEFINE_bool('debug', False, '货币对')
flags.DEFINE_bool('gpu',True, '')
flags.DEFINE_string('period','h1', '')
FLAGS=flags.FLAGS
#data=data[::-1]      #反转，使数据按照日期先后顺序排列
#以折线图展示data
#plt.figure()
#plt.plot(data)
#plt.show()
# normalize_data=(data-np.mean(data))/np.std(data)  #标准化
#———————————————————形成训练集—————————————————————
#设置常量
num_steps= 20     #时间步
num_layers=5
num_units= 2000    #hidden layer units
num_trtimes=5
num_epochs=10000
keep_prob=0.5
batch_size=400   #每一批次训练多少个样例
input_size=2      #输入层维度
output_size=1     #输出层维度
grad_normal=5
lr=0.00001     #学习率
last_lr=lr
lr_decay=0.9      #学习率衰减系数
is_training=True
chunk_size=1000;
global_mean=1

def get_dataframe(symbol,period):
    dfs=[]
    files=['train','valid','test']
    for path in files:
        f=open('./data/%s/%s/%s.txt'%(symbol,period,path))
        df=pd.read_csv(f,sep='\t',chunksize=chunk_size)
        dfs.append(df)
    return (dfs)
def iterate_chunk(df,norm=True):
    dbuffer=[]
    for da in df:
        dfao=np.array(da['OPEN'])
        dfac=np.array(da['CLOSE'])
        # if norm:
            # dfa=(dfa-dfa.mean())/dfa.std()
        for i in range(len(da)):
            if len(dbuffer)<batch_size+num_steps+1:
                dbuffer.append([dfao[i],dfac[i]])
            else:
                d=np.array(dbuffer[:])
                d=d[:,np.newaxis]  #增加维度
                xa,ya=[],[]
                ni=len(d)-num_steps-1
                for j in range(ni):
                    x=d[j:j+num_steps]
                    y=d[j+1:j+num_steps+1]
                    xa.append(x)
                    ya.append(y)
                dbuffer=dbuffer[-num_steps+1:]
                xa=np.array(xa)
                ya=np.array(ya)
                xa=xa.reshape(batch_size,num_steps,input_size)
                ya=ya.reshape(batch_size,num_steps,input_size)
                yield (xa,ya)

LR= tf.placeholder(tf.float32, shape=[])

def train1():
    X=tf.placeholder(tf.float32, [batch_size,num_steps,input_size])    #每批次输入网络的tensor
    Y=tf.placeholder(tf.float32, [batch_size,num_steps,input_size]) #每批次tensor对应的标签
    # softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=data_type())
    # softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    # logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
    # w_out=tf.Variable(tf.random_normal(shape=[num_units,input_size],mean=global_mean,name="w_out",dtype=tf.float32))
    # b_out=tf.Variable(tf.random_normal(shape=[input_size],mean=global_mean,name="b_out",dtype=tf.float32))
    # cell=tf.contrib.rnn.LSTMBlockFusedCell(num_units, forget_bias=-0.2,cell_clip=-0.2)
    global lr,lr_decay,is_training
    save_path='./model/'+FLAGS.symbol
    perp,perp_last=0.0,float('inf')
    cell=tf.contrib.rnn.LSTMBlockCell(num_units, forget_bias=-0.2,cell_clip=-0.2)
    w = tf.get_variable("w", [num_units, input_size],tf.float32)
    b = tf.get_variable("b", [input_size],tf.float32 )
    if is_training:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    layers = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)], state_is_tuple=True)
    inputs=tf.transpose(X,[1,0,2])
    outputs, state = tf.nn.dynamic_rnn(layers,inputs=inputs,dtype=tf.float32)
    outputs=tf.reshape(outputs,[-1,num_units])
    outputs = tf.nn.xw_plus_b(outputs,w,b)
    outputs=tf.reshape(outputs,[batch_size,num_steps,input_size])
    loss=tf.losses.mean_squared_error(outputs,Y)
    # tf.losses.absolute_difference(output,Y)
    # loss=tf.square(Y-output)
    # loss=tr.sqrt(tf.reduce_sum(loss,1))
    # p_loss=tf.Print(tf.shape(loss),[tf.shape(loss)])
    # loss=tf.reduce_sum(loss)/(batch_size*num_steps)
    train_op=tf.train.AdamOptimizer(LR).minimize(loss)
    cost,iters=0,0;
    saver=tf.train.Saver()

    with tf.Session() as sess:
        if FLAGS.debug:
            sess = tfd.LocalCLIDebugWrapperSession(sess)
        sess.run(tf.global_variables_initializer())
        for ep in range(1,num_epochs+1):
            trd,vd,td=get_dataframe(FLAGS.symbol,FLAGS.period)
#===============train begin=============================#yy
            is_training=True
            start_time=time.time()
            iters=1
            cost=0
            for x,y in iterate_chunk(trd):
                try:
                    _,cost_=sess.run([train_op,loss],feed_dict={LR:lr,X:x,Y:y})
                    iters+=1
                    cost+=cost_
                except ValueError,e:
                    print e
                    pass;
            tr_perp=np.sum(cost)/iters
#===============valid begin=============================#yy
            iters=1
            cost=0
            is_training=False
            for x,y in iterate_chunk(vd):
                try:
                    cost+=sess.run(loss,feed_dict={X:x,Y:y})
                    iters+=1
                except ValueError,e:
                    pass;
            perp=np.sum(cost)/iters
            dstr='\033[1;32m-\033[0m'
#===============valid end=============================#yy
            if(perp<perp_last):
                saver.save(sess,save_path)
                perp_last=perp
            else:
                if(os.path.exists(save_path)):
                    saver.restore(sess,save_path)
                dstr='\033[1;31m+\033[0m'
            if ep%100==0:
                lr=lr*lr_decay
            print('%s epoch:%5d tr_perp:%3.6f perp:%3.6f lr:%f time:%3d'%(dstr,ep,tr_perp,perp,lr,time.time()-start_time))

def train2():
    global lr,lr_decay,is_training
    save_path=FLAGS.model_path+'/'+FLAGS.symbol
    perp,perp_last=0.0,float('inf')

    X=tf.placeholder(tf.float32, [batch_size,num_steps,input_size])    #每批次输入网络的tensor
    Y=tf.placeholder(tf.float32, [batch_size,num_steps,input_size]) #每批次tensor对应的标签
    w = tf.get_variable("w", [num_units, input_size],tf.float32)
    b = tf.get_variable("b", [input_size],tf.float32 )
    inputs = tf.transpose(X, [1, 0, 2])
    cell = tf.contrib.cudnn_rnn.CudnnLSTM(
        num_layers=num_layers,
        num_units=num_units,
        input_size=input_size,
        dropout=keep_prob if is_training else 0)
    params_size_t = cell.params_size()
    rnn_params = tf.get_variable(
        "lstm_params",
        initializer=tf.random_uniform([params_size_t], -0.1, 0.1),validate_shape=False)
    c = tf.zeros([num_layers, batch_size, num_units],tf.float32)
    h = tf.zeros([num_layers, batch_size, num_units], tf.float32)
    state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)
    outputs, h, c = cell(inputs, h, c, rnn_params, is_training)
    outputs = tf.transpose(outputs, [1, 0, 2])
    outputs = tf.reshape(outputs, [-1, num_units])
    outputs = tf.nn.xw_plus_b(outputs,w,b)
    outputs=tf.reshape(outputs,[batch_size,num_steps,input_size])
    outputs=tf.transpose(outputs,[1,0,2])
    outputs=tf.gather(outputs,num_steps-1)
    targets=tf.transpose(Y,[1,0,2])
    targets=tf.gather(targets,num_steps-1)
    loss=tf.losses.mean_squared_error(outputs,targets)
    train_op=tf.train.AdamOptimizer(LR).minimize(loss)
    cost,iters=0,0;

    saver=tf.train.Saver()

    with tf.Session() as sess:
        if FLAGS.debug:
            sess = tfd.LocalCLIDebugWrapperSession(sess)
        sess.run(tf.global_variables_initializer())
        for ep in range(1,num_epochs+1):
            trd,vd,td=get_dataframe(FLAGS.symbol,FLAGS.period)
#===============train begin=============================#yy
            is_training=True
            start_time=time.time()
            iters=1
            cost=0
            for x,y in iterate_chunk(trd):
                try:
                    _,cost_=sess.run([train_op,loss],feed_dict={LR:lr,X:x,Y:y})
                    iters+=1
                    cost+=cost_
                except ValueError,e:
                    print e
                    pass;
            tr_perp=np.sum(cost)/iters
#===============valid begin=============================#yy
            iters=1
            cost=0
            is_training=False
            for x,y in iterate_chunk(vd):
                try:
                    cost+=sess.run(loss,feed_dict={X:x,Y:y})
                    iters+=1
                except ValueError,e:
                    pass;
            perp=np.sum(cost)/iters
            dstr='\033[1;32m-\033[0m'
#===============valid end=============================#yy
            if(perp<perp_last):
                saver.save(sess,save_path)
                perp_last=perp
            else:
                if(os.path.exists(save_path)):
                    saver.restore(sess,save_path)
                dstr='\033[1;31m+\033[0m'
            if ep%100==0:
                lr=lr*lr_decay
            print('%s epoch:%5d tr_perp:%3.6f perp:%3.6f lr:%f time:%3d'%(dstr,ep,tr_perp,perp,lr,time.time()-start_time))

def predict():
    is_training=False
    batch_size=1
    X=tf.placeholder(tf.float32, [batch_size,num_steps,input_size])    #每批次输入网络的tensor
    Y=tf.placeholder(tf.float32, [batch_size,num_steps,input_size]) #每批次tensor对应的标签
    # with tf.variable_scope('V1', reuse=True):
    w = tf.get_variable("w", [num_units, input_size],tf.float32)
    b = tf.get_variable("b", [input_size],tf.float32 )
    inputs = tf.transpose(X, [1, 0, 2])
    cell = tf.contrib.cudnn_rnn.CudnnLSTM(
        num_layers=num_layers,
        num_units=num_units,
        input_size=input_size,
        dropout=keep_prob if is_training else 0)
    params_size_t = cell.params_size()
    rnn_params = tf.get_variable(
        "lstm_params",
        initializer=tf.random_uniform([params_size_t], -0.1, 0.1),validate_shape=False)
    c = tf.zeros([num_layers, batch_size, num_units],tf.float32)
    h = tf.zeros([num_layers, batch_size, num_units], tf.float32)
    state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)
    outputs, h, c = cell(inputs, h, c, rnn_params, is_training)
    outputs = tf.transpose(outputs, [1, 0, 2])
    outputs = tf.reshape(outputs, [-1, num_units])
    outputs = tf.nn.xw_plus_b(outputs,w,b)
    outputs=tf.reshape(outputs,[batch_size,num_steps,input_size])

    save_path=FLAGS.model_path+'/'+FLAGS.symbol
    saver=tf.train.Saver()
    trd,vd,td=get_dataframe(FLAGS.symbol,FLAGS.period)
    is_training=False
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,save_path)
        pa=[]
	op,oy=[],[]
        step=0
        for x,y in iterate_chunk(trd,True):
            if random.random()<0.5:
                continue
            for i in range(len(x)-1):
                pred=sess.run(outputs,feed_dict={X:x[i:i+1]})
                # print np.array(pred).shape
                pa.append(pred[0])
                if i>80:
                    break
                step+=1
            for i in range(len(pa)):
                for j in range(len(pa[i])):
                    if j==19:
                        op.append(pa[i][j])
                        oy.append(y[i][j])
            break
    tf.reset_default_graph()
    return (op,oy)

def listen():
    server=BaseHTTPServer.HTTPServer(('',8000), MyRequestHandler)
    print'started httpserver...'
    server.serve_forever()

class MyRequestHandler(BaseHTTPServer.BaseHTTPRequestHandler):
    def end_headers (self):
        self.send_header('Access-Control-Allow-Origin', '*')
        BaseHTTPServer.BaseHTTPRequestHandler.end_headers(self)

    def do_GET(self):
	enc="UTF-8"
        p,y=predict()
        po,pc,yo,yc=[],[],[],[]
        for i in range(len(p)):
            po.append(float(p[i][0]))
            pc.append(float(p[i][1]))
            yo.append(float(y[i][0]))
            yc.append(float(y[i][1]))
        dic={'po':po,'pc':pc,'yo':yo,'yc':yc}
        content=json.dumps(dic)
	content=content.encode(enc)
	self.send_response(200)
	self.send_header("Content-type","application/json; charset=%s"  %  enc)
	self.send_header("Content-Length",str(len(content)))
	self.end_headers()
	self.wfile.write(content)

if sys.argv[1]=='train1':train1()
if sys.argv[1]=='train2':train2()
if sys.argv[1]=='predict':predict()
if sys.argv[1]=='listen':listen()

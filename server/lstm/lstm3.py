#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os
import time
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python import debug as tfd
from tensorflow.contrib.cudnn_rnn.python.ops.cudnn_rnn_ops import CudnnLSTM
# from tensorflow.contrib.cudnn_rnn.python.layers.cudnn_rnn import CudnnLSTM
flags = tf.flags
flags.DEFINE_string('symbol', 'AUDUSD', '货币对')
flags.DEFINE_bool('debug', False, '货币对')
flags.DEFINE_bool('gpu',True, '')
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
num_layers=3
num_units= 377    #hidden layer units
num_trtimes=5
num_epochs=10000
keep_prob=0.5
batch_size=1000   #每一批次训练多少个样例
input_size=1      #输入层维度
output_size=1     #输出层维度
grad_normal=5
lr=0.25     #学习率
last_lr=lr
lr_decay=0.9      #学习率衰减系数
is_training=True
chunk_size=1000;
global_mean=1

def get_dataframe(symbol):
    dfs=[]
    files=['train','valid','test']
    for path in files:
        f=open('./data/%s/%s.txt'%(symbol,path))
        df=pd.read_csv(f,chunksize=chunk_size)
        dfs.append(df)
    return (dfs)
def iterate_chunk(df,norm=True):
    dbuffer=[]
    for da in df:
        dfa=np.array(da['<CLOSE>'])
        if norm:
            dfa=(dfa-dfa.mean())/dfa.std()
        for i in range(len(dfa)):
            if len(dbuffer)<batch_size+num_steps+1:
                dbuffer.append(dfa[i])
            else:
                d=np.array(dbuffer[:])
                # if norm:
                    # d=(d-d.mean())/d.std()
                d=d[:,np.newaxis]  #增加维度
                xa,ya=[],[]
                ni=len(d)-num_steps-1
                for j in range(ni):
                    x=d[j:j+num_steps]
                    y=d[j+num_steps:j+num_steps+1]
                    xa.append(x)
                    ya.append(y)
                ya=np.reshape(ya,[-1])
                dbuffer=dbuffer[-num_steps+1:]
                yield (xa,ya)

X=tf.placeholder(tf.float32, [None,num_steps,input_size])    #每批次输入网络的tensor
Y=tf.placeholder(tf.float32, [batch_size]) #每批次tensor对应的标签
LR= tf.placeholder(tf.float32, shape=[])
w_out=tf.Variable(tf.random_normal(shape=[num_steps*num_units,input_size],mean=global_mean,name="w_out",dtype=tf.float32))
b_out=tf.Variable(tf.random_normal(shape=[batch_size],mean=global_mean,name="b_out",dtype=tf.float32))

def lstm():
    global input_x,input_rnn,output,last
    cell=tf.contrib.rnn.LSTMBlockCell(num_units, forget_bias=-0.2,cell_clip=-0.2)
    # cell=tf.contrib.rnn.LSTMBlockFusedCell(num_units, forget_bias=-0.2,cell_clip=-0.2)
    if is_training:
         cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    layers = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)], state_is_tuple=True)
    # state = cell.zero_state(batch_size,tf.float32)
    x=tf.transpose(X,[1,0,2])
    output, state = tf.nn.dynamic_rnn(layers,inputs=x,dtype=tf.float32)
    # output = tf.transpose(output, [1, 0, 2])
    # output = tf.gather(output, int(output.get_shape()[0]) - 1)
    # output, state = tf.nn.static_rnn(layers,inputs=X,dtype=tf.float32)
    output=tf.reshape(output,[-1,num_steps*num_units])
    output=tf.matmul(output,w_out)
    output=tf.reshape(output,[-1])
    output=output+b_out
    return output

def train():
    global lr,lr_decay
    save_path='./model/'+FLAGS.symbol
    perp,perp_last=0.0,float('inf')
    pred=lstm()
    # loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    loss=tf.sqrt(tf.reduce_mean(tf.square(pred-Y)))
    # loss=tf.reduce_mean(pred)-tf.reduce_mean(Y)
    train_op=tf.train.AdamOptimizer(LR).minimize(loss)
    cost,iters=0,0;
    saver=tf.train.Saver()

    with tf.Session() as sess:
        if FLAGS.debug:
            sess = tfd.LocalCLIDebugWrapperSession(sess)
        sess.run(tf.global_variables_initializer())
        for ep in range(1,num_epochs+1):
            # print('\033[1;34m' + 'blue' + '\033[0m')
            # dstr='\033[1;34m' + u'\u2191' + '\033[0m'
            # dstr='\\032[1;34m'+'blue'+'\\032[0m' #绿色
            # print(dstr)
            trd,vd,td=get_dataframe(FLAGS.symbol)
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
                    # print x,e
            tr_perp=cost/iters
            # print('train epoch:%d perp:%f lr:%f time:%d'%(ep,perp,lr,time.time()-start_time))
            # print('iters:%d perp:%f '%(iters,perp))
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
            perp=cost/iters
            # print('\033[1;34m' + 'blue' + '\033[0m')
            dstr='\033[1;32m-\033[0m'
            # dstr='\032[1;35m'+u'\u2191'+'\032[0m' #绿色
#===============valid end=============================#yy
            if(perp<perp_last):
                saver.save(sess,save_path)
            else:
                if(os.path.exists(save_path)):
                    saver.restore(sess,save_path)
                dstr='\033[1;30m+\033[0m'
                # dstr='\031[1;35m'+u'\u2193'+'\031[0m' #绿色
            if ep%100==0:
                lr=lr*lr_decay
            # dstr='\033[1;34m-\033[0m'
            print('%s epoch:%5d tr_perp:%3.6f perp:%3.6f lr:%f time:%3d'%(dstr,ep,tr_perp,perp,lr,time.time()-start_time))
            # lr= np.abs(np.sin(perp))

def print_weights(w):
    in_w=w['in']
    out_w=w['out']
    a=np.array(in_w)
    a=np.reshape(in_w,[-1])
    b=np.array(out_w)
    b=np.reshape(out_w,[-1])
    print '==========================in====================='
    print a
    print '==========================out====================='
    print b

def predict():
    is_training=False
    pred=lstm()
    save_path='./model/'+FLAGS.symbol
    saver=tf.train.Saver()
    trd,vd,td=get_dataframe(FLAGS.symbol)
    h = np.zeros(shape=(num_layers, num_steps, num_units), dtype=np.float32)
    c = np.zeros(shape=(num_layers, num_steps, num_units), dtype=np.float32)
    is_training=False
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,save_path)
        # print sess.run(weights)
        pa=[]
        for x,y in iterate_chunk(trd,True):
            for i in range(len(x)-1):
                # xi=np.array(x[i:i+1])
                xi=x[i:i+1]
                # xi=(xi-xi.mean())/xi.std()
                nq=sess.run(pred,feed_dict={X:xi})
                pa.append(nq[-1])
                # pa.append(nq[-1]*xi.std()+xi.mean())
            for i in range(len(nq)):
                print('%f %f'%(nq[i],y[i]))
        # print y[num_steps:]

def predict2():
    is_training=False
    pred=lstm2()
    save_path='./model/'+FLAGS.symbol
    saver=tf.train.Saver()
    trd,vd,td=get_dataframe(FLAGS.symbol)
    h = np.zeros(shape=(num_layers, num_steps, num_units), dtype=np.float32)
    c = np.zeros(shape=(num_layers, num_steps, num_units), dtype=np.float32)
    is_training=False
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,save_path)
        # print sess.run(weights)
        pa=[]
        for x,y in iterate_chunk(trd,True):
            for i in range(len(x)-1):
                # xi=np.array(x[i:i+1])
                xi=x[i:i+1]
                # xi=(xi-xi.mean())/xi.std()
                nq=sess.run(pred,feed_dict={X:xi,H:h,C:c})
                pa.append(nq[-1])
                # pa.append(nq[-1]*xi.std()+xi.mean())
            for i in range(len(nq)):
                print('%f %f'%(nq[i],y[i]))
        # print y[num_steps:]
if sys.argv[1]=='train':train()
if sys.argv[1]=='predict':predict()

#def read_data(symbol):
#    data=[]
#    files=['train','valid','test']
#    for path in files:
#        f=open('./data/%s/%s.txt'%(symbol,path))
#        df=pd.read_csv(f)     #读入股票数据
#        d=np.array(df['<CLOSE>'])
#        d=d*10
#        # d=(d-np.mean(d))/np.std(d)
#        d=d[:,np.newaxis]  #增加维度
#        xa,ya=[],[]
#        for i in range(len(d)-num_steps-1):
#            x=d[i:i+num_steps]
#            y=d[i+1:i+num_steps+1]
#            xa.append(x.tolist())
#            ya.append(y.tolist())
#        f.close()
#        data.append((xa,ya))
#    return (data)

#def predict(x,nstep):
#    pred,_=lstm(1)    #预测时只输入[1,num_steps,input_size]的测试数据
#    saver=tf.train.Saver(tf.global_variables())
#    with tf.Session() as sess:
#        #参数恢复
#        #module_file = tf.train.latest_checkpoint('./')
#        module_file=FLAGS.model_path
#        saver.restore(sess, module_file)
#        #取训练集最后一行为测试样本。shape=[1,num_steps,input_size]
#        prev_seq=train_x[-1]
#        predict=[]
#        #得到之后100个预测结果
#        for i in range(nstep):
#            next_seq=sess.run(pred,feed_dict={X:[prev_seq]})
#            predict.append(next_seq[-1])
#            #每次得到最后一个时间步的预测结果，与之前的数据加在一起，形成新的测试样本
#            prev_seq=np.vstack((prev_seq[1:],next_seq[-1]))
#        #以折线图表示结果
#        # plt.figure()
#        # plt.plot(data)
#        # plt.plot(list(range(len(normalize_data))), normalize_data, color='b')
#        # plt.plot(list(range(len(normalize_data), len(normalize_data) + len(predict))), predict, color='r')
#        # plt.show()

#def train_lstm(sess,d,pred):
#    is_training=False
#    x,y=d
#    #损失函数
#    loss=tf.sqrt(tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1]))))
#    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
#    h = np.zeros(shape=(num_layers, num_steps, num_units), dtype=np.float32)
#    c = np.zeros(shape=(num_layers, num_steps, num_units), dtype=np.float32)
#    cost,iters=0,0;
#    # with tf.Session() as sess:
#        # sess.run(tf.global_variables_initializer())
#        #重复训练10000次
#    # sess.run(tf.global_variables_initializer())
#    for i in range(num_trtimes):
#        step=0
#        start=0
#        end=start+batch_size
#        while(end<len(x)):
#            _,cost_=sess.run([train_op,loss],feed_dict={X:x[start:end],Y:y[start:end],H:h,C:c})
#            start+=batch_size
#            end=start+batch_size
#            iters+=1
#            cost+=cost_
#        cm=cost/iters
#        print("iter:%d loss:%f"%(i,cm))
#    return np.exp(cm);
#
#def valid_lstm(sess,d,pred):
#    is_training=False
#    x,y=d
#    loss=tf.sqrt(tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1]))))
#    h = np.zeros(shape=(num_layers, num_steps, num_units), dtype=np.float32)
#    c = np.zeros(shape=(num_layers, num_steps, num_units), dtype=np.float32)
#    cost,iters=0,0;
#    start=0
#    end=start+batch_size
#    while(end<len(x)):
#        cost+=sess.run(loss,feed_dict={X:x[start:end],Y:y[start:end],H:h,C:c})
#        start+=batch_size
#        end=start+batch_size
#        iters+=1
#    return np.exp(cost/iters)

#def train():
#    global lr,lr_decay
#    save_path='./model/'+FLAGS.symbol
#    perp,perp_last=0.0,float('inf')
#    pred=lstm()
#    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
#    # loss=tf.clip_by_value(loss,0.1,0.5)
#    train_op=tf.train.AdamOptimizer(LR).minimize(loss)
#    # train_op=tf.train.GradientDescentOptimizer(LR).minimize(loss)
#    # tvars = tf.trainable_variables()
#    # grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),grad_normal)
#    # optimizer=tf.train.AdamOptimizer(LR)
#    # train_op = optimizer.apply_gradients(zip(grads, tvars))
#    h = np.zeros(shape=(num_layers, num_steps, num_units), dtype=np.float32)
#    c = np.zeros(shape=(num_layers, num_steps, num_units), dtype=np.float32)
#    cost,iters=0,0;
#    saver=tf.train.Saver()
#
#    trd,vd,td=get_dataframe(FLAGS.symbol)
#    with tf.Session() as sess:
#        sess.run(tf.global_variables_initializer())
#        print 'epoch start'
#        for ep in range(num_epochs):
#            # print_weights(sess.run(weights))
#            # print_weights(sess.run(biases))
#            start_time=time.time()
#            for x,y in iterate_chunk(trd):
#                iters=0
#                cost=0
#                start=0
#                end=start+batch_size
#                is_training=True
#                # tr_start_time=time.time()
#                while(end<len(x)):
#                    _,cost_=sess.run([train_op,loss],feed_dict={LR:lr,X:x[start:end],Y:x[start:end],H:h,C:c})
#                    # w=sess.run(weights)
#                    # b=sess.run(biases)
#                    # tx=np.reshape(x[start:end],[-1,input_size])
#                    # tr=np.matmul(tx,w['in'])+b['in']
#                    # print '=======================internal compution============================'
#                    # print x[start:end]
#                    # sess.run([tf.Print(input_x,[input_x]),tf.Print(input_rnn,[input_rnn]),tf.Print(output,[output])],feed_dict={LR:lr,X:x[start:end],Y:x[start:end],H:h,C:c})
#                    # ta,tb,tc=sess.run([tf.reduce_any(tf.is_nan(input_x)),tf.reduce_any(tf.is_nan(input_rnn)),tf.reduce_any(tf.is_nan(output))],feed_dict={LR:lr,X:x[start:end],Y:x[start:end],H:h,C:c})
#                    # print np.any(np.isnan(tr)),tb
#                    # print ta,tb,tc
#                    # print '=======================end internal compution============================'
#                    start+=batch_size
#                    end=start+batch_size
#                    iters+=1
#                    cost+=cost_
#                    # print_weights(sess.run(weights))
#                    # print_weights(sess.run(biases))
#                # print('epoch:%d cost:%f perp:%f lr:%f time:%d'%(ep,cost,perp,lr,time.time()-start_time))
#                perp=cost/iters
#                if np.isnan(perp):
#                    # print sess.run(weights)
#                    print 'reinit Variables'
#                    sess.run(tf.global_variables_initializer())
#                    continue
#                    # print sess.run(weights)
#            x,y=vd
#            start=0
#            end=start+batch_size
#            iters=0
#            cost=0
#            is_training=False
#            while(end<len(x)):
#                cost+=sess.run(loss,feed_dict={X:x[start:end],Y:y[start:end],H:h,C:c})
#                start+=batch_size
#                end=start+batch_size
#                iters+=1
#                # print sess.run(weights)
#            perp=cost/iters
#            if np.isnan(perp):continue
#            if(perp<perp_last):
#                perp_last=perp
#                saver.save(sess,save_path)
#            else:
#                if(os.path.exists(save_path)):
#                    saver.restore(sess,save_path)
#            if ep>0 and ep%20==0:
#                lr=lr*lr_decay
#            print('epoch:%d perp:%f lr:%f time:%d'%(ep,perp,lr,time.time()-start_time))

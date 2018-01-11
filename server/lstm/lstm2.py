#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os
import time
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.cudnn_rnn.python.ops.cudnn_rnn_ops import CudnnLSTM
# from tensorflow.contrib.cudnn_rnn.python.layers.cudnn_rnn import CudnnLSTM
flags = tf.flags
flags.DEFINE_string('symbol', 'AUDUSD', '货币对')
FLAGS=flags.FLAGS
#data=data[::-1]      #反转，使数据按照日期先后顺序排列
#以折线图展示data
#plt.figure()
#plt.plot(data)
#plt.show()
# normalize_data=(data-np.mean(data))/np.std(data)  #标准化
#———————————————————形成训练集—————————————————————
#设置常量
num_steps= 50     #时间步
num_layers=23
num_units= 11    #hidden layer units
num_trtimes=5
num_epochs=1000
keep_prob=0.5
num_inputs=100    #每一批次训练多少个样例
input_size=1      #输入层维度
output_size=1     #输出层维度
grad_normal=5
lr=0.618     #学习率
lr_decay=0.618      #学习率衰减系数
is_training=True

def get_dataframe(symbol):
    dfs=[]
    files=['train','valid','test']
    for path in files:
        f=open('./data/%s/%s.txt'%(symbol,path))
        df=pd.read_csv(f,chunksize=num_inputs+num_steps+1)
        dfs.append(df)
    return (dfs)

def iterate_chunk(df,norm=True):
    for da in df:
        d=np.array(da['<CLOSE>'])
        if norm:
            d=(d-d.mean())/d.std()
        d=d[:,np.newaxis]  #增加维度
        xa,ya=[],[]
        for i in range(len(d)-num_steps-1):
            x=d[i:i+num_steps]
            y=d[i+1:i+num_steps+1]
            xa.append(x.tolist())
            ya.append(y.tolist())
        yield (xa,ya)



X=tf.placeholder(tf.float32, [None,num_steps,input_size])    #每批次输入网络的tensor
Y=tf.placeholder(tf.float32, [None,num_steps,output_size]) #每批次tensor对应的标签
H= tf.placeholder(tf.float32, shape=[num_layers,num_steps, num_units])
C= tf.placeholder(tf.float32, shape=[num_layers,num_steps, num_units])
LR= tf.placeholder(tf.float32, shape=[])
#输入层、输出层权重、偏置
weights={
         'in':tf.Variable(tf.random_normal([input_size,num_units],seed=1)),
         'out':tf.Variable(tf.random_normal([num_units,1],seed=1))
         }
biases={
        'in':tf.Variable(tf.random_normal(shape=[num_units,],seed=1)),
        'out':tf.Variable(tf.random_normal(shape=[1,],seed=1))
        }

        # 'in':tf.Variable(tf.constant(0.1,shape=[num_units,])),
        # 'out':tf.Variable(tf.constant(0.1,shape=[1,]))
input_x,input_rnn,output=[],[],[]
def lstm():  #参数：输入网络批次数目
    global input_x,input_rnn,output
    w_in=weights['in']
    b_in=biases['in']
    input_x=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input_x,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,num_steps,num_units])  #将tensor转成3维，作为lstm cell的输入
    rnn = CudnnLSTM(num_layers, num_units, input_size, input_mode='linear_input', direction='unidirectional',
            dropout=0.5 if is_training else keep_prob, seed=0)
    params_size_t = rnn.params_size()
    params = tf.Variable(tf.random_uniform([params_size_t], minval=-0.1, maxval=0.1, dtype=tf.float32), validate_shape=False)
    output, output_h, output_c = rnn(is_training=is_training, input_data=input_rnn, input_h=H,input_c=C, params=params)
    output=tf.reshape(output,[-1,num_units]) #作为输出层的输入
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred


def train():
    global lr,lr_decay
    save_path='./model/'+FLAGS.symbol
    perp,perp_last=0.0,float('inf')
    pred=lstm()
    # loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    loss=tf.sqrt(tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1]))))
    train_op=tf.train.AdamOptimizer(LR).minimize(loss)
    h = np.zeros(shape=(num_layers, num_steps, num_units), dtype=np.float32)
    c = np.zeros(shape=(num_layers, num_steps, num_units), dtype=np.float32)
    cost,iters=0,0;
    saver=tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for ep in range(1,num_epochs+1):
            trd,vd,td=get_dataframe(FLAGS.symbol)
#===============train begin=============================#yy
            is_training=True
            start_time=time.time()
            iters=1
            cost=0
            for x,y in iterate_chunk(trd):
                _,cost_=sess.run([train_op,loss],feed_dict={LR:lr,X:x,Y:y,H:h,C:c})
                iters+=1
                cost+=cost_
            perp=cost/iters
            # print('iters:%d perp:%f '%(iters,perp))
#===============valid begin=============================#yy
            iters=1
            cost=0
            is_training=False
            for x,y in iterate_chunk(vd):
                cost+=sess.run(loss,feed_dict={X:x,Y:y,H:h,C:c})
                iters+=1
            perp=cost/iters
#===============valid end=============================#yy
            if(perp<perp_last):
                perp_last=perp
                saver.save(sess,save_path)
            else:
                if(os.path.exists(save_path)):
                    saver.restore(sess,save_path)
            if ep%20==0:
                lr=lr*lr_decay
            print('epoch:%d perp:%f lr:%f time:%d'%(ep,perp,lr,time.time()-start_time))

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
                nq=sess.run(pred,feed_dict={X:xi,H:h,C:c})
                pa.append(nq[-1])
                # pa.append(nq[-1]*xi.std()+xi.mean())
            for i in range(len(nq)):
                print('%f %f'%(nq[i],y[i][-1][0]))
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
#        end=start+num_inputs
#        while(end<len(x)):
#            _,cost_=sess.run([train_op,loss],feed_dict={X:x[start:end],Y:y[start:end],H:h,C:c})
#            start+=num_inputs
#            end=start+num_inputs
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
#    end=start+num_inputs
#    while(end<len(x)):
#        cost+=sess.run(loss,feed_dict={X:x[start:end],Y:y[start:end],H:h,C:c})
#        start+=num_inputs
#        end=start+num_inputs
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
#                end=start+num_inputs
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
#                    start+=num_inputs
#                    end=start+num_inputs
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
#            end=start+num_inputs
#            iters=0
#            cost=0
#            is_training=False
#            while(end<len(x)):
#                cost+=sess.run(loss,feed_dict={X:x[start:end],Y:y[start:end],H:h,C:c})
#                start+=num_inputs
#                end=start+num_inputs
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

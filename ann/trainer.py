import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import time
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True,precision=6)

PERIOD=6
DIM=2
LABLES=120
BATCH_SIZE=1
REGULAR_LAMBDA=0.1
POINT=100000
LR=0.001

def get_datafame(fname):
	df=pd.read_csv(fname,header=None)
	df.columns=['date','time','open','high','low','close','volume']
	df['timestamp']=df['date']+' '+df['time']
	df['timestamp']=df['timestamp'].apply(lambda x:time.mktime(time.strptime(x,'%Y.%m.%d %H:%M'))/time.mktime(time.strptime('2082-02-11 00:00','%Y-%m-%d %H:%M')))
	return df

def get_data(fname):
	df=get_datafame(fname)
	xi=[],yi=[]
	for i in range(PERIOD,len(df)):
		xi.append(df['close'][i-PERIOD:i].tolist())
		yi.append([df.iloc[i]['close']])
	return (xi,yi)

STDDEV=1/tf.sqrt(tf.cast(BATCH_SIZE,tf.float32))
def weight_variable(shape,name):
    initial = tf.truncated_normal(shape, STDDEV)
    var=tf.Variable(initial,name=name)
	tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULAR_LAMBDA)(var))
	return var
def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name=name)


def main():
	sess=tf.Session()
	xs=tf.placeholder(tf.float32,shape=[BATCH_SIZE,PERIOD],name="xs")
	ys=tf.placeholder(tf.float32,shape=[BATCH_SIZE],name="ys")

	SIZES=[6,12,1]
	w1=weight_variable([PERIOD,SIZES[0]],name="w1")
	b1=bias_variable([SIZES[0]],name="b1")

	w2=weight_variable(SIZES[0],SIZES[1]],name="w2")
	b2=bias_variable([SIZES[1]],name="b2")

	w3=weight_variable(SIZES[1],SIZES[2]],name="w2")
	b3=bias_variable([SIZES[2]],name="b2")

	y=tf.xw_plus_b(xs,w1,b1)
	y=tf.xw_plus_b(y,w2,b2)
	y=tf.nn.elu(y)
	y=tf.xw_plus_b(y,w3,b3)

	mse=tf.reduce_sum(tf.square(y-ys))
	tf.add_to_collection("losses",mse) 
	loss = tf.add_n(tf.get_collection('losses'))
	train_op = tf.train.AdamOptimizer(LR).minimize(loss) # 调

	tf.summary.scalar("ac",0.5)
	merged = tf.summary.merge_all()

	train_writer = tf.summary.FileWriter('./train',sess.graph)
	test_writer = tf.summary.FileWriter('./test')
	sess.run(tf.global_variables_initializer())
	for epoch in range(0,1000):
		train_data=get_data('1.csv');
		for i in xrange(0,len(train_data),BATCH_SIZE):
			x=train_data[0][i:i+BATCH_SIZE]
			y=train_data[1][i:i+BATCH_SIZE]
		_,_loss=sess.run([train_op,loss],feed_dict={xs:x,ys:y})
		summary=sess.run([merged],feed_dict={xs:xi,ys:yi})
		print("loss:%.6f"%(loss))


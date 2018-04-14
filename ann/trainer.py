import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import time
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True,precision=6)
tf.reset_default_graph()

PERIOD=33
DIM=2
LABLES=120
EPOCH=50000
BATCH_SIZE=80
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
    xi=[];yi=[]
    for i in range(PERIOD,len(df)):
        xi.append(df['close'][i-PERIOD:i].tolist())
        yi.append([df.iloc[i]['close']])
    xi=np.array(xi)
    xi=(xi-xi.mean())/(xi.max()-xi.min())
    yi=np.array(yi)
    yi=(yi-yi.mean())/(yi.max()-yi.min())
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


sess=tf.Session()
xs=tf.placeholder(tf.float32,shape=[BATCH_SIZE,PERIOD],name="xs")
ys=tf.placeholder(tf.float32,shape=[BATCH_SIZE,1],name="ys")

SIZES=[50,100,50,21,1]
w1=weight_variable([PERIOD,SIZES[0]],name="w1")
b1=bias_variable([SIZES[0]],name="b1")

w2=weight_variable([SIZES[0],SIZES[1]],name="w2")
b2=bias_variable([SIZES[1]],name="b2")

w3=weight_variable([SIZES[1],SIZES[2]],name="w3")
b3=bias_variable([SIZES[2]],name="b3")

w4=weight_variable([SIZES[2],SIZES[3]],name="w4")
b4=bias_variable([SIZES[3]],name="b4")

w5=weight_variable([SIZES[3],SIZES[4]],name="w5")
b5=bias_variable([SIZES[4]],name="b5")

y=tf.nn.xw_plus_b(xs,w1,b1)
y=tf.nn.xw_plus_b(y,w2,b2)
y=tf.nn.elu(y)
y=tf.nn.xw_plus_b(y,w3,b3)
y=tf.nn.elu(y)
y=tf.nn.xw_plus_b(y,w4,b4)
y=tf.nn.elu(y)
y=tf.nn.xw_plus_b(y,w5,b5)

mse=tf.losses.mean_squared_error(y,ys)
tf.add_to_collection("losses",mse)
regular_loss = tf.add_n(tf.get_collection('losses'))
loss=mse
train_op = tf.train.AdamOptimizer(LR).minimize(loss) # 调

#tf.summary.scalar("ac",0.5)
#merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('./train',sess.graph)
test_writer = tf.summary.FileWriter('./test')
train_data=get_data('1.csv');
valid_data=get_data('2.csv')
print('data is ready')
sess.run(tf.global_variables_initializer())
_last_vloss=np.infty;_loss=0;_vloss=0
for epoch in range(0,EPOCH):
    print(epoch)
    for i in range(0,len(train_data),BATCH_SIZE):
        _x=train_data[0][i:i+BATCH_SIZE]
        _y=train_data[1][i:i+BATCH_SIZE]
        _,_loss=sess.run([train_op,loss],feed_dict={xs:_x,ys:_y})
        _x=valid_data[0][0:BATCH_SIZE]
        _y=valid_data[1][0:BATCH_SIZE]
        _vloss=sess.run(loss,feed_dict={xs:_x,ys:_y})
    #if _vloss<_last_vloss:_last_vloss=_vloss
    #else:break
print("loss:%.6f vloss:%.6f"%(_loss,_vloss))
print('finished')

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
#价格除数，用于归一化，防止出现大于1
MASK=2
POINT=100000
LR=0.001

df=pd.read_csv('1.csv',header=None)
df.columns=['date','time','open','high','low','close','volume']
df['timestamp']=df['date']+' '+df['time']
df['timestamp']=df['timestamp'].apply(lambda x:time.mktime(time.strptime(x,'%Y.%m.%d %H:%M'))/time.mktime(time.strptime('2082-02-11 00:00','%Y-%m-%d %H:%M')))
# df['high']=df['high']-df['high'].mean()
# df['high']=df['high']/df['high'].std()
# df['low']=df['low']-df['low'].mean()
# df['low']=df['low']/df['low'].std()
# df['open']=df['open']-df['open'].mean()
# df['open']=df['open']/df['open'].std()
# df['close']=df['close']-df['close'].mean()
# df['close']=df['close']/df['close'].std()

xi=[]
yi=[]
# print(len(df))
# for i in range(PERIOD,len(df)):
for i in range(PERIOD,PERIOD+10):
    xi.append(df['close'][i-PERIOD:i].tolist())
    yi.append([df.iloc[i]['close']])
    # break
print(xi)
print(yi)
exit()
# for i in range(0,len(yi)):
    # print(np.argmax(yi[i]))
STDDEV=1/tf.sqrt(tf.cast(BATCH_SIZE,tf.float32))
def weight_variable(shape,name):
    # 正态分布，标准差为0.1，默认最大为1，最小为-1，均值为0
    initial = tf.truncated_normal(shape, STDDEV)
    return tf.Variable(initial,name=name)
def bias_variable(shape,name):
    # 创建一个结构为shape矩阵也可以说是数组shape声明其行列，初始化所有值为0.1
    # initial = tf.constant(0.1, shape=shape)
    initial = tf.truncated_normal(shape,1)
    return tf.Variable(initial,name=name)

sess=tf.Session()
xs=tf.placeholder(tf.float32,shape=[BATCH_SIZE,PERIOD],name="xs")
ys=tf.placeholder(tf.float32,shape=[BATCH_SIZE],name="ys")
# yo=tf.placeholder(tf.float32,shape=[None,LABLES])
# for i in range(0,LABLES):
xt=tf.reshape(xs,[-1,PERIOD*DIM])
#for i in range(0,2):
## for i in range(0,10):
#    w=weight_variable([PERIOD*DIM,LABLES],name="w1")
#    b=bias_variable([LABLES],name="b1")
#
#    mat=tf.nn.xw_plus_b(xt,w,b)
#    # resh=tf.reshape(mat,[-1,LABLES])
#    x=tf.nn.elu(mat,name="elu")
#    # drop=tf.reshape(mat,[-1,LABLES,1])
#    w2=weight_variable([LABLES,LABLES],name="w2")
#    b2=bias_variable([LABLES],name="b2")
#    # y=tf.matmul(drop,w2)+b2
#    y=tf.nn.xw_plus_b(x,w2,b2)
#    y=tf.nn.elu(y)
#
#    w3=weight_variable([LABLES,LABLES],name="w3")
#    b3=bias_variable([LABLES],name="b3")
#
#    z=tf.nn.xw_plus_b(y,w3,b3)
#    z=tf.nn.elu(z)
#    # y=tf.nn.dropout(y,0.5,name="dropout")
#    # y=tf.multiply(drop,w2)+b2
#    if yt==None:yt=z
#    else: yt=yt+z
#    # tf.Print(yt,[yt])
#    # yo+=y
#    # yo=yo+tf.matmul(drop,w2)+b2

w=weight_variable([PERIOD*DIM,LABLES],name="w1")
b=bias_variable([LABLES],name="b1")

mat=tf.nn.xw_plus_b(xt,w,b)
# resh=tf.reshape(mat,[-1,LABLES])
x=tf.nn.elu(mat,name="elu")
# drop=tf.reshape(mat,[-1,LABLES,1])
y=x
for i in range(0,10):
    w2=weight_variable([LABLES,LABLES],name="w2")
    b2=bias_variable([LABLES],name="b2")
# y=tf.matmul(drop,w2)+b2
    y=tf.nn.xw_plus_b(y,w2,b2)
    y=tf.nn.elu(y)

yt=y
softmax=tf.nn.softmax(yt)
softmax=tf.clip_by_value(softmax,0.0001,0.99)
# a=tf.reduce_min(tf.log(softmax))
# b=tf.reduce_min(yt)
# c=tf.reduce_min(softmax)
cross_entropy = -tf.reduce_sum(ys * tf.log(softmax)) # 定义交叉熵为loss函数
train_op = tf.train.AdamOptimizer(LR).minimize(cross_entropy) # 调

correct_prediction = tf.equal(tf.argmax(softmax), tf.argmax(ys))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("ac",0.5)
merged = tf.summary.merge_all()

# yh=np.zeros(shape=[len(yi),LABLES])
# print(yh.shape)
# print(ys.shape)
# print(np.array(yi).shape)
# 写到指定的磁盘路径中
train_writer = tf.summary.FileWriter('./train',sess.graph)
test_writer = tf.summary.FileWriter('./test')
sess.run(tf.global_variables_initializer())
for i in range(0,1000):
    _,ce,cp,ac=sess.run([train_op,cross_entropy,correct_prediction,accuracy],feed_dict={xs:xi,ys:yi})
    _ys,_yt=sess.run([ys,softmax],feed_dict={xs:xi,ys:yi})
    # print(_ys)
    # print(_yt)
    # _,ce,cp,ac,sm,_a,_b,_c,_yt=sess.run([train_op,cross_entropy,correct_prediction,accuracy,softmax,a,b,c,yt],feed_dict={xs:xi,ys:yi})
    summary=sess.run([merged],feed_dict={xs:xi,ys:yi})
    # run_metadata = tf.RunMetadata()
    # train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
    # print(summary)
    # train_writer.add_summary(summary, i)
    # sess.run(tf.Print(yt,[yt]),feed_dict={xs:xi,ys:yi})
    # print('sm:%d a:%f b:%f c:%f'%(len(sm[sm<=0]),_a,_b,_c))
    # print(_yt.shape)
    # print(cp)
    # break
    print("ce:%.6f ac:%.6f"%(ce,ac))

    # yo,so,ce=sess.run([ys,softmax,cross_entropy],feed_dict={xs:xi,ys:yi})
    # print(yo)
    # print(so)
    # en=yo*np.log(so)
    # print(np.sum(en))
    # print(ce)

train_writer.close()
test_writer.close()
# tf.global_variables_initializer().run()
# for i in range(20000):
	# batch = mnist.train.next_batch(50)
	# if i%100 == 0:
			# train_accuracy = accuracy.eval(feed_dict={x:batch[0], ys: batch[1], keep_prob: 1.0})
			# print("step %d, training accuracy %g"%(i, train_accuracy))
	# train_step.run(feed_dict={x: batch[0], ys: batch[1], keep_prob: 0.5})
# print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, ys: mnist.test.labels, keep_prob: 1.0}))

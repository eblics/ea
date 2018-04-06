import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import time
pd.set_option('display.float_format', lambda x: '%.5f' % x)

PERIOD=6
DIM=2
LABLES=120
#价格除数，用于归一化，防止出现大于1
MASK=2
POINT=100000
LR=0.01

df=pd.read_csv('1.csv',header=None)
df.columns=['date','time','open','high','low','close','volume']
df['timestamp']=df['date']+' '+df['time']
df['timestamp']=df['timestamp'].apply(lambda x:time.mktime(time.strptime(x,'%Y.%m.%d %H:%M'))/time.mktime(time.strptime('2082-02-11 00:00','%Y-%m-%d %H:%M')))
df['high']=df['high']/MASK
df['low']=df['low']/MASK
df['open']=df['open']/MASK
df['close']=df['close']/MASK

xi=[]
yi=[]
# print(len(df))
# for i in range(PERIOD,len(df)):
for i in range(PERIOD,PERIOD+200):
    item=[]
    item.append(df['open'][i-PERIOD:i].tolist())
    item.append(df['close'][i-PERIOD:i].tolist())
    # item.append(df['timestamp'][i-PERIOD:i].tolist())
    gap=df.iloc[i]['close']-df.iloc[i]['open']
    gap=gap*POINT
    if gap>LABLES/2:gap=LABLES/2-1
    if gap<-LABLES/2:gap=-LABLES/2
    il=int(gap+LABLES/2)
    lable=np.zeros([LABLES]).tolist()
    lable[il]=1
    item=np.array(item)
    item=item.transpose(1,0)
    xi.append(item)
    yi.append(lable)
    # break
# print(xi)
# for i in range(0,len(yi)):
    # print(np.argmax(yi[i]))

def weight_variable(shape,name,trainable=True):
    # 正态分布，标准差为0.1，默认最大为1，最小为-1，均值为0
    # initial = tf.truncated_normal(shape, stddev=0.1)
    initial = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(initial,name=name,trainable=trainable)
def bias_variable(shape,name,trainable=True):
    # 创建一个结构为shape矩阵也可以说是数组shape声明其行列，初始化所有值为0.1
    # initial = tf.constant(0.1, shape=shape)
    # initial = tf.truncated_normal(shape, stddev=0.1)
    initial = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(initial,name=name,trainable=trainable)

sess=tf.Session()
xs=tf.placeholder(tf.float32,shape=[None,PERIOD,DIM],name="xs")
ys=tf.placeholder(tf.float32,shape=[None,LABLES],name="ys")
# yo=tf.placeholder(tf.float32,shape=[None,LABLES])
yt=None
# for i in range(0,LABLES):
xt=tf.reshape(xs,[-1,PERIOD*DIM])
#设置过滤器，筛选一部分数据用于训练；不满足过滤器的视为不可识别
wf1=weight_variable([PERIOD*DIM,13],name="wf",trainable=False)
bf1=bias_variable([13],name="bf",trainable=False)
wf2=weight_variable([13,1],name="wf",trainable=False)
bf2=bias_variable([1],name="bf",trainable=False)
# xf=tf.nn.xw_plus_b(xt,wf1,bf1)
xf=tf.matmul(xt,wf1)
xf=tf.nn.sigmoid(xf)
xf=tf.matmul(xf,wf2)
pxf=tf.Print(xf,[xf])
# xf=tf.nn.xw_plus_b(xf,wf2,bf2)
# xmask=tf.zeros_like(xf)
# xf=tf.greater(xf,xmask)
# xf=tf.cast(xf,tf.float32)
ys=tf.multiply(ys,xf)

w=weight_variable([PERIOD*DIM,LABLES],name="w1")
b=bias_variable([LABLES],name="b1")
xt=tf.nn.xw_plus_b(xt,w,b)
x=tf.nn.elu(xt,name="elu")
y=x
for i in range(0,2):
    w2=weight_variable([LABLES,LABLES],name="w2")
    b2=bias_variable([LABLES],name="b2")
    y=tf.nn.xw_plus_b(y,w2,b2)
    y=tf.nn.elu(y)

yt=y
yt=tf.nn.elu(yt)
yt=tf.multiply(yt,xf)
yt=tf.clip_by_value(yt,0.0001,0.99)
cross_entropy = -tf.reduce_sum(ys * tf.log(yt)) # 定义交叉熵为loss函数
train_op = tf.train.AdamOptimizer(LR).minimize(cross_entropy) # 调


correct_prediction = tf.equal(tf.argmax(yt), tf.argmax(ys))
cp_sum = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
ys_zero=tf.reduce_sum(tf.cast(tf.equal(ys,tf.zeros_like(ys)),tf.float32))
cp_sum=cp_sum-ys_zero
accuracy=cp_sum/(tf.cast(tf.size(correct_prediction),tf.float32)-ys_zero)
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
    _xf,_wf1,_bf=sess.run([xf,wf1,bf1],feed_dict={xs:xi,ys:yi})
    print(_xf)
    # print(_wf1)
    # print(_bf)
    sess.run(pxf,feed_dict={xs:xi,ys:yi})
    break
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
    # print("ce:%.6f ac:%.6f yz:%.6f"%(ce,ac,_yz[0]))

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

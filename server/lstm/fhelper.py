import sys
from datetime import datetime
from datetime import timedelta
import pandas as pd
import tensorflow as tf
import numpy as np


FLAGS=tf.flags.FLAGS
tf.flags.DEFINE_string('rawpath','','')
tf.flags.DEFINE_string('outpath','','')
tf.flags.DEFINE_string('starttime','','')
tf.flags.DEFINE_string('endtime','','')
g_chunk_size=1000

def raw_to_date_serices():
    reader=pd.read_csv(FLAGS.rawpath,dtype={'<DTYYYYMMDD>':str,'<TIME>':str},chunksize=g_chunk_size)
    header=True
    for chunk in reader:
        # df=df.assign(time_x=lambda x:str(df['<DTYYYYMMDD>'])+''+str(df['<TIME>']))
        # df.assign(time_x=lambda x:df['<DTYYYYMMDD>']+df['<TIME>']).head()
        chunk['TIME_X']=chunk['<DTYYYYMMDD>']+chunk['<TIME>']
        del chunk['<DTYYYYMMDD>']
        del chunk['<TIME>']
        chunk=chunk.rename(columns={'TIME_X':'TIME','<OPEN>':'OPEN','<CLOSE>':'CLOSE','<HIGH>':'HIGH','<LOW>':'LOW'})
        chunk=chunk[['TIME','OPEN','CLOSE','HIGH','LOW']]
        chunk.to_csv(FLAGS.outpath,header=header,index=False,sep='\t',mode='a')
        header=False

def to_period():
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m%d%H%M%S')
    reader=pd.read_csv(FLAGS.rawpath,sep='\t',index_col='TIME',date_parser=dateparse,chunksize=g_chunk_size)
    starttime=pd.datetime.strptime(FLAGS.starttime,'%Y%m%d')
    endtime=pd.datetime.strptime(FLAGS.endtime,'%Y%m%d')
    chunk_num=0
    buf=None
    for chunk in reader:
        try:
            chunk_num+=1
            print chunk_num
            min_date=chunk.index.min()
            max_date=chunk.index.max()
            if max_date<starttime:
                continue
            buf=chunk if buf is None else pd.concat([buf,chunk])
            rng=pd.date_range(min_date,max_date,freq='1H')
            data=[]
            for i in range(len(rng)-1):
                if (rng[i]-max_date)<timedelta(hours=1):
                    break
                s=buf[rng[i]:rng[i+1]]
                data.append([s['CLOSE'].max(),s['OPEN'].min()])
            buf=buf[rng[-1]:]
            # print chunk
            # print buf
            # break
        except ValueError,e:
            print chunk_num
        # for i in range(len(rng)):
            # td=chunk[rgn[i]:rng[1]]
        # ts=pd.Series(chunk,rng)
        # print ts


if sys.argv[1]=='reformat':
    raw_to_date_serices()
if sys.argv[1]=='period':
    to_period()



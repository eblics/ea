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
g_chunk_size=24*60

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
    header=True
    for chunk in reader:
        try:
            min_date=chunk.index.min()
            max_date=chunk.index.max()
            if max_date<starttime or min_date>endtime:
                continue
            buf=chunk if buf is None else pd.concat([buf,chunk])
            rng=pd.date_range(min_date,max_date,freq='1H')
            data=[]
            for i in range(len(rng)-1):
                if (max_date-rng[i])<timedelta(hours=1) or rng[i]>endtime:
                    break
                s=buf[rng[i]:rng[i+1]]
                if len(s)==0:
                    continue
                data.append([rng[i].strftime('%Y%m%d%H'),s['OPEN'][0],s['CLOSE'][-1]])
            buf=buf[rng[-1]:]
            outdf=pd.DataFrame(data,columns=['TIME','OPEN','CLOSE'])
            outdf.to_csv(FLAGS.outpath,header=header,index=False,sep='\t',mode='a')
            header=False
            if max_date>endtime:
                break
        except ValueError,e:
            print chunk_num
        # for i in range(len(rng)):
            # td=chunk[rgn[i]:rng[1]]
        # ts=pd.Series(chunk,rng)
        # print ts

def h1_to_volat():
    reader=pd.read_csv(FLAGS.rawpath,sep='\t',dtype={'TIME':str},chunksize=g_chunk_size)
    header=True
    start=0
    for chunk in reader:
        # print chunk
        # break
        data=[]
        for index,row in chunk.iterrows():
<<<<<<< HEAD
            data.append([row['CLOSE']-row['OPEN'],row['HIGH']-row['LOW']])
        outdf=pd.DataFrame(data,columns=['CO','HL'])
=======
            data.append([row['TIME'],row['CLOSE']-row['OPEN']])
        outdf=pd.DataFrame(data,columns=['TIME','VOL'])
>>>>>>> 883936747d99c6bfaf557db60b993dead6d4894e
        outdf.to_csv(FLAGS.outpath,header=header,index=False,sep='\t',mode='a')
        header=False

def h1_to_volat2():
    print FLAGS.rawpath
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m%d,%H%M')
    df=pd.read_csv(FLAGS.rawpath,sep=',',header=None,date_parser=dateparse)
    data=[]
    for index,row in df.iterrows():
        data.append([row[5]-row[2],row[2],row[5]])
    outdf=pd.DataFrame(data)
    outdf.to_csv(FLAGS.outpath,header=None,index=False)

if sys.argv[1]=='reformat':
    raw_to_date_serices()
if sys.argv[1]=='period':
    to_period()
if sys.argv[1]=='volat':
    h1_to_volat()
if sys.argv[1]=='volat2':
    h1_to_volat2()




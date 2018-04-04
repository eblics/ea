import pandas as pd
import numpy as np
import sys
import time

pd.set_option('display.float_format', lambda x: '%.5f' % x)
PERIOD=21
GAP=100
POINT=0.00001
GAP=GAP*POINT

def gen_target_file(fname):
    df=pd.read_csv(fname,header=None,index_col=None)
    df.columns=['date','time','open','high','low','close','vol']
    df['time']=df['date']+' '+df['time']
    df['time']=df['time'].apply(lambda x:time.strptime(x,'%Y.%m.%d %H:%M'))
    df['timestamp']=df['time'].apply(lambda x:time.mktime(x))
    shape=df.shape
    tgarr=[]
    for i in range(PERIOD,shape[0]-PERIOD):
        prerow=df.iloc[i]
        price=prerow['open']
        for j in range(i+1,shape[0]):
            row=df.iloc[j]
            pretime=time.strftime('%Y-%m-%d %H:%M',prerow['time'])
            now=time.strftime('%Y-%m-%d %H:%M',row['time'])
            if row['high']-price-GAP>=0:
                tgarr.append([j,1,pretime,now])
                break
            elif row['low']-price+GAP<=0:
                tgarr.append([j,-1,pretime,now])
                break
        if len(tgarr)<i-PERIOD+1:
            tgarr.append([j,0,pretime,now])
    tgdf=pd.DataFrame(tgarr,columns=['j','target','prevtime','now'])
    tgdf.to_csv(fname+'.tg')

if sys.argv[1]=='gen':
    gen_target_file(sys.argv[2])

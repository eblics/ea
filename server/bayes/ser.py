#-*- coding: utf-8 -*-
from socket import *
from time import ctime
from struct import *
import redis

K=9;
HOST='0.0.0.0'
PORT=8888
BUFSIZ=1024
ADDR=(HOST, PORT)
sock=socket(AF_INET, SOCK_STREAM)

sock.bind(ADDR)
sock.listen(5)
rc=redis.Redis(host='localhost', port=6379, db=0);

def literal(h,l):
    h=float(h);
    l=float(l);
    if h>l:
        d='2';
    if h==l:
        d='1';
    if h<l:
        d='0';
    return d;

def getAdvArray(s):
    sa=s.split(',');
    pa=[];
    for x in sa:
        pa.append(float(x[0:7]))
    h,l=[],[]
    i=0
    while i<len(pa)-3:
        h.append(literal(pa[i+2],pa[i]))
        l.append(literal(pa[i+3],pa[i+1]))
        i+=2

    return h,l

def advice(h,l):
    kh='H'+''.join(h)
    kl='L'+''.join(l)
    print(kh)
    print(kl)
    vh,vl,pha,pla=[],[],[],[];
    for x in ['0','1','2']:
        v=rc.get(kh+x);
        if not v:v=0;
        vh.append(float(v))
        v=rc.get(kl+x);
        if not v:v=0;
        vl.append(float(v))
    th,tl=0,0;
    for x in vh:
        s=sum(vh);
        v=0 if s==0 else x/s
        pha.append(v)
    for x in vl:
        s=sum(vl)
        v=0 if s==0 else x/s;
        pla.append(v)

    ph=max(pha);dh=pha.index(ph);
    pl=max(pla);dl=pla.index(pl);
    return (dh,ph,dl,pl)


def main():
    while True:
        print('waiting for connection')
        tcpClientSock, addr=sock.accept()
        print('connect from ', addr)
        try:
            data=tcpClientSock.recv(BUFSIZ)
            if len(data)<198:continue;
            h,l=getAdvArray(data)
            adv=advice(h,l)
            s='{0},{1},{2},{3}'.format(adv[0],adv[1],adv[2],adv[3])
            print(s)
        except KeyboardInterrupt:
            tcpClientSock.close()
        except:
            tcpClientSock.close()
            raise;
            break
        if not data:
            break
        tcpClientSock.send(s)
    tcpClientSock.close()

main()
sock.close()




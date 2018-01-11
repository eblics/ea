#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import os
import redis

K=10;
KT='KT'
H,LH,L,LL=[],[],[],[];

rc=redis.Redis(host='localhost', port=6379, db=0);
f=open('../../data/'+sys.argv[1]+'.txt')

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


print f.next()
i=0;
while True:
    try:
        r=f.next();
    except StopIteration:
        break;
    if not r:
        break;
    ls=r.split(',');
    h,l=ls[4],ls[5];
    #去掉最后一位小数，太小的变化没有意义
    #h=float(h[0:len(h)-1]);
    #l=float(l[0:len(l)-1]);
    if len(H)<K and len(L)<K:
        H.append(h);
        L.append(l);
        if len(H)>1  and len(L)>1:
            LH.append(literal(H[-2],H[-1]));
            LL.append(literal(L[-2],L[-1]));

    if len(H)==K and len(L)==K:
        kh='H'+''.join(LH);
        kl='L'+''.join(LL);
        fh=rc.get(kh);
        fl=rc.get(kl);
        ft=rc.get(KT);
        if not fh:fh=0;
        if not fl:fl=0;
        if not ft:ft=0;
        fh=float(fh)+1;
        fl=float(fl)+1;
        ft=float(ft)+1;
        rc.set(kh,fh);
        rc.set(kl,fl);
        ft=rc.set('KT',ft+1);
        H.pop(0);L.pop(0);LH.pop(0);LL.pop(0);



//+------------------------------------------------------------------+
//|                                                        bayes.mq4 |
//|                                                           eblics |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "eblics"
#property link      ""
#property version   "1.00"
#property strict
#property show_inputs
#include <winsock.mqh>

input string HOST="118.31.116.176";
input ushort PORT=8888;
input double PROB=0.5;
input double Lots;
input double TAKEPROFIT=50;
input double STOPLOSS=100;
input double SLIPPAGE=3;
const int K=9;
const int BUFSIZE=64;


int sock=INVALID_SOCKET;
int k=9;
int OnInit()
{
    return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{

    Print("停止服务器");
}

//价格列表，发往咨询服务器
double p[];
//前一个价格柱
int bar;
//前一个最低值和最高值
double low,high;
//建议的方向和概率
int dh,dl=0;
double ph,pl=0;

int ticket=-1;
void OnTick()
{
    int ret=0;
    if(Bars<9)
        return;
    if(bar<Bars){
        bar=Bars;
        low=Low[1];
        high=High[1];
        ArrayResize(p,K*2);
        for(int i=1;i<K+1;i++){
            p[2*K-2*i]=High[i];
            p[2*k-2*i+1]=Low[i];
        }
        string data[];ArrayResize(data,K*2);
        arr_double2str(data,p);
        string str=arr_join(data,',');
        uchar buf[];StringToCharArray(str,buf);
        //Print(str);
        //Print("bufsize "+ArraySize(buf));
        sock=sock_connect(PORT,HOST);
        if(sock==INVALID_SOCKET||sock==SOCKET_ERROR){
            Print("Invalid socket or error");
            return;
        }
        send(sock,buf,ArraySize(buf),0);
        if (ret == SOCKET_ERROR) {
            Print(" send() failed: error "+WSAGetLastError());
        }
        ArrayInitialize(buf,0);
        ret=recv(sock,buf,BUFSIZE,0);
        //Print(CharArrayToString(buf));
        str=CharArrayToString(buf);
        string adv[];StringSplit(str,',',adv);
        if(ArraySize(adv)<4)
            return;
        dh=StringToInteger(adv[0]);dl=StringToInteger(adv[2]);
        ph=StringToDouble(adv[1]);pl=StringToDouble(adv[3]);

        if (ret == SOCKET_ERROR) {
            Print("recv failed: error "+WSAGetLastError());
        }
        closesocket(sock);
        WSACleanup();
    }
    plc_max_prob1(ph,pl,dh,dl);
    //RefreshRates();
    //Print(CharArrayToString(buf));
    //Print(sock_receive(sock));
}

//最大概率法，双涨双跌才入仓
void plc_max_prob1(double ph,double pl,int dh,int dl){
    if(ph<PROB && pl<PROB)
        return;

    if(OrdersTotal()==0){
        int dir;
        double price;
        double stoploss;
        double takeprofit;
        //如果呈下行趋势，当且仅当最低价也下降，且超过点差；同时价格不能高于上一次，如果已经高于上一次，说明预测已经失效
        //PrintFormat("ask:%f bid:%f high:%f,low:%f dh:%d",Ask,Bid,high,low,dh);
        if(ph>=PROB&&dh==0&&(Bid-low)>10*Point)
        {
            dir=OP_SELL;
            price=Bid;
            stoploss=high+100*Point;
            takeprofit=price-TAKEPROFIT*Point;
            PrintFormat("sell %f %f %f",price,stoploss,takeprofit);
            ticket=OrderSend(Symbol(),dir,Lots,price,SLIPPAGE,stoploss,takeprofit,
                    StringFormat("自动下单 %d,%f,%d,%f",dh,ph,dl,pl),16384,0,Green);
            return;
        }

        if(pl>=PROB&&dh==2&&(high-Ask)>10*Point)
        {
            dir=OP_BUY;
            price=Ask;
            stoploss=low-100*Point;
            takeprofit=price+TAKEPROFIT*Point;
            PrintFormat("buy %f %f %f",price,stoploss,takeprofit);
            ticket=OrderSend(Symbol(),dir,Lots,price,SLIPPAGE,stoploss,takeprofit,
                    StringFormat("自动下单 %d,%f,%d,%f",dh,ph,dl,pl),16384,0,Green);
            return;
        }
    }
    else{
        bool ret=OrderSelect(ticket,SELECT_BY_TICKET);
        if(!ret){
            Print("选择订单失败 "+GetLastError());
            return;
        }
        if(OrderSymbol()!=Symbol()){
            Print("货币对不匹配");
            return;
        }
        int orderType=OrderType();
        double price=OrderOpenPrice();
        double stoploss=OrderStopLoss();
        double takeprofit=OrderTakeProfit();
        double profit=OrderProfit();

        if(orderType==OP_BUY){
            if((Bid-price)>100*Point){
                OrderModify(ticket,price,Bid-80*Point,Bid+100*Point,0,Green);
            }
            if(ph>=PROB&&dh==0){
                if(profit<=0)
                    OrderClose(ticket,OrderLots(),Bid,SLIPPAGE,Red);
                else
                    OrderClose(ticket,OrderLots(),Bid,SLIPPAGE,Green);
                ticket=-1;
                return;
            }
        }
        if(orderType==OP_SELL){
            if((price-Ask)>100*Point){
                OrderModify(ticket,price,Ask-80*Point,Ask-100*Point,0,Green);
            }
            if(pl>=PROB&&dl==2){
                if(profit<=0)
                    OrderClose(ticket,OrderLots(),Ask,SLIPPAGE,Red);
                else
                    OrderClose(ticket,OrderLots(),Ask,SLIPPAGE,Green);
                ticket=-1;
                return;
            }
        }

    }
}

double OnTester()
{
    double ret=0.0;

    return(ret);
}

int arr_float2char(char& dst[],double& src[]){
    int n;
    for(int i=0;i<ArraySize(src);i++){
        n=n+4;
        dst[i]=(char)src[i];
        dst[i+1]=(char)src[i]<<8;
        dst[i+1]=(char)src[i]<<16;
        dst[i+3]=(char)src[i]<<24;
        PrintFormat("%f %a %x %x %x %x %x",src[i],src[i],src[i],dst[i],dst[i+1],dst[i+2],dst[i+3]);
        PrintFormat("%x %x %x %x",(int)src[i],(int)src[i]<<8,(int)src[i]<<16,(int)src[i]<<24);
    }
    return n;
}

int arr_double2str(string& dst[],double& src[]){
    for(int i=0;i<ArraySize(src);i++){
        dst[i]=DoubleToString(src[i]);
    }
    return 0;
}

string arr_join(string& src[],char sep){
    string str="";
    for(int i=0;i<ArraySize(src);i++){
        str+=src[i];
        if(i<ArraySize(src)-1){
            str+=",";
        }
    }
    return str;
}

int sock_connect(int port,string ip_address){
    int sock,retval;
    sockaddr_in addrin;
    ref_sockaddr ref;
    char wsaData[]; // byte array of the future structure
    ArrayResize(wsaData, sizeof(WSAData)); // resize it to the size of the structure

    //int2struct(remote,sin_family,AF_INET);
    //int2struct(remote,sin_addr,inet_addr(ip_address));
    //int2struct(remote,sin_port,htons(port));
    char ch[];
    StringToCharArray(ip_address,ch);
    addrin.sin_family=AF_INET;
    addrin.sin_addr=inet_addr(ch);
    addrin.sin_port=htons(port);
    retval = WSAStartup(0x202, wsaData);
    if (retval != 0) {
        Print("Server: WSAStartup() failed with error "+ retval);
        return(-1);
    }
    sock=socket(AF_INET, SOCK_STREAM,0);
    if (sock == INVALID_SOCKET){
        Print("Server: socket() failed with error "+WSAGetLastError());
        WSACleanup();
        return(-1);
    }

    //ref.ref=addrin;
    retval=connect(sock,addrin,16);
    if (retval != 0) {
        Print("connect failed with error "+ +WSAGetLastError());
        WSACleanup();
        return(-1);
    }
    return sock;
}


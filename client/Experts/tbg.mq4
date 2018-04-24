//+------------------------------------------------------------------+
//|                                                        bayes.mq4 |
//|                                                           eblics |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "eblics"
#property link      ""
#property version   "1.00"
#property strict


#include <winsock.mqh>
#define MAGICMA  19820211
//#define MAGICMA  20131111
input string HOST="47.96.248.23";
input ushort PORT=8899;
input double Lots=0.1;
input double MAXLOTS=2;
//input double TESTLOTS=0.01;
input double TAKEPROFIT=40;
input double STOPLOSS=2000;
input double SLIPPAGE=3;
input int PERIOD=1440;
input double FACTOR=1.2;
input int GAP=200;
input double RATE=0.8;

//input int TOTAL_LIMIT=10;


const int BUFSIZE=16;
double open,low,high,close;
int ticket=-1;
int bars=0,op_bars=0;
int sock=INVALID_SOCKET;
double lots=Lots;

struct order
{
    int op;
    double profit;
};

order myorders[1024];

int CalculateCurrentOrders(int op)
{
    int buys=0,sells=0;
    for(int i=0;i<OrdersTotal();i++)
     {  
      if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES)==false) break;
      //Print("i:",i," buys:",buys," sells:",sells," symbol:",OrderSymbol()==Symbol()," magic:",OrderMagicNumber()," type:",OrderType());
      //Print("magic:",OrderMagicNumber());
      if(OrderSymbol()==Symbol() && OrderMagicNumber()==MAGICMA)
        {
         
         if(OrderType()==OP_BUY)  buys++;
         if(OrderType()==OP_SELL) sells++;
        }
     }
     if(op==0)return buys;
     if(op==1)return sells;
     if(op==2)return buys+sells;
     if(op==3)return buys-sells;
     return 0;
}
int CalculateOrdersTotal()
{
    int total=0;
    for(int i=0;i<OrdersHistoryTotal();i++)
    {
        if(OrderSelect(i,SELECT_BY_POS,MODE_HISTORY)==false) break;
        if(OrderSymbol()==Symbol() && OrderMagicNumber()==MAGICMA)
        {
            total+=1;
        }
    }
    return total;
}

double CalculateSuccRate()
{
    double total=0,profit=0;
    for(int i=0;i<OrdersHistoryTotal();i++)
    {
        if(OrderSelect(i,SELECT_BY_POS,MODE_HISTORY)==false) break;
        if(OrderSymbol()==Symbol() && OrderMagicNumber()==MAGICMA)
        {
            //PrintFormat("history ticket:%d",OrderTicket());
            total+=1;
            if(OrderProfit()>0) profit+=1;
        }
    }
    //PrintFormat("total:%d profit:%d %d",total,profit,OrdersHistoryTotal());
    if(total==0) return 0.5;
    return profit/total;
}

double GetSpread(){
    return MarketInfo(Symbol(), MODE_SPREAD);
}

int SelectMaxOrder(int op){
     int ticket=-1;
     double openprice=0,price=0;
     //price=(op==OP_SELL)?0:100000;
     for(int i=0;i<OrdersTotal();i++)
     {
      if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES)==false) break;
      if(OrderSymbol()==Symbol() && OrderMagicNumber()==MAGICMA && OrderType()==op)
        {
         if(op==OP_BUY){
            openprice=OrderOpenPrice();
            if(price==0){
                price=openprice;
                ticket=OrderTicket();
            }
            if(openprice<price){
                price=openprice;
                ticket=OrderTicket();
            }
         }
         if(op==OP_SELL){
            openprice=OrderOpenPrice();
            if(price==0){
                price=openprice;
                ticket=OrderTicket();
            }
            if(openprice>price){
                price=openprice;
                ticket=OrderTicket();
            }
         }
        }
     }
     if(ticket!=-1){
        if(!OrderSelect(ticket,SELECT_BY_TICKET)){
            Print("Order Select Error");
        }
     }
     return ticket;
 }
  
int OpenOrder(int op,double lots,string notes)
{
    int res=-1;
    if(op==OP_SELL)
    {
        //res=OrderSend(Symbol(),OP_SELL,Lots,Bid,SLIPPAGE,0,0,"",MAGICMA,0,Red);
        res=OrderSend(Symbol(),OP_SELL,lots,Bid,SLIPPAGE,Ask+STOPLOSS*Point,Bid-TAKEPROFIT*Point,"",MAGICMA,0,Red);
        if(res==-1){
            Print(GetLastError());
        }
    }
    if(op==OP_BUY)
    {
        //res=OrderSend(Symbol(),OP_BUY,Lots,Ask,SLIPPAGE,0,0,"",MAGICMA,0,Blue);
        res=OrderSend(Symbol(),OP_BUY,lots,Ask,SLIPPAGE,Bid-STOPLOSS*Point,Ask+TAKEPROFIT*Point,"",MAGICMA,0,Blue);
        if(res==-1){
            Print(GetLastError());
        }
    }
    return res;
}

void CloseAllOrders()
{
    int res=0;
    for(int i=0;i<OrdersTotal();i++)
    {
        if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES)==false) break;
        if(OrderType()==OP_BUY && OrderMagicNumber()==MAGICMA)
        {
            if(OrderClose(OrderTicket(),OrderLots(),Bid,SLIPPAGE,Red)==false){
                Print(GetLastError());
            }
        }
        else if(OrderType()==OP_SELL && OrderMagicNumber()==MAGICMA){
            if(OrderClose(OrderTicket(),OrderLots(),Ask,SLIPPAGE,Red)==false){
                Print(GetLastError());
            }
        }
    }
    Print("close all orders");
}

int OnInit()
{
    //int sock=Sock_Connect(PORT,HOST);
    //string str=StringFormat("REIN,%s,%s,%f,%f,%f,%f"," "," ",0,0,0,0);
    //SendMessage(str,"IN);  
    ArraySetAsSeries(Open,false);
    ArraySetAsSeries(Low,false);
    ArraySetAsSeries(High,false);
    ArraySetAsSeries(Close,false);
    ArraySetAsSeries(Time,false);    
    //Print("init: ",Bars);
    double open,low,high,close,time;
    string history=NULL;
    bars=(Bars<PERIOD)?0:(Bars-PERIOD);
    for(;bars<Bars-1;bars++)
    {
        low=Low[bars];high=High[bars];open=Open[bars];close=Close[bars];time=Time[bars];
        string date=TimeToStr(time,TIME_DATE);
        string minute=TimeToStr(time,TIME_MINUTES);
        string str=StringFormat("%s,%s,%f,%f,%f,%f;",date,minute,open,high,low,close);
        //Print(str);
        history+=str;    
    }
    Print(history);
    SendMessage(history,"INIT");
    //Sock_Close(sock);
    ArraySetAsSeries(Open,true);
    ArraySetAsSeries(Low,true);
    ArraySetAsSeries(High,true);
    ArraySetAsSeries(Close,true);
    ArraySetAsSeries(Time,true);
    Print("init finished:",Bars);
    // sock=Sock_Connect(PORT,HOST);
    return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
    Sock_Close(sock);
    Print("停止服务器");
}

void OnTick()
{
    int ret=0;
    //if(Bars<PERIOD) return;
    //Print("Tick Bars:",Bars);
    //Print("tick: ",Bars);
    if(bars<Bars){
        bars=Bars;
        low=Low[1];high=High[1];open=Open[1];close=Close[1];
        string date=TimeToStr(Time[1],TIME_DATE);
        string minute=TimeToStr(Time[1],TIME_MINUTES );
        string str=StringFormat("%s,%s,%f,%f,%f,%f",date,minute,open,high,low,close);   
        int reply=StringToInteger(SendMessage(str,"TICK"));
        //Print("bars",Bars," reply:",reply);
        CheckOrders(reply);
        //Print(Open[1],Close[1],Ask,Bid,Open[0],Close[0]);
    }
}
double equity=0;
void CheckOrders(int op){
    double price=0;
    //int total=CalculateOrdersTotal();
    int ticket;
    if(op==0)return;
    if(GetSpread()>SLIPPAGE)return;
    if(AccountProfit()>=Lots*TAKEPROFIT){
        CloseAllOrders();
    }
    if(op==1){
        ticket=SelectMaxOrder(OP_BUY);
        price=OrderOpenPrice();
        //PrintFormat("ask:%f price:%f ticket:%d",Ask,price,ticket);
        if(ticket!=-1&&price-Ask<GAP*Point)
            return;
        lots=AccountEquity()>equity?lots*FACTOR:Lots;
        lots=MathMin(lots,MAXLOTS);
        ticket=OpenOrder(OP_BUY,lots,"buy");
        
    }
    if(op==-1){
        ticket=SelectMaxOrder(OP_SELL);
        price=OrderOpenPrice();
        //PrintFormat("bid:%f price:%f ticket:%d",Bid,price,ticket);
        if(ticket!=-1&&Bid-price<GAP*Point)
            return;
        lots=AccountEquity()>equity?lots*FACTOR:Lots;
        lots=MathMin(lots,MAXLOTS);
        ticket=OpenOrder(OP_SELL,lots,"sell");
    }
    equity=AccountEquity();
}



string SendMessage(string msg)
{
    int sock=Sock_Connect(PORT,HOST);
    string reply=SendMessage(sock,msg);
    Sock_Close(sock);
    return reply;
}

string SendMessage(int sock,string msg)
{
    uchar buf[];
    int ret;
    StringToCharArray(msg,buf);
    
    if(sock==INVALID_SOCKET||sock==SOCKET_ERROR){
        Print("Invalid socket or error");
        return NULL;
    }
    Print("size:",ArraySize(buf));
    send(sock,buf,ArraySize(buf),0);
    if (ret == SOCKET_ERROR) {
        Print(" send() failed: error "+WSAGetLastError());
        return NULL;
    }
    ArrayInitialize(buf,0);
    recv(sock,buf,BUFSIZE,0);
    string str=CharArrayToString(buf);
    return str;
}

string SendMessage(string msg,string cmd)
{
    uchar buf[];
    uchar cmdbuf[];
    int ret;
    
    int sock=Sock_Connect(PORT,HOST);
    if(sock==INVALID_SOCKET||sock==SOCKET_ERROR){
        Print("Invalid socket or error");
        return NULL;
    }
    StringToCharArray(msg,buf);
    int size=ArraySize(buf);
    string cmdstr=StringFormat("%s,%6d",cmd,size);
    StringToCharArray(cmdstr,cmdbuf);
    send(sock,cmdbuf,ArraySize(cmdbuf),0);
    //Print("cmd:",cmd," size:",ArraySize(cmdbuf));
    send(sock,buf,size,0);
    if (ret == SOCKET_ERROR) {
        Print(" send() failed: error "+WSAGetLastError());
        return NULL;
    }
    ArrayInitialize(buf,0);
    recv(sock,buf,BUFSIZE,0);
    Sock_Close(sock);
    string str=CharArrayToString(buf);
    return str;
}

void SendMessageNoReply(string msg){
    int sock=Sock_Connect(PORT,HOST);
    SendMessageNoReply(sock,msg);
    Sock_Close(sock);
}

int SendMessageNoReply(int sock,string msg){
    uchar buf[];
    int ret;
    StringToCharArray(msg,buf);
    if(sock==INVALID_SOCKET||sock==SOCKET_ERROR){
        Print("Invalid socket or error");
        return NULL;
    }
    send(sock,buf,ArraySize(buf),0);
    if (ret == SOCKET_ERROR) {
        Print(" send() failed: error "+WSAGetLastError());
        return NULL;
    }
    return NULL;
}

void Sock_Close(int sock){
    closesocket(sock);
    WSACleanup();
}

int Sock_Connect(int port,string ip_address){
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



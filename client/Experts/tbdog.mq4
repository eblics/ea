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
#define MAGICMA  19820211
//#define MAGICMA  20131111
input string HOST="47.96.248.23";
input ushort PORT=8899;
input double Lots=0.2;
input double TAKEPROFIT=200;
input double STOPLOSS=40;
input double SLIPPAGE=3;
input int PERIOD=55;
input int TOTAL_LIMIT=10;


const int BUFSIZE=16;
double open,low,high,close;
int ticket=-1;
int bars=0,op_bars=0;
int sock=INVALID_SOCKET;


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
            total+=1;
            if(OrderProfit()>0) profit+=1;
        }
    }
    return profit/total;
}

double GetSpread(){
    return MarketInfo(Symbol(), MODE_SPREAD);
}
void OpenOrder(int op,string notes)
{
    int res=0;
    if(op==OP_SELL)
    {
        //res=OrderSend(Symbol(),OP_SELL,Lots,Bid,SLIPPAGE,0,Bid-TAKEPROFIT*Point,"",MAGICMA,0,Red);
        res=OrderSend(Symbol(),OP_SELL,Lots,Bid,SLIPPAGE,Ask+STOPLOSS*Point,Bid-TAKEPROFIT*Point,"",MAGICMA,0,Red);
        if(res!=0){
            Print(GetLastError());
        }
        return;
    }
    if(op==OP_BUY)
    {
        //res=OrderSend(Symbol(),OP_BUY,Lots,Ask,SLIPPAGE,0,Ask+TAKEPROFIT*Point,"",MAGICMA,0,Blue);
        res=OrderSend(Symbol(),OP_BUY,Lots,Ask,SLIPPAGE,Bid-STOPLOSS*Point,Ask+TAKEPROFIT*Point,"",MAGICMA,0,Blue);
        if(res!=0){
            Print(GetLastError());
        }
        return;
    }
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
    op_bars=0;
}

int OnInit()
{
    ArraySetAsSeries(Open,false);
    ArraySetAsSeries(Low,false);
    ArraySetAsSeries(High,false);
    ArraySetAsSeries(Close,false);
    ArraySetAsSeries(Time,false);
    double open,low,high,close,time;
    for(bars=0;bars<Bars;bars++)
    {
        //low=Low[bars];high=High[bars];open=Open[bars];close=Close[bars];time=Time[bars];
        //string date=TimeToStr(time,TIME_DATE);
        //string minute=TimeToStr(time,TIME_MINUTES);
        //string str=StringFormat("INIT,%s,%s,%f,%f,%f,%f",date,minute,open,high,low,close);
        //SendMessage(str);    
    }
    ArraySetAsSeries(Open,true);
    ArraySetAsSeries(Low,true);
    ArraySetAsSeries(High,true);
    ArraySetAsSeries(Close,true);
    ArraySetAsSeries(Time,true);
    Print("init",Bars);
    // sock=Sock_Connect(PORT,HOST);
    return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
    Print("停止服务器");
}

void OnTick()
{
    int ret=0;
    if(Bars<PERIOD) return;
    //Print("Tick Bars:",Bars);
    if(bars<Bars){
        bars=Bars;
        low=Low[1];high=High[1];open=Open[1];close=Close[1];
        string date=TimeToStr(Time[1],TIME_DATE);
        string minute=TimeToStr(Time[1],TIME_MINUTES );
        string str=StringFormat("TICK,%s,%s,%f,%f,%f,%f",date,minute,open,high,low,close);   
        int reply=StringToInteger(SendMessage(str));
        //Print("bars",Bars," reply:",reply);
        CheckOrders(reply);
        //Print(Open[1],Close[1],Ask,Bid,Open[0],Close[0]);
    }
}
//double lots=Lots;
void CheckOrders(int op){
    int total=CalculateOrdersTotal();
    if(op==0)return;
    if(GetSpread()>SLIPPAGE)return;
    if(AccountProfit()>=Lots*TAKEPROFIT){
        CloseAllOrders();
    }
    total=CalculateCurrentOrders(2);
    //if(total>=TOTAL_LIMIT)
    //    return;
    //Print("total:",total," op:",op);
    //if(total!=0&&Bars-op_bars<PERIOD)
    //    return;
    //total=CalculateCurrentOrders(3);
    //if(Bars-op_bars<PERIOD)
    //    return;
    if(op==1){
        OpenOrder(OP_BUY,"buy");
        op_bars=Bars;
    }
    if(op==-1){
        OpenOrder(OP_SELL,"sell");
        op_bars=Bars;
    }
    //if(op==1&&total>0&&total<TOTAL_LIMIT/2&&Bars-op_bars>PERIOD){
    //    OpenOrder(OP_BUY,"buy");
    //    op_bars=Bars;
    //}
    //if(op==1&&total<=0){
    //    OpenOrder(OP_BUY,"buy");
    //    op_bars=Bars;
    //}
    //if(op==-1&&total>=0){
    //    OpenOrder(OP_SELL,"sell");
    //    op_bars=Bars;
    //}
    //if(op==-1&&total<0&&-total<TOTAL_LIMIT/2&&Bars-op_bars>PERIOD){
    //    OpenOrder(OP_SELL,"sell");
    //    op_bars=Bars;
    //}
    
    //if(op==1) OpenOrder(OP_BUY,"buy");
    //if(op==-1) OpenOrder(OP_SELL,"sell");
}
string SendMessage(string msg)
{
    uchar buf[];
    int ret;
    StringToCharArray(msg,buf);
    
    sock=Sock_Connect(PORT,HOST);
    if(sock==INVALID_SOCKET||sock==SOCKET_ERROR){
        Print("Invalid socket or error");
        return NULL;
    }
    send(sock,buf,ArraySize(buf),0);
    if (ret == SOCKET_ERROR) {
        Print(" send() failed: error "+WSAGetLastError());
        return NULL;
    }
    ArrayInitialize(buf,0);
    recv(sock,buf,BUFSIZE,0);
    string str=CharArrayToString(buf);
    closesocket(sock);
    WSACleanup();
    return str;
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


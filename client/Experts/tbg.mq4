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

input double MAXLOTS=2;
//input double TESTLOTS=0.01;
input double TAKEPROFIT=100;
input double STOPLOSS=8000;
input double SLIPPAGE=3;
input int PERIOD=144;
input int Interval=30;
input int LIMIT=11;
input double FACTOR=1.5;
input int GAP=300;
input double RATE=0.8;
input double DECAY_RATE=1;
input double HEDGE_FACTOR=0.9;
input int SPACE=10;
input string STARTTIME="2:00";
input string ENDTIME="13:00";

//input int TOTAL_LIMIT=10;

double Lots=0.1;
const int BUFSIZE=16;
double open,low,high,close;
int sock=INVALID_SOCKET;
int ticket=-1;
int bars=0,op_bars=0;
int starthour,startminute,endhour,endminute;
//double lots=Lots;
//input int LOSS_LIMIT=3000;
//double steps=STOPLOSS/GAP;
//double Lots_CONST=GAP*((1-MathPow(FACTOR,steps))/MathPow(1-FACTOR,2)-steps*MathPow(FACTOR,steps)/(1-FACTOR));

double CalculateCurrentOrdersLots(int op)
{
    double buys=0,sells=0;
    for(int i=0;i<OrdersTotal();i++)
     {  
      if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES)==false) break;
      //Print("i:",i," buys:",buys," sells:",sells," symbol:",OrderSymbol()==Symbol()," magic:",OrderMagicNumber()," type:",OrderType());
      //Print("magic:",OrderMagicNumber());
      if(OrderSymbol()==Symbol() && OrderMagicNumber()==MAGICMA)
        {
         
         if(OrderType()==OP_BUY)  buys+=OrderLots();
         if(OrderType()==OP_SELL) sells+=OrderLots();
        }
     }
     if(op==0)return buys;
     if(op==1)return sells;
     if(op==2)return buys+sells;
     if(op==3)return buys-sells;
     return 0;
}

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
         
         if(OrderType()==OP_BUY)  buys+=1;
         if(OrderType()==OP_SELL) sells+=1;
        }
     }
     if(op==0)return buys;
     if(op==1)return sells;
     if(op==2)return buys+sells;
     if(op==3)return buys-sells;
     return 0;
}

double CalculateCurrentProfit(int op)
{
    double buys=0,sells=0;
    for(int i=0;i<OrdersTotal();i++)
     {  
      if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES)==false) break;
      //Print("i:",i," buys:",buys," sells:",sells," symbol:",OrderSymbol()==Symbol()," magic:",OrderMagicNumber()," type:",OrderType());
      //Print("magic:",OrderMagicNumber());
      if(OrderSymbol()==Symbol() && OrderMagicNumber()==MAGICMA)
        {
         
         if(OrderType()==OP_BUY)  buys+=OrderProfit();
         if(OrderType()==OP_SELL) sells+=OrderProfit();
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
    for(int i=0;i<OrdersTotal();i++)
    {
        if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES)==false) break;
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
  
int OpenOrder(int op,double lots,double stoploss,double takeprofit,string notes)
{
    int res=-1;
    if(op==OP_SELL)
    {
        //res=OrderSend(Symbol(),OP_SELL,lots,Bid,SLIPPAGE,0,0,"",MAGICMA,0,Red);
        //res=OrderSend(Symbol(),OP_SELL,lots,Bid,SLIPPAGE,Ask+STOPLOSS*Point,Bid-TAKEPROFIT*Point,"",MAGICMA,0,Red);
        res=OrderSend(Symbol(),OP_SELL,lots,Bid,SLIPPAGE,stoploss,takeprofit,"",MAGICMA,0,Red);
        if(res==-1){
            Print(GetLastError());
        }
    }
    if(op==OP_BUY)
    {
        //res=OrderSend(Symbol(),OP_BUY,lots,Ask,SLIPPAGE,0,0,"",MAGICMA,0,Blue);
        //res=OrderSend(Symbol(),OP_BUY,lots,Ask,SLIPPAGE,Bid-STOPLOSS*Point,Ask+TAKEPROFIT*Point,"",MAGICMA,0,Blue);
        res=OrderSend(Symbol(),OP_BUY,lots,Ask,SLIPPAGE,stoploss,takeprofit,"",MAGICMA,0,Blue);
        if(res==-1){
            Print(GetLastError());
        }
    }
    return res;
}

void CloseAllOrders(int op)
{
    int res=0;
    //Print("close orderstotal:",OrdersTotal());
    int total=OrdersTotal();
    int tickets[];
    int index=0;
    ArrayResize(tickets,total);
    ArrayInitialize(tickets,-1);
    for(int i=0;i<total;i++)
    {
        if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES)==false){
            Print("i:",i," ",GetLastError());
        }
        if(OrderMagicNumber()!=MAGICMA)continue;
        tickets[i]=OrderTicket();
        
    }
    for(int i=0;i<total;i++){
        if(tickets[i]==-1)continue;
        if(OrderSelect(tickets[i],SELECT_BY_TICKET,MODE_TRADES)==false){
            Print("i:",i," ",GetLastError());
        }
        if(OrderType()==OP_BUY && OrderMagicNumber()==MAGICMA)
        {
            if(op==0||op==2){
                if(OrderClose(OrderTicket(),OrderLots(),Bid,SLIPPAGE,Red)==false){
                    Print(GetLastError());
                }
            }
        }
        else if(OrderType()==OP_SELL && OrderMagicNumber()==MAGICMA){
            if(op==1||op==2){
                if(OrderClose(OrderTicket(),OrderLots(),Ask,SLIPPAGE,Red)==false){
                    Print(GetLastError());
                }
            }
        }
    }
    ArrayFree(tickets);
    Print("close all orders");
}

int OnInit()
{
    sock=Sock_Connect(PORT,HOST);
    if(sock==INVALID_SOCKET)
    {
        Print("Sock init error:");
        return INIT_FAILED;
    }
    //PrintFormat("steps:%f 1;%f 2:%f",steps,(1-MathPow(FACTOR,steps))/MathPow(1-FACTOR,2),steps*MathPow(FACTOR,steps)/(1-FACTOR));
    //int sock=Sock_Connect(PORT,HOST);
    //string str=StringFormat("REIN,%s,%s,%f,%f,%f,%f"," "," ",0,0,0,0);
    //SendMessage(str,"IN);  
    ArraySetAsSeries(Open,false);
    ArraySetAsSeries(Low,false);
    ArraySetAsSeries(High,false);
    ArraySetAsSeries(Close,false);
    ArraySetAsSeries(Time,false);    
    double open,low,high,close,time;
    string history=NULL;
    bars=(Bars<PERIOD)?0:(Bars-PERIOD);
    for(;bars<Bars-1;bars++)
    {
        low=Low[bars];high=High[bars];open=Open[bars];close=Close[bars];time=Time[bars];
        string date=TimeToStr(time,TIME_DATE);
        string minute=TimeToStr(time,TIME_MINUTES);
        string str=StringFormat("%s,%s,%f,%f,%f,%f;",date,minute,open,high,low,close);
        history+=str;    
    }
    SendMessage(sock,history,"INIT");
    ArraySetAsSeries(Open,true);
    ArraySetAsSeries(Low,true);
    ArraySetAsSeries(High,true);
    ArraySetAsSeries(Close,true);
    ArraySetAsSeries(Time,true);
    Print("init finished:",Bars);
    // sock=Sock_Connect(PORT,HOST);
    //string result[];               
    //ushort u_sep;
    //int k;
    //u_sep=StringGetCharacter(":",0);
    //k=StringSplit(STARTTIME,u_sep,result);
    //starthour=StringToInteger(result[0]);
    //startminute=StringToInteger(result[1]);
    //k=StringSplit(ENDTIME,u_sep,result);
    //endhour=StringToInteger(result[0]);
    //endminute=StringToInteger(result[1]);
    return(INIT_SUCCEEDED);
}

void Split(string str,string sep,string& result[],int &i){
    //string result[];               // An array to get strings
   //--- Get the separator code
    ushort u_sep=StringGetCharacter(sep,0);
    //--- Split the string to substrings
    int k=StringSplit(str,u_sep,result);
    //return result;
}

void OnDeinit(const int reason)
{
    SendMessage(sock,"","CLSE");
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
        //if(bars%(20*PERIOD)==0)CloseAllOrders(2);
        low=Low[0];high=High[0];open=Open[0];close=Close[0];
        string date=TimeToStr(Time[0],TIME_DATE);
        string minute=TimeToStr(Time[0],TIME_MINUTES );
        string str=StringFormat("%s,%s,%f,%f,%f,%f",date,minute,open,high,low,close);   
        int reply=StringToInteger(SendMessage(sock,str,"TICK"));
        CheckOrders4(reply);
        //Print("bars",Bars," reply:",reply);
        //CheckOrders(reply);
        //Print(Open[1],Close[1],Ask,Bid,Open[0],Close[0]);
    }
}
double equity=0;
int opbars=0;
void CheckOrders(int op){
    if(AccountProfit()>=Lots*TAKEPROFIT){
        CloseAllOrders(2);
    }
    double price=0,takeprofit,stoploss;
    double buys=CalculateCurrentOrders(0);
    double sells=CalculateCurrentOrders(1);
    double total=buys+sells;
    double cur=buys-sells;
    //double total=CalculateCurrentOrders(2);
    if(op==0)return;
    if(GetSpread()>SLIPPAGE)return; 
    if(Bars-opbars<Interval||total>=LIMIT*Lots)return;
    int ticket;
    //double takeprofit,stoploss;
    if(op==1){
        stoploss=Bid-STOPLOSS*Point;
        takeprofit=Ask+TAKEPROFIT*Point;
        if(cur>0){
            ticket=SelectMaxOrder(OP_BUY);
            price=OrderOpenPrice();
            if(ticket!=-1&&price-Ask<GAP*Point)
                return;
            OpenOrder(OP_BUY,Lots,stoploss,0,"buy");
        }
        else
            OpenOrder(OP_BUY,Lots,stoploss,takeprofit,"buy");
    }
    if(op==-1){
        stoploss=Ask+STOPLOSS*Point;
        takeprofit=Bid-TAKEPROFIT*Point;
        if(cur<0){
            ticket=SelectMaxOrder(OP_SELL);
            price=OrderOpenPrice();
            if(ticket!=-1&&Bid-price<GAP*Point)
                return;
            OpenOrder(OP_SELL,Lots,stoploss,0,"sell");
        }
        else{
            OpenOrder(OP_SELL,Lots,stoploss,takeprofit,"sell");
        }
    }
    opbars=Bars;
}

void CheckOrders3(int op){
    double stoploss,takeprofit,total,profit;
    
    if(GetSpread()>SLIPPAGE)return; 
    int ticket,type;
    bool seleted;
    for(int i=0;i<OrdersTotal();i++){
        seleted=OrderSelect(i,SELECT_BY_POS,MODE_TRADES);
        //PrintFormat("i:%d selected:%d",i,seleted);
    }
    if(seleted){
        //Print("selected");
        ticket=OrderTicket();
        type=OrderType();
        profit=OrderProfit();
       PrintFormat("ticket:%d type:%d profit:%f",ticket,type,profit);
        if(type==OP_BUY){
            if(profit>0){
                stoploss=OrderOpenPrice();
                takeprofit=Ask+TAKEPROFIT*Point;
                if(NormalizeDouble(OrderStopLoss(),Digits)==stoploss)
                    return;     
                if(OrderModify(OrderTicket(),OrderOpenPrice(),stoploss,OrderTakeProfit(),0,Blue)==false){
                    Print("modify order error ticket:",ticket,GetLastError());
                    return;
                }
                
            }
        }
        if(type==OP_SELL){
            if(profit>0){
                stoploss=OrderOpenPrice();
                takeprofit=Bid-TAKEPROFIT*Point;
                if(NormalizeDouble(OrderStopLoss(),Digits)==stoploss)
                    return;           
                if(OrderModify(OrderTicket(),OrderOpenPrice(),stoploss,OrderTakeProfit(),0,Blue)==false){
                    Print("modify order error ticket:",ticket,GetLastError());
                    return;
                }   
            }
        }
    }
    
    total=CalculateCurrentOrders(2);
    if(total>0)return;
    if(op==0)return;
    //if(Bars-opbars<Interval||total>=LIMIT*Lots)return;
    //int ticket;
    //double takeprofit,stoploss;
    if(op==1){
        stoploss=Bid-STOPLOSS*Point;
        takeprofit=Ask+TAKEPROFIT*Point;
        OpenOrder(OP_BUY,Lots,stoploss,takeprofit,"buy");
    }
    if(op==-1){
        stoploss=Ask+STOPLOSS*Point;
        takeprofit=Bid-TAKEPROFIT*Point;
        OpenOrder(OP_SELL,Lots,stoploss,takeprofit,"sell");
    }
    opbars=Bars;
}

void CheckOrders9(int op){
    double lots,stoploss,takeprofit,total,profit;   
    
    int ticket,type,step;
    bool seleted;
    for(int i=0;i<OrdersTotal();i++){
        seleted=OrderSelect(i,SELECT_BY_POS,MODE_TRADES);
        //PrintFormat("i:%d selected:%d",i,seleted);
    }
    if(seleted){
        //Print("selected");
        ticket=OrderTicket();
        type=OrderType();
        profit=OrderProfit();
       PrintFormat("ticket:%d type:%d profit:%f",ticket,type,profit);
        if(type==OP_BUY){
            if(profit>0){
                stoploss=Bid-STOPLOSS*Point;     
                if(OrderStopLoss()>=stoploss)return;
                if(NormalizeDouble(OrderStopLoss(),Digits)==stoploss)
                    return;     
                if(OrderModify(OrderTicket(),OrderOpenPrice(),stoploss,OrderTakeProfit(),0,Blue)==false){
                    Print("modify order error ticket:",ticket,GetLastError());
                    return;
                }
                
            }
        }
        if(type==OP_SELL){
            if(profit>0){
                stoploss=Ask+STOPLOSS*Point;
                if(OrderStopLoss()<=stoploss)return;
                if(NormalizeDouble(OrderStopLoss(),Digits)==stoploss)
                    return;           
                if(OrderModify(OrderTicket(),OrderOpenPrice(),stoploss,OrderTakeProfit(),0,Blue)==false){
                    Print("modify order error ticket:",ticket,GetLastError());
                    return;
                }   
            }
            
        }
    } 
    total=CalculateCurrentOrders(2);
    if(total>0)return;
    if(op==0)return;
    if(GetSpread()>SLIPPAGE)return; 
    //if(Bars-opbars<Interval||total>=LIMIT*Lots)return;
    //int ticket;
    //double takeprofit,stoploss;
    if(op==1){
        stoploss=Bid-STOPLOSS*Point;
        takeprofit=Ask+TAKEPROFIT*Point;
        OpenOrder(OP_BUY,Lots,stoploss,takeprofit,"buy");
    }
    if(op==-1){
        stoploss=Ask+STOPLOSS*Point;
        takeprofit=Bid-TAKEPROFIT*Point;
        OpenOrder(OP_SELL,Lots,stoploss,takeprofit,"sell");
    }
    opbars=Bars;
}
void CheckOrders5(int op){
    double stoploss,takeprofit,total,profit,lots;
    Lots=AccountBalance()/100000;
    //PrintFormat("const:%f lots:%f",Lots_CONST,Lots);
    if(AccountProfit()>=Lots*TAKEPROFIT){
        CloseAllOrders(2);
    }
    if(CalculateCurrentProfit(0)>Lots*TAKEPROFIT){
        CloseAllOrders(0);
    }
    if(CalculateCurrentProfit(1)>Lots*TAKEPROFIT){
        CloseAllOrders(1);
    }
    int ticket=-1,type;
    if(op==0)return;
    if(GetSpread()>SLIPPAGE)return; 
    double buys=CalculateCurrentOrders(0);
    double sells=CalculateCurrentOrders(1);
    double cur=buys-sells;
    
    if(op==1){
        if(buys>MAXLOTS)
            return;
        ticket=SelectMaxOrder(OP_BUY);
        if(ticket!=-1){
            if(OrderOpenPrice()-Bid<GAP*Point){
               if(cur!=0)
                  return;
               takeprofit=Ask+TAKEPROFIT*Point;
               stoploss=Bid-STOPLOSS*Point;
               lots=Lots;
            }
            else{
               takeprofit=0;
               stoploss=Bid-STOPLOSS*Point;
               lots=OrderLots()*FACTOR;
            }
        }
        else{
            takeprofit=Ask+TAKEPROFIT*Point;
            stoploss=Bid-STOPLOSS*Point;
            lots=Lots;
        }
        ticket=OpenOrder(OP_BUY,lots,stoploss,takeprofit,"buy");
    }
    if(op==-1){
        if(sells>MAXLOTS)return;
        ticket=SelectMaxOrder(OP_SELL);
        if(ticket!=-1){
            if(Ask-OrderOpenPrice()<GAP*Point){
               if(cur!=0)
                  return;
               stoploss=Ask+STOPLOSS*Point;
               takeprofit=Bid-TAKEPROFIT*Point;
               lots=Lots;
            }
            else{
               stoploss=Ask+STOPLOSS*Point;
               takeprofit=0;
               lots=OrderLots()*FACTOR;
            }              
        }
        else{
            stoploss=Ask+STOPLOSS*Point;
            takeprofit=Bid-TAKEPROFIT*Point;
            lots=Lots;
        }
        ticket=OpenOrder(OP_SELL,lots,stoploss,0,"sell");
    }
}

void CheckOrders7(int op){
    double stoploss,takeprofit,total,profit,lots;
    Lots=AccountEquity()*DECAY_RATE/100000;
    //PrintFormat("equity:%f losts:%f",AccountBalance(),Lots);
    //PrintFormat("const:%f lots:%f",Lots_CONST,Lots);
    if(AccountProfit()>=Lots*TAKEPROFIT){
        CloseAllOrders(2);
    }
    if(CalculateCurrentProfit(0)>Lots*TAKEPROFIT){
        CloseAllOrders(0);
    }
    if(CalculateCurrentProfit(1)>Lots*TAKEPROFIT){
        CloseAllOrders(1);
    }
    int ticket=-1,type;
    if(op==0)return;
    if(GetSpread()>SLIPPAGE)return; 
    
    double buys_lots=CalculateCurrentOrdersLots(0);
    double sells_lots=CalculateCurrentOrdersLots(1);
    double cur_lots=buys_lots-sells_lots;
    int buys=CalculateCurrentOrders(0);
    int sells=CalculateCurrentOrders(1);
    double gap=0;
    //MathSrand(GetTickCount());
    //double random=((double)rand())/32767;
    PrintFormat("buys_lots:%f sells_lots:%f buys:%d sells:%d",buys_lots,sells_lots,buys,sells);
    if(op==1){
        if(buys_lots>MAXLOTS)return;
        ticket=SelectMaxOrder(OP_BUY);
        if(ticket!=-1){
            gap=GAP*MathPow(buys,1/3);
            //Print("buys gap:",gap);
            //gap=GAP;
            //PrintFormat("ticket:%d ordertakeprofit:%f",ticket,OrderTakeProfit());
            if(OrderOpenPrice()-Bid<gap*Point){
               if(cur_lots!=0)
                  return;
               //PrintFormat("ticket:%d ordertakeprofit:%f buys:%d",ticket,OrderTakeProfit(),buys);
               takeprofit=Ask+TAKEPROFIT*Point;
               stoploss=Bid-STOPLOSS*Point;
               lots=Lots;
            }
            else{
               if(ticket%SPACE==0) takeprofit=0;else takeprofit=Ask+TAKEPROFIT*Point;
               stoploss=Bid-STOPLOSS*Point;
               //if(cur_lots<0) lots=-cur_lots*MathPow(HEDGE_FACTOR,sells); else lots=Lots;
               lots=Lots;
               //if(cur_lots<0) lots=-cur_lots/sells else lots=Lots;
               //lots=OrderLots()*FACTOR;
            }
        }
        else{
            takeprofit=Ask+TAKEPROFIT*Point;
            stoploss=Bid-STOPLOSS*Point;
            //if(cur_lots<0) lots=-cur_lots*MathPow(HEDGE_FACTOR,sells); else lots=Lots;
            lots=Lots;
        }
        ticket=OpenOrder(OP_BUY,lots,stoploss,takeprofit,"buy");
    }
    if(op==-1){
        if(sells_lots>MAXLOTS)return;
        ticket=SelectMaxOrder(OP_SELL);
        if(ticket!=-1){
            gap=GAP*MathPow(sells,1/3);
            //Print("sells gap:",gap);
            //gap=GAP;
            if(Ask-OrderOpenPrice()<gap*Point){
               if(cur_lots!=0)
                  return;
               //PrintFormat("ticket:%d ordertakeprofit:%f sells:%d",ticket,OrderTakeProfit(),sells);
               stoploss=Ask+STOPLOSS*Point;
               takeprofit=Bid-TAKEPROFIT*Point;
               lots=Lots;
            }
            else{
               stoploss=Ask+STOPLOSS*Point;
               //takeprofit=Bid-TAKEPROFIT*Point;
               //if(cur_lots>0) lots=cur_lots*MathPow(HEDGE_FACTOR,buys); else lots=Lots;
               lots=Lots;
               if(ticket%SPACE==0) takeprofit=0;else takeprofit=Bid-TAKEPROFIT*Point;
               //lots=OrderLots()*FACTOR;
            }              
        }
        else{
            stoploss=Ask+STOPLOSS*Point;
            takeprofit=Bid-TAKEPROFIT*Point;
            //if(cur_lots>0) lots=cur_lots*MathPow(HEDGE_FACTOR,buys); else lots=Lots;
            lots=Lots;
        }
        ticket=OpenOrder(OP_SELL,lots,stoploss,takeprofit,"sell");
    }
}

void CheckOrders10(int op){
    double stoploss,takeprofit,total,profit,lots;
    Lots=AccountEquity()*DECAY_RATE/100000;
    //PrintFormat("equity:%f losts:%f",AccountBalance(),Lots);
    //PrintFormat("const:%f lots:%f",Lots_CONST,Lots);
    if(AccountProfit()>=Lots*TAKEPROFIT){
        CloseAllOrders(2);
    }
    //if(CalculateCurrentProfit(0)>Lots*TAKEPROFIT){
    //    CloseAllOrders(0);
    //}
    //if(CalculateCurrentProfit(1)>Lots*TAKEPROFIT){
    //    CloseAllOrders(1);
    //}
    if(op==0)return;
    if(GetSpread()>SLIPPAGE)return; 
    
    double buys_lots=CalculateCurrentOrdersLots(0);
    double sells_lots=CalculateCurrentOrdersLots(1);
    double cur_lots=buys_lots-sells_lots;
    int buys=CalculateCurrentOrders(0);
    int sells=CalculateCurrentOrders(1);
    PrintFormat("cur_lots:%f buys_lots:%f sells_lots:%f buys:%d sells:%d",cur_lots,buys_lots,sells_lots,buys,sells);
    if(op==1){   
        //if(cur_lots>0)return;
        if(buys>0&&Bars-opbars<Interval)return;
        takeprofit=Ask+TAKEPROFIT*Point; 
        stoploss=Bid-STOPLOSS*Point;
        ticket=OpenOrder(OP_BUY,Lots,stoploss,0,"sell");   
    }
    if(op==-1){
        //if(cur_lots<0)return;
        if(sells>0&&Bars-opbars<Interval)return;
       stoploss=Ask+STOPLOSS*Point;
       takeprofit=Bid-TAKEPROFIT*Point;
       ticket=OpenOrder(OP_SELL,Lots,stoploss,0,"sell");
    }
    opbars=Bars;
    return;
   
    //if(Bars-opbars>Interval){
    //    if(cur_lots>0){
    //       //stoploss=Ask+STOPLOSS*Point;
    //       takeprofit=Bid-TAKEPROFIT*Point;
    //       lots=Lots;
    //       ticket=OpenOrder(OP_SELL,lots,0,takeprofit,"sell");
    //    }
    //    if(cur_lots<0){
    //        //stoploss=Bid-STOPLOSS*Point;
    //        //takeprofit=Ask+TAKEPROFIT*Point;
    //        //ticket=SelectMaxOrder(OP_SELL);
    //        lots=Lots;  
    //        ticket=OpenOrder(OP_BUY,lots,0,takeprofit,"sell");   
    //    }
    //    opbars=Bars;
    //}
}

void CheckOrders6(int op){
    double stoploss,takeprofit,total,profit;
    int ticket=-1,type;
    if(op==0)return;
    if(GetSpread()>SLIPPAGE)return; 

    
    if(OrderSelect(0,SELECT_BY_POS,MODE_TRADES)){
        ticket=OrderTicket();
        type=OrderType();
        profit=OrderProfit();
        if(type==OP_BUY){
            if(profit>0){
                stoploss=OrderOpenPrice()+10*Point;
                takeprofit=Ask+TAKEPROFIT*Point;
                if(NormalizeDouble(OrderStopLoss(),Digits)==stoploss)
                    return;     
                if(OrderModify(OrderTicket(),OrderOpenPrice(),stoploss,OrderTakeProfit(),0,Blue)==false){
                    Print("modify order error ticket:",ticket,GetLastError());
                    return;
                }
                
            }
        }
        if(type==OP_SELL){
            if(profit>0){
                stoploss=OrderOpenPrice()-10*Point;
                takeprofit=Bid-TAKEPROFIT*Point;
                if(NormalizeDouble(OrderStopLoss(),Digits)==stoploss)
                    return;           
                if(OrderModify(OrderTicket(),OrderOpenPrice(),stoploss,OrderTakeProfit(),0,Blue)==false){
                    Print("modify order error ticket:",ticket,GetLastError());
                    return;
                }   
            }
        }
    }
    total=CalculateCurrentOrders(2);
    if(total>0)return;
    
    //double takeprofit,stoploss;
    //PrintFormat("op:%d profit:%f",op,AccountProfit());
    if(op==1){
        stoploss=Bid-STOPLOSS*Point;
        takeprofit=Ask+TAKEPROFIT*Point;
        ticket=OpenOrder(OP_BUY,Lots,stoploss,takeprofit,"buy");
    }
    if(op==-1){
        stoploss=Ask+STOPLOSS*Point;
        takeprofit=Bid-TAKEPROFIT*Point;
        ticket=OpenOrder(OP_SELL,Lots,stoploss,takeprofit,"sell");
    }
    opbars=Bars;
}
void CheckOrders8(int op){
    double stoploss,takeprofit,total,profit,tiny=30;
    int ticket=-1,type;
    if(op==0)return;
    if(GetSpread()>SLIPPAGE)return; 
    
    if(OrderSelect(0,SELECT_BY_POS,MODE_TRADES)){
        ticket=OrderTicket();
        type=OrderType();
        profit=OrderProfit();
        if(type==OP_BUY){
            if(profit>tiny*OrderLots()){
                stoploss=OrderOpenPrice()+tiny*Point;
                takeprofit=Ask+TAKEPROFIT*Point;
                if(NormalizeDouble(OrderStopLoss(),Digits)==stoploss)
                    return;     
                if(OrderModify(OrderTicket(),OrderOpenPrice(),stoploss,OrderTakeProfit(),0,Blue)==false){
                    Print("modify order error ticket:",ticket,GetLastError());
                    return;
                }
                
            }
        }
        if(type==OP_SELL){
            if(profit>tiny*OrderLots()){
                stoploss=OrderOpenPrice()-tiny*Point;
                takeprofit=Bid-TAKEPROFIT*Point;
                if(NormalizeDouble(OrderStopLoss(),Digits)==stoploss)
                    return;           
                if(OrderModify(OrderTicket(),OrderOpenPrice(),stoploss,OrderTakeProfit(),0,Blue)==false){
                    Print("modify order error ticket:",ticket,GetLastError());
                    return;
                }   
            }
        }
    }
    
    //int hour=Hour();
    //int minute=Minute();
    total=CalculateCurrentOrders(2);
//    PrintFormat("hour:%d minute:%d starthour:%d startminute:%d endminute:%d endhour:%d",
//        hour,minute,starthour,startminute,endhour,endminute); 
//    
//    
//    if(hour<starthour)return;
//    else if(minute<startminute)return;
//    if(hour>endhour)return;
//    if(hour==endhour){
//        if(minute>=endminute){
//            if(total>0) CloseAllOrders(2);
//            return;
//        }
//    }
//    
    total=CalculateCurrentOrders(2);
    if(total>0)return;
    
    //double takeprofit,stoploss;
    //PrintFormat("op:%d profit:%f",op,AccountProfit());
    if(op==1){
        stoploss=Bid-STOPLOSS*Point;
        takeprofit=Ask+TAKEPROFIT*Point;
        ticket=OpenOrder(OP_BUY,Lots,stoploss,takeprofit,"buy");
    }
    if(op==-1){
        stoploss=Ask+STOPLOSS*Point;
        takeprofit=Bid-TAKEPROFIT*Point;
        ticket=OpenOrder(OP_SELL,Lots,stoploss,takeprofit,"sell");
    }      
}


//void CheckOrders(int op){
//    double price=0,takeprofit,stoploss;
//    double buys=CalculateCurrentOrders(0);
//    double sells=CalculateCurrentOrders(1);
//    double total=buys+sells;
//    double cur=buys-sells;
//    int ticket;
//    if(op==0)return;
//    if(GetSpread()>SLIPPAGE)return;
//    if(AccountProfit()>=Lots*TAKEPROFIT){
//        CloseAllOrders();
//    }
//    
//    if(op==1){
//        if(buys==0){
//            stoploss=Bid-STOPLOSS*Point;
//            takeprofit=Ask+TAKEPROFIT*Point;
//            OpenOrder(OP_BUY,Lots,stoploss,takeprofit,"buy");
//        }
//        if(buys>0){
//            if(cur>0){
//                ticket=SelectMaxOrder(OP_BUY);
//                price=OrderOpenPrice();
//                if(ticket!=-1&&price-Ask<GAP*Point)
//                    return;
//                OpenOrder(OP_BUY,Lots,0,0,"buy");
//            }
//            //if(cur<0){
//            //    stoploss=Bid-STOPLOSS*Point;
//            //    takeprofit=Ask+TAKEPROFIT*Point;
//            //    OpenOrder(OP_BUY,Lots,stoploss,takeprofit,"buy");
//            //}
//        }        
//    }
//    if(op==-1){
//        if(sells==0){
//            stoploss=Ask+STOPLOSS*Point;
//            takeprofit=Bid-TAKEPROFIT*Point;
//            OpenOrder(OP_SELL,Lots,stoploss,takeprofit,"sell");
//        }
//        if(sells>0){
//            if(cur<0){
//                ticket=SelectMaxOrder(OP_SELL);
//                price=OrderOpenPrice();
//                if(ticket!=-1&&Bid-price<GAP*Point)
//                    return;
//                OpenOrder(OP_SELL,Lots,0,0,"sell");
//            }
//            //if(cur>0){
//            //    stoploss=Ask+STOPLOSS*Point;
//            //    takeprofit=Bid-TAKEPROFIT*Point;
//            //    OpenOrder(OP_SELL,Lots,stoploss,takeprofit,"sell");
//            //}
//        }    
//    }
//    equity=AccountEquity();
//}
void CheckOrders2(){
    
    int ticket;
    if(GetSpread()>SLIPPAGE)return;
    if(AccountProfit()>=Lots*TAKEPROFIT){
        CloseAllOrders(2);
    }
    if(CalculateCurrentProfit(0)>=Lots*TAKEPROFIT){
        CloseAllOrders(0);
    }
    if(CalculateCurrentProfit(1)>=Lots*TAKEPROFIT){
        CloseAllOrders(1);
    }
   
    double price=0,takeprofit,stoploss;
    double buys=CalculateCurrentOrders(0);
    double sells=CalculateCurrentOrders(1);
    double total=buys+sells;
    double cur=buys-sells;
    
    if(buys==0){
        stoploss=Bid-STOPLOSS*Point;
        takeprofit=Ask+TAKEPROFIT*Point;
        OpenOrder(OP_BUY,Lots,0,0,"buy");
    }
    if(buys>0){
        ticket=SelectMaxOrder(OP_BUY);
        price=OrderOpenPrice();
        if(ticket!=-1&&price-Ask<GAP*Point)
            return;
        stoploss=Bid-STOPLOSS*Point;
        takeprofit=Ask+TAKEPROFIT*Point;
        OpenOrder(OP_BUY,FACTOR*buys,0,0,"buy");
    }
    if(sells==0){
        stoploss=Ask+STOPLOSS*Point;
        takeprofit=Bid-TAKEPROFIT*Point;
        OpenOrder(OP_SELL,Lots,0,0,"sell");
    } 
    if(sells>0){
        ticket=SelectMaxOrder(OP_SELL);
        price=OrderOpenPrice();
        if(ticket!=-1&&Bid-price<GAP*Point)
            return;
        stoploss=Ask+STOPLOSS*Point;
        takeprofit=Bid-TAKEPROFIT*Point;
        OpenOrder(OP_SELL,FACTOR*sells,0,0,"sell");
    }       
    
}
void CheckOrders4(int op){
    
    int ticket;
    double stoploss,takeprofit;
    double total=CalculateCurrentOrders(2);
    if(total>0)return;
    //if(total>=Lots*LIMIT)return;
    if(GetSpread()>SLIPPAGE)return;
    //if(AccountProfit()>=Lots*TAKEPROFIT){
    //    CloseAllOrders(2);
    //}
    
    if(-op==1){
        stoploss=Bid-STOPLOSS*Point;
        takeprofit=Ask+TAKEPROFIT*Point;
        //OpenOrder(OP_BUY,Lots,0,0,"buy");
        OpenOrder(OP_BUY,Lots,stoploss,takeprofit,"buy");
    }
    
    if(-op==-1){
        stoploss=Ask+STOPLOSS*Point;
        takeprofit=Bid-TAKEPROFIT*Point;
        //OpenOrder(OP_SELL,Lots,0,0,"sell");
        OpenOrder(OP_SELL,Lots,stoploss,takeprofit,"sell");
    }  
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

string SendMessage(int sock,string msg,string cmd)
{
    uchar buf[];
    uchar cmdbuf[];
    int ret;
    
    //int sock=Sock_Connect(PORT,HOST);
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
    //Print("send ",msg);
    if (ret == SOCKET_ERROR) {
        Print(" send() failed: error "+WSAGetLastError());
        return NULL;
    }
    ArrayInitialize(buf,0);
    recv(sock,buf,BUFSIZE,0);
    //Sock_Close(sock);
    string str=CharArrayToString(buf);
    //Print("recv:",str);
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
        Print("Server: WSAStartup() failed with error ",retval);
        return(-1);
    }
    sock=socket(AF_INET, SOCK_STREAM,0);
    if (sock == INVALID_SOCKET){
        Print("Server: socket() failed with error ",WSAGetLastError());
        WSACleanup();
        return(-1);
    }

    //ref.ref=addrin;
    retval=connect(sock,addrin,16);
    if (retval != 0) {
        Print("connect failed with error ",WSAGetLastError());
        WSACleanup();
        return(-1);
    }
    return sock;
}



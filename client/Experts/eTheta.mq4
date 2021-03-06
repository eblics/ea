//+------------------------------------------------------------------+
//|                                               Moving Average.mq4 |
//|                   Copyright 2005-2014, MetaQuotes Software Corp. |
//|                                              http://www.mql4.com |
//+------------------------------------------------------------------+
#property copyright   "2005-2014, MetaQuotes Software Corp."
#property link        "http://www.mql4.com"
#property description "Moving Average sample expert advisor"

#define MAGICMA  20131111
//--- Inputs
input double Lots          =0.1;
//input int LotsLimit=10;
input int SLIPPAGE=3;
input int STOPLOSS=50;
input int TAKEPROFIT=20;
input int Period=13;
input double Theta=0.5;
input int Interval=13;

double GetTheta(){
    return iCustom(Symbol(),0,"ea/iTheta",Period,0,0);
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

int ticket=-1;
int lastbar=0;
//+------------------------------------------------------------------+
//| Check for open order conditions                                  |
//+------------------------------------------------------------------+
void CheckOrders()
{
    int res=0;
    int op=0;
    int i,total;
    double price,type,stoploss,takeprofit,profit;
    
    double theta=GetTheta();
    int tickets[];
    total=OrdersTotal();
    ArrayResize(tickets,total);
    ArrayInitialize(tickets,-1);
    for(i=0;i<total;i++){
        if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES)==false){
            Print("i:",i," ",GetLastError());
        }
        tickets[i]=OrderTicket();
        
    }
    for(i=0;i<total;i++){
        if(tickets[i]==-1)continue;
        if(OrderSelect(tickets[i],SELECT_BY_TICKET,MODE_TRADES)==false){
            Print("i:",i," ",GetLastError());
        }
        type=OrderType();
        profit=OrderProfit();
        if(type==OP_BUY){
            if(theta>Theta&&profit>0){
                if(OrderClose(ticket,OrderLots(),Bid,SLIPPAGE,Red)==false){
                    Print("close order error ticket:",ticket,GetLastError());
                    return;
                }
                ticket=-1;
            }
        }
        if(type==OP_SELL){
            if(-theta>Theta&&profit>0){
                if(OrderClose(ticket,OrderLots(),Ask,SLIPPAGE,Red)==false){
                    Print("close order error ticket:",ticket,GetLastError());
                    return;
                }
                ticket=-1;
            }
        }
    }
    ArrayFree(tickets);
    if(Bars-lastbar<Interval)return;
    
    PrintFormat("ticket:%d theta:%f",ticket,theta); 
    if(theta>Theta){
        stoploss=Ask+STOPLOSS*Point;
        takeprofit=Bid-TAKEPROFIT*Point;
        ticket=OpenOrder(OP_SELL,Lots,stoploss,takeprofit,"");
    }
    if(-theta>Theta){    
        stoploss=Bid-STOPLOSS*Point;
        takeprofit=Ask+TAKEPROFIT*Point;
        ticket=OpenOrder(OP_BUY,Lots,stoploss,takeprofit,"");
    } 
    lastbar=Bars;    
}

int cbar=0;
///+------------------------------------------------------------+/
//+------------------------------------------------------------------+
//| OnTick function                                                  |
//+------------------------------------------------------------------+
void OnTick()
{
     if(Bars>cbar){
        CheckOrders();
        cbar=Bars;
     }  
}

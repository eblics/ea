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
input double Blast  =0.000100;
input double BlastLimit=0.00100;
input int SlipPage=3;
input int StopLoss=50;
input int TakeProfit=20;
input int Period=3;

//int cnt=0;
//+------------------------------------------------------------------+
//| Calculate open positions                                         |
//+------------------------------------------------------------------+
int CalculateCurrentOrders(string symbol)
{
    int buys=0,sells=0,bl=0,sl=0;
    //---
    for(int i=0;i<OrdersTotal();i++)
    {
        if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES)==false) break;
        if(OrderSymbol()==Symbol() && OrderMagicNumber()==MAGICMA)
        {
            if(OrderType()==OP_BUY)  buys++;
            if(OrderType()==OP_SELL) sells++;
            if(OrderType()==OP_BUYLIMIT) bl++;
            if(OrderType()==OP_SELLLIMIT) sl++;
        }
    }
    //--- return orders volume
    return buys+sells+bl+sl;
}

double GetLastOrderProfit()
{
    int i,hstTotal=OrdersHistoryTotal();
    if(OrderSelect(hstTotal-1,SELECT_BY_POS,MODE_HISTORY)==false)
    {
        Print("Access to history failed with error (",GetLastError(),")");
        return 0;
    }
    return OrderProfit();
    
}
double GetIV()
{
    return iCustom(Symbol(),0,"ea/iV",5,33,0,0);
}

int OpenOrder(int op,double price,string notes)
{
    Print(notes);
    int res=0;
    //if(iv2==0x7FFFFFFF)
    if(op==OP_SELL)
    {
        res=OrderSend(Symbol(),OP_SELL,Lots,Bid,SlipPage,0,0,"",MAGICMA,0,Red);
        //res=OrderSend(Symbol(),OP_SELL,Lots,Bid,SlipPage,Ask+StopLoss*Point,Ask-TakeProfit*Point,"",MAGICMA,0,Red);
        if(res!=0){
            Print(GetLastError());
        }
        return res;
    }
    //--- buy conditions
    if(op==OP_BUY)
    {
        res=OrderSend(Symbol(),OP_BUY,Lots,Ask,SlipPage,0,0,"",MAGICMA,0,Blue);
        //res=OrderSend(Symbol(),OP_BUY,Lots,Ask,SlipPage,Bid-StopLoss*Point,Bid+TakeProfit*Point,"",MAGICMA,0,Blue);
        if(res!=0){
            Print(GetLastError());
        }
        return res;
    }
}

void CloseAllOrders()
{
    int res=0;
    for(int i=0;i<OrdersTotal();i++)
    {
        if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES)==false) break;
        if(OrderType()==OP_BUY)
        {
            if(OrderClose(OrderTicket(),OrderLots(),Bid,SlipPage,Red)==false){
                Print(GetLastError());
            }
        }
        else if(OrderType()==OP_SELL){
            if(OrderClose(OrderTicket(),OrderLots(),Ask,SlipPage,Red)==false){
                Print(GetLastError());
            }
        }
    }
}


int ticket=-1;
//+------------------------------------------------------------------+
//| Check for open order conditions                                  |
//+------------------------------------------------------------------+
void CheckOrders()
{
    int res=0;
    int op=-1,type=-1;
    double price,stoploss,takeprofit,profit;
    double x=0,sx=1;  
    if(ticket==-1){
        op=OP_BUY;
    }
    else
    {
        if(OrderSelect(ticket,SELECT_BY_TICKET)==false){
            Print("select order error ",GetLastError());
            return;
        }
        type=OrderType();
        profit=OrderProfit();
        if(profit<0){
            op=type==OP_SELL?OP_BUY:OP_SELL;
        }
        else{
            op=type;
        }
        PrintFormat("type:%d op:%d profit:%f",type,op,profit);
    }
    if(op==OP_SELL){
        //op=OP_SELL;
        price=Bid;
        stoploss=price+StopLoss*Point;
        takeprofit=price-TakeProfit*Point;
        
    }
    if(op==OP_BUY) {
        //op=OP_BUY;
        price=Ask;
        stoploss=price-StopLoss*Point;
        takeprofit=price+TakeProfit*Point;
    }
    
    ticket=OrderSend(Symbol(),op,Lots,price,SlipPage,stoploss,takeprofit,"",MAGICMA,0,Red);
    
   //x=Close[1]-Close[Period+1];    
//    if(x>=Blast&&x<=BlastLimit){
//        PrintFormat("x:%f spread:%f",x,MathAbs(Ask-Bid));
//        op=OP_SELL;
//        price=Bid;
//        stoploss=price+StopLoss*Point;
//        takeprofit=price-TakeProfit*Point;
//        
//    }
//    if(-x>=Blast&&-x<=BlastLimit) {
//        op=OP_BUY;
//        price=Ask;
//        stoploss=price-StopLoss*Point;
//        takeprofit=price+TakeProfit*Point;
//    }
    
}

int cbar=0;
///+------------------------------------------------------------+/
//+------------------------------------------------------------------+
//| OnTick function                                                  |
//+------------------------------------------------------------------+
void OnTick()
{
    //--- check for history and trading
    if(Bars<Period || IsTradeAllowed()==false)
        return;
    //--- calculate open orders by current symbol
    int cnt=CalculateCurrentOrders(Symbol());
    if(cnt>0)return;
    if(cbar==Bars)return;
    CheckOrders();    
    cbar=Bars;
}

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
input double Blast  =0.00050;
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

void OpenOrder(int op,double price,string notes)
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
        return;
    }
    //--- buy conditions
    if(op==OP_BUY)
    {
        res=OrderSend(Symbol(),OP_BUY,Lots,Ask,SlipPage,0,0,"",MAGICMA,0,Blue);
        //res=OrderSend(Symbol(),OP_BUY,Lots,Ask,SlipPage,Bid-StopLoss*Point,Bid+TakeProfit*Point,"",MAGICMA,0,Blue);
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



//+------------------------------------------------------------------+
//| Check for open order conditions                                  |
//+------------------------------------------------------------------+
void CheckOrders()
{
    int res=0;
    int op=0;
    double price,stoploss,takeprofit,profit;
    double x=0,sx=1;
    x=sx=Close[1]-Open[1];
    for(int i=2;i<Period;i++){
      sx*=(Close[i]-Open[i]);
      if(sx<0)
        break;
       x+=Close[i]-Open[i];  
    }
    //PrintFormat("x:%f",x);
    //profit=AccountProfit();
    //if(profit>0){
    //    CloseAllOrders();
    //}
    //if(MathAbs(x)>Blast && MathAbs(x)<BlastLimit){
    if(MathAbs(x)>Blast){
        PrintFormat("x:%f spread:%f",x,MathAbs(Ask-Bid));
        if(x>0){
            op=OP_SELL;
            price=Bid;
            stoploss=price+StopLoss*Point;
            takeprofit=price-TakeProfit*Point;
        }
        if(x<0) {
            op=OP_BUY;
            price=Ask;
            stoploss=price-StopLoss*Point;
            takeprofit=price+TakeProfit*Point;
        }
        res=OrderSend(Symbol(),op,Lots,price,SlipPage,stoploss,takeprofit,"",MAGICMA,0,Red);
    }
}

int cbar=0;
///+------------------------------------------------------------+/
//+------------------------------------------------------------------+
//| OnTick function                                                  |
//+------------------------------------------------------------------+
void OnTick()
{
    //--- check for history and trading
    if(Bars<100 || IsTradeAllowed()==false)
        return;
    //--- calculate open orders by current symbol
    int cnt=CalculateCurrentOrders(Symbol());
    double profit;
    if(cnt==0){
        profit=GetLastOrderProfit();
        if(profit<0){
            if(Bars-cbar<5)
                return;
        }
        if(Bars-cbar>=1)
            CheckOrders();    
    }
    if(Bars>cbar) cbar=Bars;
}

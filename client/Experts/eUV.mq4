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
input double MaximumRisk   =1;
input double DecreaseFactor=3;
double MINU=0.000012;
double MAXU=0.00004;
double MAXV=0.001;
double MINV=0.00020;
input int    MovingPeriod  =144;
input int    MovingShift   =0;
input int    PeriodU=33;
int    Loses=0;
input int    StopLoss=200;
input int    TakeProfit=200;



//+------------------------------------------------------------------+
//| Calculate open positions                                         |
//+------------------------------------------------------------------+
int CalculateCurrentOrders(string symbol)
  {
   int buys=0,sells=0;
//---
   for(int i=0;i<OrdersTotal();i++)
     {
      if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES)==false) break;
      if(OrderSymbol()==Symbol() && OrderMagicNumber()==MAGICMA)
        {
         if(OrderType()==OP_BUY)  buys++;
         if(OrderType()==OP_SELL) sells++;
        }
     }
//--- return orders volume
   if(buys>0) return(buys);
   else       return(-sells);
  }
//+------------------------------------------------------------------+
//| Calculate optimal lot size                                       |
//+------------------------------------------------------------------+
double LotsOptimized()
  {
   double lot=Lots;
   int    orders=HistoryTotal();     // history orders total
   int    losses=0;                  // number of losses orders without a break
//--- select lot size
   lot=NormalizeDouble(AccountFreeMargin()*MaximumRisk/1000.0,2);
//--- calcuulate number of losses orders without a break
   if(DecreaseFactor>0)
     {
      for(int i=orders-1;i>=0;i--)
        {
         if(OrderSelect(i,SELECT_BY_POS,MODE_HISTORY)==false)
           {
            Print("Error in history!");
            break;
           }
         if(OrderSymbol()!=Symbol() || OrderType()>OP_SELL)
            continue;
         //---
         if(OrderProfit()>0) losses--;
         if(OrderProfit()<0) losses++;
        }
      if(losses>1)
         lot=NormalizeDouble(lot-lot*losses/DecreaseFactor,1);
         
     }
     PrintFormat("lot1:%f lot2:%f",lot,NormalizeDouble(AccountFreeMargin()*MaximumRisk/1000.0,2));
//--- return lot size
   if(lot<Lots) lot=Lots;
   return(lot);
  }
  
double GetU(){
    double ma_cur,ma_pre;
    ma_cur=iMA(NULL,0,MovingPeriod,MovingShift,MODE_SMA,PRICE_CLOSE,0);
    ma_pre=iMA(NULL,0,MovingPeriod,MovingShift,MODE_SMA,PRICE_CLOSE,PeriodU);
    return (ma_cur-ma_pre)/PeriodU;
}
double GetV(){
    double ma,v;
    for(int i=1;i<=PeriodU;i++)
    {
        ma=iMA(NULL,0,MovingPeriod,MovingShift,MODE_SMA,PRICE_CLOSE,i);
        v+=MathAbs(Close[i]-ma);
    }
    v=v/PeriodU;
    return v;
}
double op_u=0,op_v=0;
//+------------------------------------------------------------------+
//| Check for open order conditions                                  |
//+------------------------------------------------------------------+
void CheckForOpen()
  {
   double price,u,v,b,ma,stoploss;
   int    res;
//--- go trading only for first tiks of new bar
   if(Volume[0]>1) return;
   if(Loses<0)Loses=0;
   if(Loses>=3){
    PrintFormat("loses:%d",Loses);
    //PeriodU+=17;
   }
   //if(PeriodU>200)PeriodU=33;
//--- get Moving Average 
    u=GetU();
    v=GetV();
    ma=iMA(NULL,0,MovingPeriod,MovingShift,MODE_SMA,PRICE_CLOSE,0);
    b=Bid-ma;
    if(MathAbs(u)>MAXU||v>MAXV){
        PrintFormat("u:%f or v:%f is too big",u,v);
        return;
    }
    if(u>=MINU){
        if(b<v/3){
            stoploss=Ask-StopLoss*Point;
            res=OrderSend(Symbol(),OP_BUY,LotsOptimized(),Ask,3,stoploss,0,"",MAGICMA,0,Blue);
            op_u=u;op_v=v;
            Print("open buy because 1");
            PrintFormat("Ask:%f Bid:%f ma:%f u:%f v:%f b:%f op_u:%f op_v:%f",Ask,Bid,ma,u,v,b,op_u,op_v);
            return;
        }
    }
    if(u<=-MINU){
        if(b>v/3){
            stoploss=Bid+StopLoss*Point;
            res=OrderSend(Symbol(),OP_SELL,LotsOptimized(),Bid,3,stoploss,0,"",MAGICMA,0,Red);
            op_u=u;op_v=v;
            Print("open sell because 1");
            PrintFormat("Ask:%f Bid:%f ma:%f u:%f v:%f b:%f op_u:%f op_v:%f",Ask,Bid,ma,u,v,b,op_u,op_v);
            return;
        }
    }
    if(MathAbs(u)<MINU){
        if(b>2*v){
            stoploss=Bid+StopLoss*Point;
            res=OrderSend(Symbol(),OP_SELL,LotsOptimized(),Bid,3,stoploss,0,"",MAGICMA,0,Red);
            op_u=u;op_v=v;
            Print("open sell because 2");
            PrintFormat("Ask:%f Bid:%f ma:%f u:%f v:%f b:%f op_u:%f op_v:%f",Ask,Bid,ma,u,v,b,op_u,op_v);
            return;
        }
        if(-b>2*v){
            stoploss=Ask-StopLoss*Point;
            res=OrderSend(Symbol(),OP_BUY,LotsOptimized(),Ask,3,stoploss,0,"",MAGICMA,0,Blue);
            op_u=u;op_v=v;
            Print("open buy because 2");
            PrintFormat("Ask:%f Bid:%f ma:%f u:%f v:%f b:%f op_u:%f op_v:%f",Ask,Bid,ma,u,v,b,op_u,op_v);
            return;
        }
    }
  }
//+------------------------------------------------------------------+
//| Check for close order conditions                                 |
//+------------------------------------------------------------------+
void CheckForClose()
  {
   double ma,u,v,b,stoploss;
//--- go trading only for first tiks of new bar
   if(Volume[0]>1) return;
//--- get Moving Average 
    u=GetU();
    v=GetV();
    ma=iMA(NULL,0,MovingPeriod,MovingShift,MODE_SMA,PRICE_CLOSE,0);
    b=Bid-ma;
    //PrintFormat("Ask:%f Bid:%f ma:%f u:%f v:%f b:%f op_u:%f op_v:%f",Ask,Bid,ma,u,v,b,op_u,op_v);
   for(int i=0;i<OrdersTotal();i++)
     {
      if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES)==false) break;
      if(OrderMagicNumber()!=MAGICMA || OrderSymbol()!=Symbol()) continue;
      //--- check order type 
      if(OrderType()==OP_BUY)
        {
         if(Bid-OrderOpenPrice()>=30){
            stoploss=OrderOpenPrice()+10;
            if(OrderModify(OrderTicket(),OrderOpenPrice(),stoploss,OrderTakeProfit(),0,Blue)==false){
                PrintFormat("modify order#%d error:%d",OrderTicket(),GetLastError());
            }
         }
         if(op_u>MINU&&MathAbs(u)>MINU&&u<=0)
           {
            if(!OrderClose(OrderTicket(),OrderLots(),Bid,3,White))
               Print("OrderClose error ",GetLastError());
            else{
                Print("close buy because 1");
                PrintFormat("Ask:%f Bid:%f ma:%f u:%f v:%f b:%f op_u:%f op_v:%f",Ask,Bid,ma,u,v,b,op_u,op_v);
                if(OrderProfit()<0)Loses++; 
                op_u=0;op_v=0;
                continue;
            }
            
           }
          if(op_u>MINU&&b>3*v){
            if(!OrderClose(OrderTicket(),OrderLots(),Bid,3,White))
               Print("OrderClose error ",GetLastError());
            else{
                Print("close buy because 2");
                PrintFormat("Ask:%f Bid:%f ma:%f u:%f v:%f b:%f op_u:%f op_v:%f",Ask,Bid,ma,u,v,b,op_u,op_v);
                if(OrderProfit()<0)Loses++; 
                op_u=0;op_v=0;
                continue;
            }  
          }
          if(MathAbs(op_u)<=MINU){
            if(u<-MINU)
            {
                if(!OrderClose(OrderTicket(),OrderLots(),Bid,3,White))
                   Print("OrderClose error ",GetLastError());
                else{
                    Print("close buy because 3");
                    PrintFormat("Ask:%f Bid:%f ma:%f u:%f v:%f b:%f op_u:%f op_v:%f",Ask,Bid,ma,u,v,b,op_u,op_v);
                    if(OrderProfit()<0)Loses++; 
                    op_u=0;op_v=0;
                    continue;
                }
            }
            if(MathAbs(u)<MINU&&b>v&&OrderProfit()>0){
                if(!OrderClose(OrderTicket(),OrderLots(),Bid,3,White))
                   Print("OrderClose error ",GetLastError());
                else{
                    Print("close buy because 4");
                    PrintFormat("Ask:%f Bid:%f ma:%f u:%f v:%f b:%f op_u:%f op_v:%f",Ask,Bid,ma,u,v,b,op_u,op_v);
                    if(OrderProfit()<0)Loses++; 
                    op_u=0;op_v=0;
                    continue;
                }
            }
          }
           
         break;
        }
      if(OrderType()==OP_SELL)
        {
         if(OrderOpenPrice()-Ask>=30){
            stoploss=OrderOpenPrice()-10;
            if(OrderModify(OrderTicket(),OrderOpenPrice(),stoploss,OrderTakeProfit(),0,Blue)==false){
                PrintFormat("modify order#%d error:%d",OrderTicket(),GetLastError());
            }
         }
         if(op_u<-MINU&&MathAbs(u)>MINU&&u>=0)
           {
            if(!OrderClose(OrderTicket(),OrderLots(),Ask,3,White))
               Print("OrderClose error ",GetLastError());
            else{
                Print("close sell because 1");
                PrintFormat("Ask:%f Bid:%f ma:%f u:%f v:%f b:%f op_u:%f op_v:%f",Ask,Bid,ma,u,v,b,op_u,op_v);
                if(OrderProfit()<0)Loses++; 
                op_u=0;op_v=0;
                continue;
            }
           }
         if(op_u<-MINU&&-b>3*v){
            if(!OrderClose(OrderTicket(),OrderLots(),Ask,3,White))
               Print("OrderClose error ",GetLastError());
            else{
                Print("close sell because 2");
                PrintFormat("Ask:%f Bid:%f ma:%f u:%f v:%f b:%f op_u:%f op_v:%f",Ask,Bid,ma,u,v,b,op_u,op_v);
                if(OrderProfit()<0)Loses++; 
                op_u=0;op_v=0;
                continue;
            }
         }
         if(MathAbs(op_u)<MINU){
            if(u>MINU){
                if(!OrderClose(OrderTicket(),OrderLots(),Ask,3,White))
                   Print("OrderClose error ",GetLastError());
                else{
                    Print("close sell because 3");
                    PrintFormat("Ask:%f Bid:%f ma:%f u:%f v:%f b:%f op_u:%f op_v:%f",Ask,Bid,ma,u,v,b,op_u,op_v);
                    if(OrderProfit()<0)Loses++; 
                    op_u=0;op_v=0;
                    continue;
                }
            }
            if(MathAbs(u)<MINU&&-b>v&&OrderProfit()>0)
            {
                if(!OrderClose(OrderTicket(),OrderLots(),Ask,3,White))
                   Print("OrderClose error ",GetLastError());
                else{
                    Print("close sell because 4");
                    PrintFormat("Ask:%f Bid:%f ma:%f u:%f v:%f b:%f op_u:%f op_v:%f",Ask,Bid,ma,u,v,b,op_u,op_v);
                    if(OrderProfit()<0)Loses++; 
                    op_u=0;op_v=0;
                    continue;
                }
            }
         }
         break;
        }
     }
//---
  }
//+------------------------------------------------------------------+
//| OnTick function                                                  |
//+------------------------------------------------------------------+
void OnTick()
  {
//--- check for history and trading
   if(Bars<MovingPeriod+PeriodU || IsTradeAllowed()==false)
      return;
//--- calculate open orders by current symbol
   if(CalculateCurrentOrders(Symbol())==0) CheckForOpen();
   else                                    CheckForClose();
//---
  }
//+------------------------------------------------------------------+

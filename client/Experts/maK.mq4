//+------------------------------------------------------------------+
//|                                               Moving Average.mq4 |
//|                   Copyright 2005-2014, MetaQuotes Software Corp. |
//|                                              http://www.mql4.com |
//+------------------------------------------------------------------+
#property copyright   "eblics"
#property link        "http://www.eblics.com"
#property description "mak"
#include <MovingAverages.mqh>
input int SHIFT=21;
input int MA1=55;
input int MA2=144;
input double SLOP=0.3;
input double Lots=0.1;
input double OFFSET=0.0008;
input int LOSSSTEP=200;
input int PROFITSTEP=500;
input double EK=0.618;
input double ED=5;
//包络线常数
input double DVT=0.11; 
const int SLIPPAGE=3;

int bar;
int ticket=-1;
void OnTick()
{
    plc_ma_kd();
}

void plc_ma_kd()
{
    double k,d;
    double ka[];
    double ma1b,ma2b;
    double price,stoploss,takeprofit,profit;
    ArrayResize(ka,SHIFT);
    k=K(SHIFT);
    d=D();
    ma1b=iMA(Symbol(),PERIOD_CURRENT,MA1,0,MODE_SMA,PRICE_CLOSE,0);
    ma2b=iMA(Symbol(),PERIOD_CURRENT,MA2,0,MODE_SMA,PRICE_CLOSE,0);
    PrintFormat("K:%f, D:%f",k,d);
    if(OrdersTotal()==0&&MathAbs(k)>EK){
      if(k<0&&d>=0){
         //if(Ask<=ma1b){
            stoploss=Ask-LOSSSTEP*Point;
            takeprofit=Ask+PROFITSTEP*Point;
            ticket=OrderSend(Symbol(),OP_BUY,Lots,Ask,SLIPPAGE,stoploss,takeprofit,"",0,0,Green);
            if(ticket==-1){
               PrintFormat("buy error %f %f %f",Bid,stoploss,takeprofit);               
            //}
         }
      }
      if(k>0&&d<=0){
         //if(Bid>=ma1b){
            stoploss=Bid+LOSSSTEP*Point;
            takeprofit=Bid-PROFITSTEP*Point;
            ticket=OrderSend(Symbol(),OP_SELL,Lots,Bid,SLIPPAGE,stoploss,takeprofit,"",0,0,Green);
            if(ticket==-1){
               PrintFormat("buy error %f %f %f",Bid,stoploss,takeprofit);
            }
         //}
      }
    } 
    
    if(OrdersTotal()>0){
      OrderSelect(0,SELECT_BY_POS);
      int ticket=OrderTicket();
      int orderType=OrderType();
      price=OrderOpenPrice();
      profit=OrderProfit();
      if(orderType==OP_BUY){
         if(d==0){
            OrderClose(ticket,OrderLots(),Bid,SLIPPAGE,Green);
         }
         else{
            stoploss=(float)(Bid-LOSSSTEP*Point);
            takeprofit=(float)(Bid+PROFITSTEP*Point);
            if(stoploss-OrderStopLoss()>20*Point) 
               OrderModify(ticket,0,stoploss,takeprofit,0,Green);
         }
      }  
      else{
          if(MathAbs(k)>EK&&k<0&&d>=0)
          if(d==0){
            //OrderClose(ticket,OrderLots(),Ask,SLIPPAGE,Green);
          }
           else{
            stoploss=(float)(Ask-LOSSSTEP*Point);
            takeprofit=(float)(Ask+PROFITSTEP*Point);
            if(stoploss-OrderStopLoss()>20*Point) 
               OrderModify(ticket,0,stoploss,takeprofit,0,Green);
         }
      }
    }   
}
double K(int shift){
   return (MA(MA1,shift)-MA(MA1,1))/shift/Point;
}
   
double D()
{
   return (MA(MA1,1)-MA(MA1,2))/Point;
}

double MA(int period,int shift){
   return iMA(Symbol(),PERIOD_CURRENT,period,0,MODE_SMA,PRICE_CLOSE,shift);
}

double EVL(int period,int shift,int dir)
{
   return iEnvelopes(Symbol(),PERIOD_CURRENT,period, MODE_SMA,shift,PRICE_CLOSE,DVT,dir,shift);
}

double plc_ma_cross(){
   double ma2a,ma2b,ma1a,ma1b;
   double k1,k2;
   int ticket=-1;
   int orderType;
   double price,stoploss,takeprofit,profit;
   if(Bars<MA2+SHIFT || IsTradeAllowed()==false)
      return;
   ma2a=iMA(Symbol(),PERIOD_CURRENT,MA2,0,MODE_SMA,PRICE_CLOSE,SHIFT);
   ma2b=iMA(Symbol(),PERIOD_CURRENT,MA2,0,MODE_SMA,PRICE_CLOSE,0);
   ma1a=iMA(Symbol(),PERIOD_CURRENT,MA1,0,MODE_SMA,PRICE_CLOSE,SHIFT);
   ma1b=iMA(Symbol(),PERIOD_CURRENT,MA1,0,MODE_SMA,PRICE_CLOSE,0);
   k2=(ma2b-ma2a)/(SHIFT*Point);
   k1=(ma1b-ma1a)/(SHIFT*Point);
   //if(MathAbs(k2)<SLOP||MathAbs(k1)<SLOP)
   //   return;
   if(k1<0&&ma1b>=ma2b&&MathAbs(k1)>SLOP){
      stoploss=(float)(Bid+LOSSSTEP*Point);
      takeprofit=(float)(Bid-PROFITSTEP*Point);
      //PrintFormat("sell bid:%f stoploss:%f takeprofit:%f",Bid,stoploss,takeprofit);
      if(OrdersTotal()==0){
         ticket=OrderSend(Symbol(),OP_SELL,Lots,Bid,SLIPPAGE,stoploss,takeprofit,"",0,0,Green);
         if(ticket==-1){
            PrintFormat("buy error %f %f %f",Bid,stoploss,takeprofit);
         }
      }
      else{
         OrderSelect(0,SELECT_BY_POS);
         ticket=OrderTicket();
         orderType=OrderType();
         price=OrderOpenPrice();
         profit=OrderProfit();
         if(orderType==OP_BUY){
            OrderClose(ticket,OrderLots(),Bid,SLIPPAGE,Green);
         }
      }
   }
   if(k1>0&&ma1b>=ma2b&&MathAbs(k1)>SLOP){
      stoploss=(float)(Ask-LOSSSTEP*Point);
      takeprofit=(float)(Ask+PROFITSTEP*Point);
      //PrintFormat("buy ask:%f stoploss:%f takeprofit:%f",Ask,stoploss,takeprofit);
      if(OrdersTotal()==0){
         ticket=OrderSend(Symbol(),OP_BUY,Lots,Ask,SLIPPAGE,stoploss,takeprofit,"",0,0,Green);
         PrintFormat("ticket is %d",ticket);
         if(ticket==-1){
            PrintFormat("sell error %f %f %f",Ask,stoploss,takeprofit);
         }
      }
      else{
         OrderSelect(0,SELECT_BY_POS);
         ticket=OrderTicket();
         orderType=OrderType();
         price=OrderOpenPrice();
         profit=OrderProfit();
         if(orderType==OP_SELL){
            OrderClose(ticket,OrderLots(),Ask,SLIPPAGE,Green);
         }
      }
   }
      
   if(OrdersTotal()>0){
      
      if(!OrderSelect(0,SELECT_BY_POS)){
         Print(GetLastError());
         return;
      }
      ticket=OrderTicket();
      orderType=OrderType();
      price=OrderOpenPrice();
      profit=OrderProfit();
      
      if(orderType==OP_BUY){
        
          stoploss=(float)(Bid-LOSSSTEP*Point);
          takeprofit=(float)(Bid+PROFITSTEP*Point);
          if(stoploss-OrderStopLoss()>20*Point) 
            OrderModify(ticket,0,stoploss,takeprofit,0,Green);
      }
      
      if(orderType==OP_SELL){
         PrintFormat("sell ####################################modify %d  %d",ticket,OrdersTotal());
          stoploss=(float)(Ask+LOSSSTEP*Point);
          takeprofit=(float)(Ask-PROFITSTEP*Point);
          if(OrderStopLoss()-stoploss>20*Point)
            OrderModify(ticket,0,stoploss,takeprofit,0,Green);
      }
     
   }    
}
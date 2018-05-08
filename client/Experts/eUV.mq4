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
input double MaximumRisk   =0.1;
input double DecreaseFactor=3;
input double MINU=0.2;
//input double MAXU=7.05;
//input double MAXV=286;
double MINV=20;
input double VFACTOR1=0;
input double VFACTOR2=1.5;
input double VFACTOR3=3;
input int    MovingPeriod  =144;
input int    MovingShift   =0;
input int    PeriodU=5;
int    Loses=0;
//input int    StopLoss=500;
//input int    TakeProfit=200;
double MA_Bars[];
double Var_Bars[];
double vfactor1=VFACTOR1,vfactor2=VFACTOR2,vfactor3=VFACTOR3;
int Ticket;



int CalculateCurrentOrders(string symbol)
  {
   int buys=0,sells=0;

   for(int i=0;i<OrdersTotal();i++)
     {
      if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES)==false) break;
      if(OrderSymbol()==Symbol() && OrderMagicNumber()==MAGICMA)
        {
         if(OrderType()==OP_BUY)  buys++;
         if(OrderType()==OP_SELL) sells++;
        }
     }

   if(buys>0) return(buys);
   else       return(-sells);
  }
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
         if(OrderProfit()>0) break;
         if(OrderProfit()<0) losses++;
        }
      if(losses>1){
         lot=NormalizeDouble(lot-lot*losses/DecreaseFactor,2);
         //vfactor3=vfactor3+0.1*losses/DecreaseFactor;
      }
     }
//--- return lot size
   if(lot<0.01) lot=0.01;
   //if(vfactor3>5)vfactor3=VFACTOR3;
   return(lot);
  }
  
double GetU(){
    double ma_cur,ma_pre;
    ArraySetAsSeries(MA_Bars,true);
    ma_cur=MA_Bars[0];
    ma_pre=MA_Bars[PeriodU-1];
    ArraySetAsSeries(MA_Bars,false);
    //ma_cur=iMA(NULL,0,MovingPeriod,MovingShift,MODE_SMA,PRICE_CLOSE,0);
    //ma_pre=iMA(NULL,0,MovingPeriod,MovingShift,MODE_SMA,PRICE_CLOSE,PeriodU-1);
    //PrintFormat("ma_pre:%f ma_cur:%f",ma_pre,ma_cur);
    return (ma_cur-ma_pre)/Point/PeriodU;
}
double GetV(){
    double ma,v;
    int size=ArraySize(Var_Bars);
    for(int i=size-1;i>=size-MovingPeriod;i--){
        v+=Var_Bars[i];
    }
    //for(int i=0;i<PeriodU;i++)
    //{
    //    ma=iMA(NULL,0,MovingPeriod,MovingShift,MODE_SMA,PRICE_CLOSE,i);
    //    v+=MathAbs(Close[i]-ma);
    //}
    v=v/Point/MovingPeriod;
    return v;
}

double GetMaxV()
{
    double maxv=0;
    int size=ArraySize(Var_Bars);
    for(int i=size-1;i>=size-MovingPeriod;i--){
        if(Var_Bars[i]>maxv)
            maxv=Var_Bars[i];
    }
    return maxv/Point;
}
double op_u=0,op_v=0;
//+------------------------------------------------------------------+
//| Check for open order conditions                                  |
//+------------------------------------------------------------------+


int CUR_BAR=0;
void OnTick()
  {
    int i=0,j=0,size;
    double ma=0,ima=0;
   if(Bars<MovingPeriod+PeriodU || IsTradeAllowed()==false)
      return;
   //if(Bars>300)return;
   if(CUR_BAR==0){
        ArraySetAsSeries(Close,false);
        ArrayResize(MA_Bars,Bars-MovingPeriod+1);
        ArrayResize(Var_Bars,Bars-MovingPeriod+1);
        for(i=0;i<MovingPeriod;i++){
            ma+=Close[i];
        }  
        ma/=MovingPeriod;
        ima=iMA(NULL,0,MovingPeriod,MovingShift,MODE_SMA,PRICE_CLOSE,MovingPeriod);
        //PrintFormat("cur_bar:0 ma:%f ima:%f",ma,ima);
        
        j=0;
        MA_Bars[j]=ma;
        Var_Bars[j]=MathAbs(Close[MovingPeriod-1]-ma);
        j++;
        for(i=MovingPeriod;i<Bars;i++,j++){
        
            ma=ma+(Close[i]-Close[i-MovingPeriod])/MovingPeriod;
            MA_Bars[j]=ma;
            Var_Bars[j]=MathAbs(Close[i]-ma);
            //ArraySetAsSeries(Close,true);
            //ima=iMA(NULL,0,MovingPeriod,MovingShift,MODE_SMA,PRICE_CLOSE,PeriodU-j);
            //ArraySetAsSeries(Close,false);
            //PrintFormat("cur_bar:%d j:%d ma:%f ima:%f var:%f",CUR_BAR,j,MA_Bars[j],ima,Var_Bars[j]);
        }
        ArraySetAsSeries(Close,true);
        CUR_BAR=Bars;  
   }
   else if(CUR_BAR<Bars){
        ArrayResize(MA_Bars,Bars-MovingPeriod+1);
        ArrayResize(Var_Bars,Bars-MovingPeriod+1);
        ArraySetAsSeries(Close,false);
        j=CUR_BAR-MovingPeriod;     
        ma=MA_Bars[j];
        //PrintFormat("cur_bar:%d j:%d ma:%f",CUR_BAR,j,ma);
        j++;
        //PrintFormat("arraysize:%d j:%d ma:%f",ArraySize(MA_Bars),j,ma);
        for(i=CUR_BAR;i<Bars;i++,j++){
            //PrintFormat("i:%d ma:%f close:%f diff:%f size:%d",i,ma,Close[i],Close[i]-Close[i-MovingPeriod],ArraySize(MA_Bars));
            ma=ma+(Close[i]-Close[i-MovingPeriod])/MovingPeriod;
            MA_Bars[j]=ma;
            Var_Bars[j]=MathAbs(Close[i]-ma);
            //ima=iMA(NULL,0,MovingPeriod,MovingShift,MODE_SMA,PRICE_CLOSE,0);
            //PrintFormat("cur_bar:%d j:%d ma:%f ima:%f var:%f",CUR_BAR,j,MA_Bars[j],ima,Var_Bars[j]);
            //PrintFormat("i:%d ma:%f var:%f close:%f bars:%d",i,MA_Bars[j],Var_Bars[j],Close[i],Bars);
        }
        //for(int k=0;k<ArraySize(MA_Bars);k++)
        //{
        //    PrintFormat("k:%d ma:%f v:%f",k,MA_Bars[k],Var_Bars[k]);
        //}
        ArraySetAsSeries(Close,true);
        CUR_BAR=Bars;
   }
//   else if(CUR_BAR==Bars){
//        ArraySetAsSeries(MA_Bars,true);
//        ArraySetAsSeries(Var_Bars,true);
//        MA_Bars[0]=MA_Bars[1]+(Close[0]-MA_Bars[0])/MovingPeriod;
//        Var_Bars[0]=MathAbs(MA_Bars[0]-Close[0]);
//        //PrintFormat("current ma:%f var:%f",MA_Bars[0],Var_Bars[0]);
//        ArraySetAsSeries(MA_Bars,false);
//        ArraySetAsSeries(Var_Bars,false);
//   }
//   
   if(CalculateCurrentOrders(Symbol())==0) CheckForOpen();
   else                                    CheckForClose();
  }

void CheckForOpen()
  {
   double price,u,v,maxv,b,ma,ma_pre,stoploss,takeprofit,open,close,ima;
   int    res;
   if(Volume[0]>1) return;
    u=GetU();
    v=GetV();  
    maxv=GetMaxV();
    ArraySetAsSeries(MA_Bars,true);
    ma=MA_Bars[0];
    ma_pre=MA_Bars[PeriodU-1];
    ArraySetAsSeries(MA_Bars,false);
    b=(Bid-ma)/Point;
    PrintFormat("checkforopen u:%f v:%f maxv:%f b:%f,Ask:%f Bid:%f ma:%f ma_pre:%f ",u,v,maxv,b,Ask,Bid,ma,ma_pre);
    
    if(-b>=v*vfactor3){
        stoploss=Bid-vfactor2*maxv*Point;
        //takeprofit=ma+v*Point;
        takeprofit=ma;
        res=OrderSend(Symbol(),OP_BUY,LotsOptimized(),Ask,3,stoploss,takeprofit,"",MAGICMA,0,Blue);
        op_u=u;op_v=v;
        Print("open buy because 1");
        PrintFormat("Ask:%f Bid:%f ma:%f %f u:%f v:%f b:%f op_u:%f op_v:%f",Ask,Bid,open,close,ma,ima,ma_pre,u,v,b,op_u,op_v);
        return;
    }
    if(b>=v*vfactor3){
        stoploss=Ask+vfactor2*maxv*Point;
        //takeprofit=ma-v*Point;
        takeprofit=ma;
        res=OrderSend(Symbol(),OP_SELL,LotsOptimized(),Bid,3,stoploss,takeprofit,"",MAGICMA,0,Red);
        op_u=u;op_v=v;
        Print("open sell because 1");
        PrintFormat("Ask:%f Bid:%f ma:%f  u:%f v:%f b:%f op_u:%f op_v:%f",Ask,Bid,open,close,ma,ima,ma_pre,u,v,b,op_u,op_v);
        return;
    }
  }
//+------------------------------------------------------------------+
//| Check for close order conditions                                 |
//+------------------------------------------------------------------+
void CheckForClose()
{
    double price,u,v,maxv,b,ma,ma_pre,stoploss,takeprofit,open,close,ima;
    int    res;
    if(Volume[0]>1) return;
    u=GetU();
    v=GetV();  
    maxv=GetMaxV();
    ArraySetAsSeries(MA_Bars,true);
    ma=MA_Bars[0];
    ma_pre=MA_Bars[PeriodU-1];
    ArraySetAsSeries(MA_Bars,false);
    b=(Bid-ma)/Point;
    PrintFormat("checkforopen u:%f v:%f maxv:%f b:%f,Ask:%f Bid:%f ma:%f ",u,v,maxv,b,Ask,Bid,ma,ma_pre);
    if(maxv==0)return;
    for(int i=0;i<OrdersTotal();i++)
    {
      if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES)==false) break;
      if(OrderType()==OP_BUY){
        if(b>v*vfactor1){
            if(!OrderClose(OrderTicket(),OrderLots(),Bid,3,White))
                Print("OrderClose error ",GetLastError());
            else{
                PrintFormat("Ask:%f Bid:%f ma:%f u:%f v:%f b:%f op_u:%f op_v:%f",Ask,Bid,ma,u,v,b,op_u,op_v);
                continue;
            }
         }
      }
      if(OrderType()==OP_SELL){
        if(-b>v*vfactor1){
            if(!OrderClose(OrderTicket(),OrderLots(),Ask,3,White))
                Print("OrderClose error ",GetLastError());
            else{
                PrintFormat("Ask:%f Bid:%f ma:%f u:%f v:%f b:%f op_u:%f op_v:%f",Ask,Bid,ma,u,v,b,op_u,op_v);
                continue;
             }
        }
      }
    }
}

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
input double MINU=0.12;
input double MAXU=7.05;
input double MAXV=286;
double MINV=20;
input double VFACTOR1=-0.166;
input double VFACTOR2=0.59;
input double VFACTOR3=4.6;
input int    MovingPeriod  =144;
input int    MovingShift   =0;
input int    PeriodU=144;
int    Loses=0;
input int    StopLoss=477;
input int    TakeProfit=200;
double MA_Bars[];
double Var_Bars[];


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
   //return Lots;
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
    //ArraySetAsSeries(MA_Bars,true);
    //ma_cur=MA_Bars[0];
    //ma_pre=MA_Bars[PeriodU-1];
    //ArraySetAsSeries(MA_Bars,false);
    ma_cur=iMA(NULL,0,MovingPeriod,MovingShift,MODE_SMA,PRICE_CLOSE,0);
    ma_pre=iMA(NULL,0,MovingPeriod,MovingShift,MODE_SMA,PRICE_CLOSE,PeriodU-1);
    //PrintFormat("ma_pre:%f ma_cur:%f",ma_pre,ma_cur);
    return (ma_cur-ma_pre)/Point/PeriodU;
}
double GetV(){
    double ma,v;
    //for(int j=0;j<ArraySize(Var_Bars);j++){
    //    PrintFormat("j:%d vb:%f",j,Var_Bars[j]);
    //}
    //ArraySetAsSeries(Var_Bars,true);
    //for(int i=0;i<PeriodU;i++)
    //{
    //    v+=Var_Bars[i];
    //    //PrintFormat("i:%d var:%f v:%f",i,Var_Bars[i],v);
    //}
    //ArraySetAsSeries(Var_Bars,false);
    for(int i=0;i<PeriodU;i++)
    {
        ma=iMA(NULL,0,MovingPeriod,MovingShift,MODE_SMA,PRICE_CLOSE,i);
        v+=MathAbs(Close[i]-ma);
    }
    v=v/Point/PeriodU;
    return v;
}
double op_u=0,op_v=0;
//+------------------------------------------------------------------+
//| Check for open order conditions                                  |
//+------------------------------------------------------------------+
void CheckForOpen()
  {
   double price,u,v,b,ma,ma_pre,stoploss,open,close,ima;
   int    res;
//--- go trading only for first tiks of new bar
   if(Volume[0]>1) return;
   //if(Loses<0)Loses=0;
   //if(Loses>=3){
   // PrintFormat("loses:%d",Loses);
   // //PeriodU+=17;
   //}
   //if(PeriodU>200)PeriodU=33;
//--- get Moving Average 
    u=GetU();
    v=GetV();  
    ArraySetAsSeries(MA_Bars,true);
    //ma=MA_Bars[0];
    ma=iMA(NULL,0,MovingPeriod,MovingShift,MODE_SMA,PRICE_CLOSE,0);
    ma_pre=MA_Bars[PeriodU-1];
    ArraySetAsSeries(MA_Bars,false);
    b=(Bid-ma)/Point;
    open=Close[2];
    close=Bid;
    //PrintFormat("checkforopen Ask:%f Bid:%f ma:%f ma_pre:%f u:%f v:%f b:%f",Ask,Bid,ma,ma_pre,u,v,b);
    //PrintFormat("open:%f close:%f ma:%f u:%f v:%f op_u:%f op_v:%f  b:%f v1:%f v2:%f v3:%f",open,close,ma,u,v,op_u,op_v,b,v*VFACTOR1,v*VFACTOR2,v*VFACTOR3);
    if(MathAbs(u)>MAXU||v>MAXV){
        PrintFormat("u:%f or v:%f is too big",u,v);
        return;
    }
    if(u>=MINU){
        if(b<v*VFACTOR1){
            stoploss=Bid-StopLoss*Point;
            res=OrderSend(Symbol(),OP_BUY,LotsOptimized(),Ask,3,stoploss,0,"",MAGICMA,0,Blue);
            op_u=u;op_v=v;
            Print("open buy because 1");
            PrintFormat("Ask:%f Bid:%f open:%f close:%f ma:%f ima:%f ma_pre:%f u:%f v:%f b:%f op_u:%f op_v:%f",Ask,Bid,open,close,ma,ima,ma_pre,u,v,b,op_u,op_v);
            return;
        }
    }
    if(u<=-MINU){
        if(-b<v*VFACTOR1){
            stoploss=Ask+StopLoss*Point;
            res=OrderSend(Symbol(),OP_SELL,LotsOptimized(),Bid,3,stoploss,0,"",MAGICMA,0,Red);
            op_u=u;op_v=v;
            Print("open sell because 1");
            PrintFormat("Ask:%f Bid:%f open:%f close:%f ma:%f ima:%f ma_pre:%f u:%f v:%f b:%f op_u:%f op_v:%f",Ask,Bid,open,close,ma,ima,ma_pre,u,v,b,op_u,op_v);
            return;
        }
    }
    if(MathAbs(u)<MINU){
        if(b>v*VFACTOR2){
            stoploss=Ask+StopLoss*Point;
            res=OrderSend(Symbol(),OP_SELL,LotsOptimized(),Bid,3,stoploss,0,"",MAGICMA,0,Red);
            op_u=u;op_v=v;
            Print("open sell because 2");
            PrintFormat("Ask:%f Bid:%f open:%f close:%f ma:%f ima:%f ma_pre:%f u:%f v:%f b:%f op_u:%f op_v:%f",Ask,Bid,open,close,ma,ima,ma_pre,u,v,b,op_u,op_v);
            return;
        }
        if(-b>v*VFACTOR2){
            stoploss=Bid-StopLoss*Point;
            res=OrderSend(Symbol(),OP_BUY,LotsOptimized(),Ask,3,stoploss,0,"",MAGICMA,0,Blue);
            op_u=u;op_v=v;
            Print("open buy because 2");
            PrintFormat("Ask:%f Bid:%f open:%f close:%f ma:%f ima:%f ma_pre:%f u:%f v:%f b:%f op_u:%f op_v:%f",Ask,Bid,open,close,ma,ima,ma_pre,u,v,b,op_u,op_v);
            return;
        }
    }
  }
//+------------------------------------------------------------------+
//| Check for close order conditions                                 |
//+------------------------------------------------------------------+
void CheckForClose()
  {
   double ma,ma_pre,u,v,b,stoploss,open,close,ima;
   string date=TimeToStr(Time[0],TIME_DATE);
   string minute=TimeToStr(Time[0],TIME_MINUTES);
   //Print("minute",minute);
//--- go trading only for first tiks of new bar
   if(Volume[0]>1) return;
//--- get Moving Average 
    u=GetU();
    v=GetV();
    ArraySetAsSeries(MA_Bars,true);
    //ma=MA_Bars[0];
    ma=iMA(NULL,0,MovingPeriod,MovingShift,MODE_SMA,PRICE_CLOSE,0);
    ma_pre=MA_Bars[PeriodU-1];
    ArraySetAsSeries(MA_Bars,false);
    b=(Bid-ma)/Point;
    open=Close[2];
    close=Bid;
    //PrintFormat("checkforclose Ask:%f Bid:%f ma:%f ma_pre:%f u:%f v:%f b:%f",Ask,Bid,ma,ma_pre,u,v,b);
    //PrintFormat("open:%f close:%f ma:%f u:%f v:%f op_u:%f op_v:%f  b:%f v1:%f v2:%f v3:%f",open,close,ma,u,v,op_u,op_v,b,v*VFACTOR1,v*VFACTOR2,v*VFACTOR3);

    
    //if(minute=="12:35"||minute=="12:37")PrintFormat("minute:%s u:%f minu:%f com:%d",minute,u/Point,MINU/Point,u<-MINU);
    //PrintFormat("Ask:%f Bid:%f ma:%f u:%f v:%f b:%f op_u:%f op_v:%f",Ask,Bid,ma,u,v,b,op_u,op_v);
   for(int i=0;i<OrdersTotal();i++)
     {
      if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES)==false) break;
      if(OrderMagicNumber()!=MAGICMA || OrderSymbol()!=Symbol()) continue;
      //--- check order type 
      if(OrderType()==OP_BUY)
        {
         //if(Bid-OrderOpenPrice()>=30){
         //   stoploss=OrderOpenPrice()+10;
         //   if(OrderModify(OrderTicket(),OrderOpenPrice(),stoploss,OrderTakeProfit(),0,Blue)==false){
         //       PrintFormat("modify order#%d error:%d",OrderTicket(),GetLastError());
         //   }
         //}
         if(op_u>MINU&&MathAbs(u)>=MINU&&u<=0)
           {
                if(!OrderClose(OrderTicket(),OrderLots(),Bid,3,White))
                   Print("OrderClose error ",GetLastError());
                else{
                    Print("close buy because 1");
                    PrintFormat("Ask:%f Bid:%f open:%f close:%f ma:%f ima:%f ma_pre:%f u:%f v:%f b:%f op_u:%f op_v:%f",Ask,Bid,open,close,ma,ima,ma_pre,u,v,b,op_u,op_v);
                    if(OrderProfit()<0)Loses++; 
                    op_u=0;op_v=0;
                    continue;
                }
            
           }
           //if(op_u>MINU&&open>=ma&&close<=ma)
           //{
           //     if(!OrderClose(OrderTicket(),OrderLots(),Bid,3,White))
           //        Print("OrderClose error ",GetLastError());
           //     else{
           //         Print("close buy because 4");
           //         PrintFormat("Ask:%f Bid:%f open:%f close:%f ma:%f ima:%f ma_pre:%f u:%f v:%f b:%f op_u:%f op_v:%f",Ask,Bid,open,close,ma,ima,ma_pre,u,v,b,op_u,op_v);
           //         if(OrderProfit()<0)Loses++; 
           //         op_u=0;op_v=0;
           //         continue;
           //     }         
           //}
          if(op_u>MINU&&b>v*VFACTOR3){
            if(!OrderClose(OrderTicket(),OrderLots(),Bid,3,White))
               Print("OrderClose error ",GetLastError());
            else{
                Print("close buy because 2");
                PrintFormat("Ask:%f Bid:%f open:%f close:%f ma:%f ima:%f ma_pre:%f u:%f v:%f b:%f op_u:%f op_v:%f",Ask,Bid,open,close,ma,ima,ma_pre,u,v,b,op_u,op_v);
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
                    PrintFormat("Ask:%f Bid:%f open:%f close:%f ma:%f ima:%f ma_pre:%f u:%f v:%f b:%f op_u:%f op_v:%f",Ask,Bid,open,close,ma,ima,ma_pre,u,v,b,op_u,op_v);
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
                    PrintFormat("Ask:%f Bid:%f open:%f close:%f ma:%f ima:%f ma_pre:%f u:%f v:%f b:%f op_u:%f op_v:%f",Ask,Bid,open,close,ma,ima,ma_pre,u,v,b,op_u,op_v);
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
         //if(OrderOpenPrice()-Ask>=30){
         //   stoploss=OrderOpenPrice()-10;
         //   if(OrderModify(OrderTicket(),OrderOpenPrice(),stoploss,OrderTakeProfit(),0,Blue)==false){
         //       PrintFormat("modify order#%d error:%d",OrderTicket(),GetLastError());
         //   }
         //}
         if(op_u<-MINU&&MathAbs(u)>MINU&&u>=0)
           {
            if(!OrderClose(OrderTicket(),OrderLots(),Ask,3,White))
               Print("OrderClose error ",GetLastError());
            else{
                Print("close sell because 1");
                PrintFormat("Ask:%f Bid:%f open:%f close:%f ma:%f ima:%f ma_pre:%f u:%f v:%f b:%f op_u:%f op_v:%f",Ask,Bid,open,close,ma,ima,ma_pre,u,v,b,op_u,op_v);
                if(OrderProfit()<0)Loses++; 
                op_u=0;op_v=0;
                continue;
            }
           }
           //if(op_u<-MINU&&open<=ma&&close>=ma)
           //{
           // if(!OrderClose(OrderTicket(),OrderLots(),Ask,3,White))
           //    Print("OrderClose error ",GetLastError());
           // else{
           //     Print("close sell because 4");
           //     PrintFormat("Ask:%f Bid:%f open:%f close:%f ma:%f ima:%f ma_pre:%f u:%f v:%f b:%f op_u:%f op_v:%f",Ask,Bid,open,close,ma,ima,ma_pre,u,v,b,op_u,op_v);
           //     if(OrderProfit()<0)Loses++; 
           //     op_u=0;op_v=0;
           //     continue;
           // }
           //}
         if(op_u<-MINU&&-b>v*VFACTOR3){
            if(!OrderClose(OrderTicket(),OrderLots(),Ask,3,White))
               Print("OrderClose error ",GetLastError());
            else{
                Print("close sell because 2");
                PrintFormat("Ask:%f Bid:%f open:%f close:%f ma:%f ima:%f ma_pre:%f u:%f v:%f b:%f op_u:%f op_v:%f",Ask,Bid,open,close,ma,ima,ma_pre,u,v,b,op_u,op_v);
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
                    PrintFormat("Ask:%f Bid:%f open:%f close:%f ma:%f ima:%f ma_pre:%f u:%f v:%f b:%f op_u:%f op_v:%f",Ask,Bid,open,close,ma,ima,ma_pre,u,v,b,op_u,op_v);
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
                    PrintFormat("Ask:%f Bid:%f open:%f close:%f ma:%f ima:%f ma_pre:%f u:%f v:%f b:%f op_u:%f op_v:%f",Ask,Bid,open,close,ma,ima,ma_pre,u,v,b,op_u,op_v);
                    if(OrderProfit()<0)Loses++; 
                    op_u=0;op_v=0;
                    continue;
                }
            }
         }
         break;
        }
     }
  }

//void OnInit()
//{
//    
//}

int CUR_BAR=0;
void OnTick()
  {
    int i=0,j=0,size;
    double ma=0;
   if(Bars<MovingPeriod+PeriodU || IsTradeAllowed()==false)
      return;
   if(CUR_BAR==0){
        ArraySetAsSeries(Close,false);
        ArrayResize(MA_Bars,Bars-MovingPeriod);
        ArrayResize(Var_Bars,Bars-MovingPeriod);
        for(i=0;i<MovingPeriod;i++){
            ma+=Close[i];
        }
        ma/=MovingPeriod;
        j=0;
        MA_Bars[j]=ma;
        Var_Bars[j]=MathAbs(Close[MovingPeriod-1]-ma);
        j++;
        for(i=MovingPeriod;i<Bars;i++,j++){
            ma=ma+(Close[i]-Close[i-MovingPeriod])/MovingPeriod;
            MA_Bars[j]=ma;
            Var_Bars[j]=MathAbs(Close[i]-ma);
            //PrintFormat("j:%d ma:%f var:%f",j,MA_Bars[j],Var_Bars[j]);
        }
        ArraySetAsSeries(Close,true);
        CUR_BAR=Bars;  
   }
   else if(CUR_BAR<Bars){
        if(ArraySize(MA_Bars)<Bars-MovingPeriod||ArraySize(Var_Bars)<Bars-MovingPeriod){
            ArrayResize(MA_Bars,Bars-MovingPeriod);
            ArrayResize(Var_Bars,Bars-MovingPeriod);
        }
        ArraySetAsSeries(Close,false);
        j=CUR_BAR-MovingPeriod-1;     
        ma=MA_Bars[j];
        
        //PrintFormat("arraysize:%d j:%d ma:%f",ArraySize(MA_Bars),j,ma);
        for(i=CUR_BAR-1;i<Bars;i++,j++){
            ma=ma+(Close[i]-Close[i-MovingPeriod])/MovingPeriod;
            MA_Bars[j]=ma;
            Var_Bars[j]=MathAbs(Close[i]-ma);
            //PrintFormat("i:%d ma:%f var:%f close:%f bars:%d",i,MA_Bars[j],Var_Bars[j],Close[i],Bars);
        }
        //for(int k=0;k<ArraySize(MA_Bars);k++)
        //{
        //    PrintFormat("k:%d ma:%f v:%f",k,MA_Bars[k],Var_Bars[k]);
        //}
        ArraySetAsSeries(Close,true);
        CUR_BAR=Bars;
   }
   else if(CUR_BAR==Bars){
        ArraySetAsSeries(MA_Bars,true);
        ArraySetAsSeries(Var_Bars,true);
        MA_Bars[0]=MA_Bars[1]+(Close[0]-MA_Bars[0])/MovingPeriod;
        Var_Bars[0]=MathAbs(MA_Bars[0]-Close[0]);
        //PrintFormat("current ma:%f var:%f",MA_Bars[0],Var_Bars[0]);
        ArraySetAsSeries(MA_Bars,false);
        ArraySetAsSeries(Var_Bars,false);
   }
   
   if(CalculateCurrentOrders(Symbol())==0) CheckForOpen();
   else                                    CheckForClose();
  }

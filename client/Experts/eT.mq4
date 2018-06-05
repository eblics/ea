//+------------------------------------------------------------------+
//|                                               Moving Average.mq4 |
//|                   Copyright 2005-2014, MetaQuotes Software Corp. |
//|                                              http://www.mql4.com |
//+------------------------------------------------------------------+
#property copyright   "2005-2014, MetaQuotes Software Corp."
#property link        "http://www.mql4.com"
#property description "Moving Average sample expert advisor"

#define MAGICMA  19820211
//--- Inputs
input double Lots          =0.1;
input double MaximumRisk   =0.1;
input double DecreaseFactor=9;
input int    Period=144;
input int    Gap=36;
input int    Losses=2;
input double Factor=2.1;
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
    double lot=Lots,lot1;
    int    orders=HistoryTotal();     // history orders total
    int    losses=0;                  // number of losses orders without a break
    lot=NormalizeDouble(AccountFreeMargin()*MaximumRisk/1000.0,2);
    if(DecreaseFactor!=0)
     {
      for(int i=orders-1;i>=0;i--)
        {
         if(OrderSelect(i,SELECT_BY_POS,MODE_HISTORY)==false)
           {
            Print("Error in history!");
            break;
           }
           
         if(OrderSymbol()!=Symbol() ||OrderMagicNumber()!=MAGICMA|| OrderType()>OP_SELL)
            continue;
         //---
         if(OrderProfit()>0) break;
         if(OrderProfit()<0) losses++;
         
        }
      if(losses>=1){
         lot1=lot;
         lot=NormalizeDouble(lot-lot*losses/DecreaseFactor,2);
      }
     }
    if(lot<0.01) lot=0.01;
    return(lot);
}

double CountLosses()
{
    int    orders=HistoryTotal();     // history orders total
    int    losses=0;                  // number of losses orders without a break
    for(int i=orders-1;i>=0;i--)
    {
     if(OrderSelect(i,SELECT_BY_POS,MODE_HISTORY)==false)
       {
        Print("Error in history!");
        break;
       }
       
     if(OrderSymbol()!=Symbol() ||OrderMagicNumber()!=MAGICMA|| OrderType()>OP_SELL)
        continue;
     if(OrderProfit()>0) break;
     if(OrderProfit()<0) losses++;
     
    }
    return losses;
}


 
double GetPQ(){
    double win,lose;
    for(int i=0;i<Period;i++)
    {
        if(Close[i]>Open[i]){win +=Close[i]-Open[i];}
        if(Close[i]<Open[i]){lose+=Open[i]-Close[i];}
    }
    return win/lose;
}
//+------------------------------------------------------------------+
//| Check for open order conditions                                  |
//+------------------------------------------------------------------+


int CUR_BAR=0;
void OnTick()
{
    if(CalculateCurrentOrders(Symbol())==0) CheckForOpen();
    else                                    CheckForClose();
}

void CheckForOpen()
{
    double v,abs,max,dir,losses,price,stoploss,takeprofit,open,close;
    int    res;
    if(Ask-Bid>3)return;
    for(int i=0;i<Period;i++)
    {
        v=Close[i]-Open[i];
        abs=MathAbs(v);
        if(abs>max) max=v;
    }
    losses=CountLosses();
    PrintFormat("checkforopen losses:%f max:%f",losses,max);
    if(losses<Losses&&max>0){
        stoploss=Bid-Factor*max;
        takeprofit=Bid+Factor*max;
        res=OrderSend(Symbol(),OP_BUY,LotsOptimized(),Ask,3,stoploss,takeprofit,"",MAGICMA,0,Blue);
        if(res==-1){
            PrintFormat("lots:%f price:%f stoploss:%f takeprofit:%f",LotsOptimized(),Bid,stoploss,takeprofit);
            //ExpertRemove();
        }
        return;
    }
    if(losses<Losses&&max<0){
        stoploss=Ask+Factor*max; 
        takeprofit=Ask-Factor*max;
        res=OrderSend(Symbol(),OP_SELL,LotsOptimized(),Bid,3,stoploss,takeprofit,"",MAGICMA,0,Red);
        if(res==-1){
            PrintFormat("lots:%f price:%f stoploss:%f takeprofit:%f",LotsOptimized(),price,stoploss,takeprofit);
            //ExpertRemove();
        }
        return;
    }
    if(losses>=Losses){
        int orders=HistoryTotal();
        if(OrderSelect(orders-1,SELECT_BY_POS,MODE_HISTORY)==false)
        {
            Print("Error in history!");
            return;
        }
        if(OrderSymbol()!=Symbol() ||OrderMagicNumber()!=MAGICMA|| OrderType()>OP_SELL)
            return;
        if(OrderType()==OP_SELL){
            stoploss=Ask-Factor*max;
            takeprofit=Bid+Factor*max;
            res=OrderSend(Symbol(),OP_BUY,LotsOptimized(),Ask,3,stoploss,takeprofit,"",MAGICMA,0,Blue); 
        }
        if(OrderType()==OP_BUY){
            stoploss=Ask+Factor*max;
            takeprofit=Ask-Factor*max;
            res=OrderSend(Symbol(),OP_SELL,LotsOptimized(),Ask,3,stoploss,takeprofit,"",MAGICMA,0,Blue); 
        }
    }
}
//+------------------------------------------------------------------+
//| Check for close order conditions                                 |
//+------------------------------------------------------------------+
void CheckForClose()
{
    double wint,loset,win,lose,p,q,price,stoploss,takeprofit,open,close,diff;
    int    res;
    int i;
    for(i=0;i<Period;i++)
    {
        if(Close[i]>Open[i]){wint+=1; win +=Close[i]-Open[i];}
        if(Close[i]<Open[i]){loset+=1;lose+=Open[i]-Close[i];}
    }
    p=wint/loset;
    q=win/lose;
    PrintFormat("checkforclose p:%f q:%f win:%f lose:%f",p,q,win,lose);
    for(i=0;i<OrdersTotal();i++)
    {
      if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES)==false) break;
      if(OrderMagicNumber()!=MAGICMA || OrderSymbol()!=Symbol()) continue;
      if(OrderType()==OP_BUY){
        stoploss=Bid-Gap*Point;
        if(stoploss>OrderOpenPrice()){
            stoploss=stoploss+Gap/2*Point;
            takeprofit=OrderTakeProfit();//+Gap/2*Point;
            if(stoploss>OrderStopLoss())
                OrderModify(OrderTicket(),OrderOpenPrice(),stoploss,takeprofit,NULL,Blue);
        }
        //if(p>=PQ){
        //    if(!OrderClose(OrderTicket(),OrderLots(),Bid,3,White))
        //        Print("OrderClose error ",GetLastError());
        //    else{
        //        continue;
        //    }
        // }
      }
      if(OrderType()==OP_SELL){
        stoploss=Ask+Gap*Point;
        if(stoploss<OrderOpenPrice()){
            stoploss=stoploss-Gap/2*Point;
            takeprofit=OrderTakeProfit();//-Gap/2*Point;
            if(stoploss<OrderStopLoss())
                OrderModify(OrderTicket(),OrderOpenPrice(),stoploss,takeprofit,NULL,Blue);
        }
        //if(p<=1/PQ){
        //    if(!OrderClose(OrderTicket(),OrderLots(),Ask,3,White))
        //        Print("OrderClose error ",GetLastError());
        //    else{
        //        continue;
        //     }
        //}
      }
    }
}

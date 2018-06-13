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
input double MaximumRisk   =0.01;
input double DecreaseFactor=3;
input double PQ=1.232;
input int    Period=144;
input int    Gap=36;
input double GapFactor=1.8;
input double    Rate=0.36;
input int    HisPeriod=4;
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
    double    wins=0,losses=0;                  // number of losses orders without a break
    lot=NormalizeDouble(AccountFreeMargin()*MaximumRisk/1000.0,2);
    if(DecreaseFactor!=0)
     {
      for(int i=orders-1;i>=0&&i>orders-HisPeriod;i--)
        {
         if(OrderSelect(i,SELECT_BY_POS,MODE_HISTORY)==false)
           {
            Print("Error in history!");
            break;
           }
           
         if(OrderSymbol()!=Symbol() ||OrderMagicNumber()!=MAGICMA|| OrderType()>OP_SELL)
            continue;
         //---
         if(OrderProfit()>0) wins+=OrderProfit();
         if(OrderProfit()<0) losses+=-OrderProfit();
         
        }
      if(losses!=0){
         //lot=NormalizeDouble(lot-lot*losses/DecreaseFactor,2);
         lot=lot*wins/(wins+losses);
      }
     }
    if(lot<0.1) lot=0.1;
    return(lot);
}

double CountSuccRate()
{
    int    orders=HistoryTotal();     // history orders total
    double    win=0;
    double    losses=0;                  // number of losses orders without a break
    for(int i=orders-1;i>0&&i>=orders-HisPeriod;i--)
    {
     if(OrderSelect(i,SELECT_BY_POS,MODE_HISTORY)==false)
       {
        Print("Error in history!");
        break;
       }
       
     if(OrderSymbol()!=Symbol() ||OrderMagicNumber()!=MAGICMA|| OrderType()>OP_SELL)
        continue;
     if(OrderProfit()>0) win+=OrderProfit();
     if(OrderProfit()<0) losses+=-OrderProfit();
     
    }
    if(losses!=0)
        return win/losses;
    return 1;
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
    double wint,loset,win,lose,rate,p,q,g,price,stoploss,takeprofit,open,close;
    int    res;
    if(Ask-Bid>3)return;
    for(int i=0;i<Period;i++)
    {
        if(Close[i]>Open[i]){wint+=1; win +=Close[i]-Open[i];}
        if(Close[i]<Open[i]){loset+=1;lose+=Open[i]-Close[i];}
    }
    p=wint/loset;
    q=win/lose;
    rate=CountSuccRate();
    PrintFormat("checkforopen p:%f q:%f win:%f lose:%f rate:%f",p,q,win,lose,rate);
    if(rate>=Rate&&p<1/PQ){
        stoploss=Bid-lose;
        takeprofit=Bid+win;
        res=OrderSend(Symbol(),OP_BUY,LotsOptimized(),Ask,3,stoploss,takeprofit,"",MAGICMA,0,Blue);
        if(res==-1){
            PrintFormat("lots:%f price:%f stoploss:%f takeprofit:%f",LotsOptimized(),Bid,stoploss,takeprofit);
            //ExpertRemove();
        }
        PrintFormat("openorder p:%f q:%f win:%f lose:%f rate:%f",p,q,win,lose,rate);
        return;
    }
    if(rate>=Rate&&p>PQ){
        stoploss=Ask+win; 
        takeprofit=Ask-lose;
        res=OrderSend(Symbol(),OP_SELL,LotsOptimized(),Bid,3,stoploss,takeprofit,"",MAGICMA,0,Red);
        if(res==-1){
            PrintFormat("lots:%f price:%f stoploss:%f takeprofit:%f",LotsOptimized(),price,stoploss,takeprofit);
            //ExpertRemove();
        }
        PrintFormat("openorder p:%f q:%f win:%f lose:%f rate:%f",p,q,win,lose,rate);
        return;
    }
    if(rate<Rate){
        int orders=HistoryTotal();
        if(OrderSelect(orders-1,SELECT_BY_POS,MODE_HISTORY)==false)
        {
            Print("Error in history!");
            return;
        }
        if(OrderSymbol()!=Symbol() ||OrderMagicNumber()!=MAGICMA|| OrderType()>OP_SELL)
            return;
        if(OrderType()==OP_SELL){
            stoploss=OrderTakeProfit();
            takeprofit=Bid+win;
            res=OrderSend(Symbol(),OP_BUY,LotsOptimized(),Ask,3,stoploss,takeprofit,"",MAGICMA,0,Blue); 
        }
        if(OrderType()==OP_BUY){
            stoploss=OrderTakeProfit();
            takeprofit=Ask-lose;
            res=OrderSend(Symbol(),OP_SELL,LotsOptimized(),Ask,3,stoploss,takeprofit,"",MAGICMA,0,Blue); 
        }
        PrintFormat("openorder p:%f q:%f win:%f lose:%f rate:%f",p,q,win,lose,rate);
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
            stoploss=stoploss+Gap/3*Point;
            takeprofit=OrderTakeProfit()+Gap/GapFactor*Point;
            if(stoploss>OrderStopLoss())
                OrderModify(OrderTicket(),OrderOpenPrice(),stoploss,takeprofit,NULL,Blue);
        }
        if(p>=PQ){
            if(!OrderClose(OrderTicket(),OrderLots(),Bid,3,White))
                Print("OrderClose error ",GetLastError());
            else{
                continue;
            }
         }
      }
      if(OrderType()==OP_SELL){
        stoploss=Ask+Gap*Point;
        if(stoploss<OrderOpenPrice()){
            stoploss=stoploss-Gap/GapFactor*Point;
            takeprofit=OrderTakeProfit()-Gap/GapFactor*Point;
            if(stoploss<OrderStopLoss())
                OrderModify(OrderTicket(),OrderOpenPrice(),stoploss,takeprofit,NULL,Blue);
        }
        if(p<=1/PQ){
            if(!OrderClose(OrderTicket(),OrderLots(),Ask,3,White))
                Print("OrderClose error ",GetLastError());
            else{
                continue;
             }
        }
      }
    }
}

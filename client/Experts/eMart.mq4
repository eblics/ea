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
input double GapFactor=1.1;
input double LotsStep=0.1;
input int LotsLimit=10;
input int SlipPage=3;
input int StopLoss=100;
input int TakeProfit=50;
input int Gap=300;
input int StopLine=4000;
input int DIRET=OP_SELL;

//int cnt=0;
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
   return buys-sells;
  }
  double GetIV()
  {
    return iCustom(Symbol(),0,"iV",5,33,0,0);
  }
  
  void OpenOrder(int op,double lots,int takeprofit,string notes)
  {
    //Print(notes);
    int res=0;
    //if(iv2==0x7FFFFFFF)
    if(op==OP_SELL)
     {
      res=OrderSend(Symbol(),OP_SELL,lots,Bid,SlipPage,Ask+StopLoss*Point,Ask-takeprofit*Point,"",MAGICMA,0,Red);
      if(res!=0){
        Print(GetLastError());
        PrintFormat("$$lots:%f",lots);
      }
      return;
     }
//--- buy conditions
   if(op==OP_BUY)
     {
      res=OrderSend(Symbol(),OP_BUY,lots,Ask,SlipPage,Bid-StopLoss*Point,Bid+takeprofit*Point,"",MAGICMA,0,Blue);
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
        
      if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES)==false){
        Print(GetLastError());
        break;
      }
      Print("try to close "+OrderTicket()); 
      if(OrderType()==OP_BUY)
        if(OrderClose(OrderTicket(),OrderLots(),Bid,SlipPage,Red)==false){
            Print(GetLastError());
        }
      else if(OrderType()==OP_SELL)
        Print("try to close2");
        if(OrderClose(OrderTicket(),OrderLots(),Ask,SlipPage,Red)==false){
             Print(GetLastError());
        }
     }
  }
  
  int SelectMaxOrder(int op){
     int ticket=-1;
     double openprice=0,price=0;
     for(int i=0;i<OrdersTotal();i++)
     {
      if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES)==false) break;
      if(OrderSymbol()==Symbol() && OrderMagicNumber()==MAGICMA && OrderType()==op)
        {
         if(op==OP_BUY){
            openprice=OrderOpenPrice();
            if(price==0) price=openprice;
            if(openprice<price){
                price=openprice;
                ticket=OrderTicket();
            }
         }
         if(op==OP_SELL){
            openprice=OrderOpenPrice();
            if(price==0) price=openprice;
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

//+------------------------------------------------------------------+
//| Check for open order conditions                                  |
//+------------------------------------------------------------------+
void CheckOrders(int cnt)
{
   double iv=0;
   double profit=0,price=0,cur=0;
   double gapfactor=0,lots=0;
   int gap=0;
   int res=0;
   int ticket=-1;
   
   
   //检查是否下过单（互相对冲为0也认为没有下单）
    if(cnt==0){
       OpenOrder(DIRET,Lots,TakeProfit,"$$order at 0");
       return;
    }
    profit=AccountProfit();
    //如果已经获利，平掉所有单子，重新开始
    if(profit>Lots*TakeProfit){
        CloseAllOrders();
        return;
    }
    if(MathAbs(cnt)>LotsLimit){
        return;
    }
    SelectMaxOrder(DIRET);
    price=OrderOpenPrice();
    if(DIRET==OP_SELL){
        gap=(int)((Ask-price)/Point);
    }
    if(DIRET==OP_BUY)
    {   
        gap=(int)((price-Bid)/Point);
    }
    gapfactor=MathPow(GapFactor,MathAbs(cnt));
    //lotsfactor=MathPow(LotsFactor,MathAbs(cnt));
    //lots=Lots+LotsStep*MathAbs(cnt)/4;
    //Print(lotsfactor);
    //Print("gap "+gap);
    if(gap>Gap*gapfactor) 
        OpenOrder(DIRET,Lots,MathAbs(cnt)*Gap+TakeProfit,"$$order gap");   
}
  
//+------------------------------------------------------------------+
//| Check for close order conditions                                 |
//+------------------------------------------------------------------+
//void CheckForClose()
//  {
//   double ma;
////--- go trading only for first tiks of new bar
//   if(Volume[0]>1) return;
////--- get Moving Average 
//   ma=iMA(NULL,0,MovingPeriod,MovingShift,MODE_SMA,PRICE_CLOSE,0);
////---
//   for(int i=0;i<OrdersTotal();i++)
//     {
//      if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES)==false) break;
//      if(OrderMagicNumber()!=MAGICMA || OrderSymbol()!=Symbol()) continue;
//      //--- check order type 
//      if(OrderType()==OP_BUY)
//        {
//         if(Open[1]>ma && Close[1]<ma)
//           {
//            if(!OrderClose(OrderTicket(),OrderLots(),Bid,3,White))
//               Print("OrderClose error ",GetLastError());
//           }
//         break;
//        }
//      if(OrderType()==OP_SELL)
//        {
//         if(Open[1]<ma && Close[1]>ma)
//           {
//            if(!OrderClose(OrderTicket(),OrderLots(),Ask,3,White))
//               Print("OrderClose error ",GetLastError());
//           }
//         break;
//        }
//     }
////---
//  }
  
  
///+-----------------------Own Logic----------------------------+/
//
//double iV()
//{
//    double u=0,v=0;
//    for(int i=0;i<PeriodU;i++){
//        u+=(Close[i]-Open[i])*(Close[i]-Open[i]);
//    }
//    u=sqrt(u/PeriodU);
//    v=(Close[0]-Close[PeriodV])/PeriodV;
//    return v/u;
//}



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
   CheckOrders(cnt);
   if(cbar<Bars){
    cbar=Bars;
    //Print("##ticket iv:"+string(GetIV()));
   }
}

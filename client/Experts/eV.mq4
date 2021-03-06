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
input int LotsLimit=10;
input double Level   =0.0015;
input int PeriodU=5;
input int PeriodV=33;
input int SlipPage=3;
input int StopLoss=100;
input int TakeProfit=50;
input int Gap=300;

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
    return iCustom(Symbol(),0,"ea/iV",5,33,0,0);
}

void OpenOrder(int op,string notes)
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
    double profit=0,price=0;
    int gap=0;
    int res=0;
    int ticket=-1;

    profit=AccountProfit();
    //如果已经获利，平掉所有单子，重新开始
    if(profit>Lots*TakeProfit){
        CloseAllOrders();
        PrintFormat("$$close with profit:%f",profit);
        return;
    }
    iv=GetIV();
    if(iv==0x7FFFFFFF)
        return;
    if(-iv>Level){
        if(cnt>LotsLimit){
            return;
        }
        SelectMaxOrder(OP_BUY);
        price=OrderOpenPrice();
        gap=(int)((price-Ask)/Point);
        if(gap>Gap)
            OpenOrder(OP_BUY,"buy 0 cnt:"+string(cnt)+" iv:"+string(iv));
        //PrintFormat("gap:%f cnt:%d",gap,cnt);
    }
    if(iv>Level){
        if(-cnt>LotsLimit){
            return;
        }
        SelectMaxOrder(OP_SELL);
        price=OrderOpenPrice();
        gap=(int)((Bid-price)/Point);
        if(gap>Gap)
            OpenOrder(OP_SELL,"sell 0 cnt:"+string(cnt)+" iv:"+string(iv));
        //PrintFormat("gap:%f cnt:%d",gap,cnt);
    }
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

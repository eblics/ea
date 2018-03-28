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
input double Level   =0.96;
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
    return iCustom(Symbol(),0,"iV",5,33,0,0);
}

void OpenOrder(int op,string notes)
{
    Print(notes);
    int res=0;
    //if(iv2==0x7FFFFFFF)
    if(op==OP_SELL)
    {
        res=OrderSend(Symbol(),OP_SELL,Lots,Bid,SlipPage,Ask+StopLoss*Point,Ask-TakeProfit*Point,"",MAGICMA,0,Red);
        //Print("=====================:"+string(iv)+"  iv2:"+string(iv2));
        return;
    }
    //--- buy conditions
    if(op==OP_BUY)
    {
        res=OrderSend(Symbol(),OP_BUY,Lots,Ask,SlipPage,Bid-StopLoss*Point,Bid+TakeProfit*Point,"",MAGICMA,0,Blue);
        //Print("=====================:"+string(iv)+"  iv2:"+string(iv2));
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
            if(!OrderClose(OrderTicket(),OrderLots(),Bid,SlipPage,Red)){
                CloseAllOrders();
            }
            else if(OrderType()==OP_SELL)
                if(!OrderClose(OrderTicket(),OrderLots(),Ask,SlipPage,Red)){
                    CloseAllOrders();
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
    //检查是否下过单（互相对冲为0也认为没有下单）
    if(cnt==0){
        iv=GetIV();
        if(iv==0x7FFFFFFF)
            return;
        if(-iv>Level){
            OpenOrder(OP_BUY,"buy 0 cnt:"+string(cnt)+" iv:"+string(iv));
        }
        if(iv>Level){
            OpenOrder(OP_SELL,"sell 0 cnt:"+string(cnt)+" iv:"+string(iv));
        }
    }
    else{
        //如果已经获利，平掉所有单子，重新开始
        if(profit>Lots*TakeProfit){
            CloseAllOrders();
            return;
        }
        if(MathAbs(cnt)>LotsLimit){
            return;
        }
        //如果买方向已经建单，且没有超过手数限制
        if(cnt>0){
            SelectMaxOrder(OP_BUY);
            price=OrderOpenPrice();
            iv=GetIV();
            if(iv==0x7FFFFFFF)
                return;
            //看是否是下单机会，如果是，且离上次同方向建单也一定距离，则可继续下单
            if(-iv>Level){
                gap=(int)(price-Ask)/Point;
                if(gap>Gap){
                    OpenOrder(OP_BUY,"buy gap");
                }
            }
            if(iv>Level){
                OpenOrder(OP_SELL,"sell hedge");
            }
        }
        else if(cnt<0){
            SelectMaxOrder(OP_SELL);
            price=OrderOpenPrice();
            iv=GetIV();
            if(iv==0x7FFFFFFF)
                return;
            if(iv>Level){
                gap=(int)(Bid-price)/Point;
                if(gap>Gap){
                    OpenOrder(OP_SELL,"sell gap");
                }
            }
            if(-iv>Level){
                OpenOrder(OP_BUY,"buy hedge");
            }
        }
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

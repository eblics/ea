#property copyright   "eblics"
#property link        "http://www.eblics.com"
#property description "mistake"

input double Lots=0.5;
input int LOSS=300;
input int PROFIT=300;
input int GAP=20;
input int SLIPPAGE=3;
input double Delta=0.618;

int dir=0;
double price,stoploss,takeprofit,profit,equity;
void OnTick()
{
   int ticket,orderType;
   
   
   if(OrdersTotal()==0){
      if(dir==0){      
         ticket=buy();
         dir=1;
         equity=AccountEquity();
      }
      else if(dir==1)
      {
         
         if(AccountEquity()>equity){
            ticket=buy_limit();
            dir=1;
         }
         else{
            ticket=sell();
            dir=-1;
         }
         //PrintFormat("dir:%d equity:%f fequity:%f %d",dir,equity,AccountEquity(),AccountEquity()>equity);
         equity=AccountEquity();
      }
      else if(dir==-1){
         if(AccountEquity()>equity){
            ticket=sell_limit();
            dir=-1;
         }
         else{
            ticket=buy();
            dir=1;
         }
         equity=AccountEquity();
      }
   }
   if(OrdersTotal()>0){
     modify();
   }
}

int buy(){
   double stoploss,takeprofit,equity;
   stoploss=Ask-LOSS*Point;
   takeprofit=Ask+PROFIT*Point;
   return OrderSend(Symbol(),OP_BUY,Lots,Ask,SLIPPAGE,stoploss,takeprofit,"",0,0,Green);
}
int buy_limit(){
   double price,stoploss,takeprofit,equity;
   price=Ask*(1-Delta);
   stoploss=price-LOSS*Point;
   takeprofit=price+PROFIT*Point;
   return OrderSend(Symbol(),OP_BUYLIMIT,Lots,price,SLIPPAGE,stoploss,takeprofit,"",0,0,Green);
}

int sell()
{
   double stoploss,takeprofit,equity;
   stoploss=(float)(Bid+LOSS*Point);
   takeprofit=(float)(Bid-PROFIT*Point);
   return OrderSend(Symbol(),OP_SELL,Lots,Bid,SLIPPAGE,stoploss,takeprofit,"",0,0,Green);
}
int sell_limit()
{
   double price,stoploss,takeprofit,equity;
   price=Bid*(1+Delta);
   stoploss=price+LOSS*Point;
   takeprofit=price-PROFIT*Point;
   return OrderSend(Symbol(),OP_SELLLIMIT,Lots,price,SLIPPAGE,stoploss,takeprofit,"",0,0,Green);
}

void modify()
{
   int ticket,orderType;
   double price,profit,stoploss,takeprofit;
   OrderSelect(0,SELECT_BY_POS);
   ticket=OrderTicket();
   orderType=OrderType();
   price=OrderOpenPrice();
   profit=OrderProfit();
   if(orderType==OP_BUY){
      stoploss=(float)(Bid-LOSS*Point);
      takeprofit=(float)(Bid+PROFIT*Point);
      if(stoploss-OrderStopLoss()>GAP*Point) 
      OrderModify(ticket,0,stoploss,takeprofit,0,Green);
   }
   else{
      stoploss=(float)(Ask-LOSS*Point);
      takeprofit=(float)(Ask+PROFIT*Point);
      if(stoploss-OrderStopLoss()>GAP*Point) 
      OrderModify(ticket,0,stoploss,takeprofit,0,Green);
   }
}
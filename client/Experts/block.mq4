#property copyright   "eblics"
#property link        "http://www.eblics.com"
#property description "block"

input double Lots=0.5;
input int LOSS=500;
input int PROFIT=40;
input int GAP=50;
input int SLIPPAGE=3;
input double Delta=0.618;

int bticket=-1,sticket=-1;
int orderType;
double price,stoploss,takeprofit,equity;
datetime opentime;
void OnTick()
{  
   if(OrdersTotal()==0)
   {  
      price=Ask+GAP*Point;   
      stoploss=price-LOSS*Point;
      takeprofit=price+PROFIT*Point;
      bticket=OrderSend(Symbol(),OP_BUYSTOP,Lots,price,SLIPPAGE,stoploss,takeprofit,"",0,0,Green);
      Print("buy ",GetLastError());
      PrintFormat("price:%f,stoploss:%f,takeprofit:%f",price,stoploss,takeprofit);
      
      price=Bid-GAP*Point;   
      stoploss=price+LOSS*Point;
      takeprofit=price-PROFIT*Point;
      sticket=OrderSend(Symbol(),OP_SELLSTOP,Lots,price,SLIPPAGE,stoploss,takeprofit,"",0,0,Green);
      Print("sell ",GetLastError());
      PrintFormat("price:%f,stoploss:%f,takeprofit:%f",price,stoploss,takeprofit);
   }
   if(OrdersTotal()>0)
   {
      if(OrderSelect(bticket,SELECT_BY_TICKET)){
         if(OrderType()==OP_BUY&&sticket>0){
            OrderDelete(sticket,Green);
            sticket=-1;
         }
      }
         
      if(OrderSelect(sticket,SELECT_BY_TICKET)){
         if(OrderType()==OP_SELL&&bticket>0){
            OrderDelete(bticket,Green);
            bticket=-1;
         }
      }    
   }
   //modify();
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
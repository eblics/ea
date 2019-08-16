//+------------------------------------------------------------------+
//|                                                         eSeg.mq4 |
//|                                  Copyright 2019-Forever, Eblics. |
//|                                            http://www.nosite.com |
//+------------------------------------------------------------------+
#property copyright   "Eblics"
#property link        "http://www.nosite.com"
#property description "expert based on law of large numbers and probability"

#define MAXINT   2147483647
#define MAGICMA  19820211
//splipage 
input int SLIPPAGE=5;
//--- Inputs
//input int    Period=144;
//the table of probabilities, 70 to 140
double PRBT[]={0.00,78.82, 83.37, 99.81, 102.50, 107.35, 110.13, 111.77, 113.56, 119.50} ;
//the number of lots should be splitted
int  NLOTS = ArraySize(PRBT);
//the coefficent of LOTS according to equity, lots=equity/CO_LOTS
input double CO_LOTS = 10000;
//the profit gap which can do stoploss.
input double PG=5;

input double SG=120;

double getUnitLots(){
    double maxlots  = AccountEquity()/CO_LOTS;
    double unitLots = maxlots/NLOTS;
    unitLots=unitLots>0.01?unitLots:0.01;
    return unitLots;
}

/*******************************************
* system functions 
********************************************/
int OnInit() 
{ 
   return(INIT_SUCCEEDED); 
} 

void OnTick()
{
    int ticket;
    int res;
    int i;
    double lots;
    double spread = Ask-Bid;
    double pr=0;
    double bp,tp;
    double op;
    double p;
    double maxlots;
    double unitlots;
    double stoploss;
    
    if(spread>SLIPPAGE){
        Print("64 spread failed");
        return;
    }

    maxlots=AccountEquity()/CO_LOTS;
    maxlots=maxlots>0.01 ? maxlots:0.01;
    unitlots=getUnitLots();
    
    for(i=0;i<OrdersTotal();i++){
        if(OrderSelect(i,SELECT_BY_POS)==false) continue;
        if(OrderProfit()<0 && OrderProfit()+OrderSwap()+OrderCommission()>0){
          OrderClose(OrderTicket(),OrderLots(),Bid,SLIPPAGE,Yellow);
        }
    }
    for(i=0;i<OrdersTotal();i++){
        if(OrderSelect(i,SELECT_BY_POS)==false) continue;
        p=OrderOpenPrice();
        if(Bid-p<PG*Point){
            lots+=OrderLots();
        }
        else{
            //stoploss=NormalizeDouble(Ask-(PG-4*SLIPPAGE)*Point,Digits);
            //stoploss=70;
            //stoploss=p+SLIPPAGE*Point;
            stoploss=p;
            if(Bid-SG*Point>stoploss){ 
                 stoploss=Bid-SG*Point;
             }
            if(stoploss>OrderStopLoss()) {
                if(!OrderModify(OrderTicket(),OrderOpenPrice(),stoploss,OrderTakeProfit(),0,Green)){
                    PrintFormat("OrderModify failed: ticket:%d error:%d ask:%f stoploss:%f orderstoploss:%f p:%f point:%f",OrderTicket(),GetLastError(),Ask,stoploss,OrderStopLoss(),p,Point);
                }
            }
        }
    }
    if(maxlots-lots<unitlots){
        //PrintFormat("86 lots exceeded,lots:%f maxlots:%f",lots,maxlots);
        return;
    }
    for(i=0;i<ArraySize(PRBT);i++){
        if(Ask<PRBT[i]){
            pr=(ArraySize(PRBT)-i)*1.0/NLOTS;
            bp=PRBT[i-1];
            tp=PRBT[i];
            break;
        }
    }
    
    op=0;
    lots=unitlots;
    //lots=unitlots*pr;
    lots=lots<0.01 ? 0.01 :lots;
    for(i=0;i<OrdersTotal();i++){
        ticket=OrderSelect(i,SELECT_BY_POS,MODE_TRADES);
        if(ticket==-1){
            Print("167 OrderSelect failed:"+GetLastError());
            continue;
        }
        p=OrderOpenPrice();
        if(bp<=p && p<tp){
            if(p>op) op=p;
        }
    }
    if(op==0){
        if(Ask<bp+(tp-bp)/6 && pr>=0.1 ) {
            ticket = OrderSend(Symbol(),OP_BUY,lots,Ask,SLIPPAGE,0,tp,"",MAGICMA,0,Blue);
            PrintFormat("111 openOrder %d pr:%f bp:%f tp:%f op:%f",ticket,pr,bp,tp,op);
        }
    } else{
        //if(Bid-op>PG*Point && Bid<tp-PG*Point){
        if(Bid-op>PG*Point && Bid<tp-PG*Point){
            ticket = OrderSend(Symbol(),OP_BUY,lots,Ask,SLIPPAGE,0,tp,"",MAGICMA,0,Blue);
            if(ticket==-1){
                PrintFormat("117 openOrder failed:%d",GetLastError());
                
            }else{
                PrintFormat("120 openOrder %d pr:%f bp:%f tp:%f",ticket,pr,bp,tp);
            }
        }
    }
}


/*******************************************
* end system functions 
********************************************/

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
int  NLOTS = 10;
//if a big jump happens(low probability), the inverting jump will happen soon after. BP is the jump number 
double  BP = 0.186;
//the value postive divided with negtive, this value is as a indicator to buy or sell
double PN  = 0.16;
//start from the price's probability
double SPP  = 0.2;
//the lowest price started with
double LP   = 70;
//the coefficent of LOTS according to equity, lots=equity/CO_LOTS
double CO_LOTS = 10000;
//the profit gap which can do stoploss.
input double PG=60;

double getUnitLots(){
    double maxlots  = AccountBalance()/CO_LOTS;
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

    maxlots=AccountBalance()/CO_LOTS;
    maxlots=maxlots>0.01 ? maxlots:0.01;
    unitlots=getUnitLots();
    for(i=0;i<OrdersTotal();i++){
        p=OrderOpenPrice();
        if(Ask-p<PG*Point){
            lots+=OrderLots();
        }
        else{
            stoploss=NormalizeDouble(Ask-(PG-2*SLIPPAGE)*Point,Digits);
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
            pr=(ArraySize(PRBT)-i)*0.1;
            bp=PRBT[i-1];
            tp=PRBT[i];
            break;
        }
    }
    
    op=0;
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
        if(Ask<(bp+tp)/2 && pr>0.3 ) {
            ticket = OrderSend(Symbol(),OP_BUY,unitlots,Ask,SLIPPAGE,0,tp,"",MAGICMA,0,Blue);
            PrintFormat("111 openOrder %d pr:%f bp:%f tp:%f op:%f",ticket,pr,bp,tp,op);
        }
    } else{
        if(Ask-op>PG*Point){
            ticket = OrderSend(Symbol(),OP_BUY,unitlots,Ask,SLIPPAGE,0,tp,"",MAGICMA,0,Blue);
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

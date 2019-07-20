//+------------------------------------------------------------------+
//|                                                         ePLN.mq4 |
//|                                  Copyright 2019-Forever, Eblics. |
//|                                            http://www.nosite.com |
//+------------------------------------------------------------------+
/******************************************
 *   This expert based on law of large number and probabilities of prices on history, which is calculated 
 * from the historical data. 
 *   Based on LLN, I found every 240 bars of m5 data, the number of pos allways near with the number of postive,
 * in other words, it's compile with LLN, so I use the signal which postive too many or negtive too many.
 *   Based on probability, can histogramed the historical price data, you can find the brown jumps are focused on 
 * certain prices, so these prices are grativy centers. I probabelise the data to get a probability table to decide
 * which prices can buy, which prices should sell.
 *****************************************/

#property copyright   "Eblics"
#property link        "http://www.nosite.com"
#property description "expert based on law of large numbers and probability"

#define MAGICMA  19820211
//splipage 
input int SLIPPAGE=5;
//--- Inputs
//input int    Period=144;
//the table of probabilities, 70 to 140
double PRBT[] = {1.00,1.00,1.00,1.00,1.00,1.00,0.96,0.94,0.90,0.86,0.84,0.82,0.80,0.80,0.79,0.79,0.79,
             0.79,0.79,0.79,0.78,0.78,0.78,0.77,0.76,0.76,0.75,0.74,0.72,0.70,0.68,0.64,0.59,0.57,
             0.54,0.53,0.51,0.49,0.46,0.41,0.35,0.29,0.23,0.18,0.16,0.16,0.15,0.13,0.11,0.08,0.06,
             0.04,0.03,0.01,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,
             0.00,0.00};
//number of bars per period
int  NBP = 240;
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
double PG=120;

bool checkLots(){
    int ticket;
    int res;
    double maxlots;
    double lots=0;
    double profit;
    double stoploss;
    double openPrice=0;
    for(int i=0;i<OrdersTotal();i++){
        ticket=OrderSelect(i,SELECT_BY_POS,MODE_TRADES);
        if(ticket==-1){
            Print("checkLots OrderSelect failed:"+GetLastError());
            continue;
        }
        profit=OrderProfit();
        openPrice=OrderOpenPrice();

        if((Bid-openPrice)>PG*Point){
            stoploss=Bid-(PG-20)*Point;
            if(OrderStopLoss()>=stoploss)
               continue;
            res=OrderModify(OrderTicket(),openPrice,stoploss,0,0,Green);
            if(res==-1){
                PrintFormat("checkLots OrderModify failed:%d %f",GetLastError(),stoploss);
                continue;
            }
        }
        else{
            lots+=OrderLots();
        }
    }
    maxlots=AccountBalance()/CO_LOTS;
    maxlots=maxlots>0.01 ? maxlots:0.01;
    //PrintFormat("checkLots balance:%f maxlots:%f lots:%f",AccountBalance(),maxlots,lots);
    return lots<maxlots;        
}

double getUnitLots(){
    double maxlots  = AccountBalance()/CO_LOTS;
    double unitLots = maxlots/NLOTS;
    unitLots=unitLots>0.01?unitLots:0.01;
    return unitLots;
}

//check for price segments, every segment can have&only have one order
bool checkPS(){
    int      i;
    int      ticket;
    double   openPrice;
    double   spp;
    int      si;
    double   sp;
    if(OrdersTotal()>0){
        ticket=OrderSelect(OrdersTotal()-1,SELECT_BY_POS,MODE_TRADES);
        if(ticket==-1){
            Print("checkPS OrderSelect failed:"+GetLastError());
        }
        openPrice=OrderOpenPrice();
        for(i=0;i<ArraySize(PRBT);i++){
            if(LP+i>=openPrice){
                spp=PRBT[i];
                break;
            }
        }
        spp+=1.0/NLOTS;
    }
    else{
        spp = (1.0/NLOTS)*OrdersTotal();
    }
    if(spp>1) spp=1;
    for(i=ArraySize(PRBT)-1;i>0;i--){
        if(PRBT[i]>spp){
            si=i;
            sp=LP+i+1;
            break;
        }
    }
    //PrintFormat("checkPS spp:%f si:%f sp:%f",spp,si,sp);
    return Ask<sp;
}

void checkForOpenBP(){
    int ticket;
    double lots;
    double spread = Ask-Bid;
    double bp     = Open[0]-Ask;
    
    if(!checkLots()){
        //Print("checkForOpenPN checkLots failed");
        return;
    }
    if(!checkPS()){
        //Print("checkForOpenPN checkPS failed");
        return;
    }
    if(spread>SLIPPAGE){
        //Print("checkForOpenPN spread failed");
        return;
    }

    lots=getUnitLots();
    if(bp>=BP){
        ticket = OrderSend(Symbol(),OP_BUY,lots,Ask,SLIPPAGE,0,0,"BP",MAGICMA,0,Blue);
        PrintFormat("checkForOpenBP openOrder %f",lots);
        if(ticket==-1){
            Print("checkForBP error"+GetLastError());
        }
    }
    PrintFormat("checkForOpenBP bp:%f",bp);
}

void checkForCloseBP(){
    int ticket;
    int res;
    double bp     = Open[0]-Ask;
    if(bp<=-BP){
        for(int i=0;i<OrdersTotal();i++){
            ticket=OrderSelect(i,SELECT_BY_POS,MODE_TRADES);
            if(ticket==-1){
                Print("checkForBP OrderSelect failed:"+GetLastError());
                continue;
            }
            double profit     = OrderProfit();
            double stoploss   = OrderStopLoss();
            double takeProfit = OrderTakeProfit();
            double openPrice  = OrderOpenPrice();
            string comment    = OrderComment();
            if(comment=="BP"){
                if(OrderProfit()>0){
                    res=OrderClose(ticket,OrderLots(),Bid,SLIPPAGE,Red);
                    if(res==-1){
                        Print("checkForCloseBP failed:"+GetLastError());
                    }
                }
            }
        }
    }
}

void checkForOpenPN(){
    double spread=Ask-Bid; 
    if(!checkLots()){
        //Print("checkForOpenPN checkLots failed");
        return;
    }
    if(!checkPS()){
        //Print("checkForOpenPN checkPS failed");
        return;
    }
    if(spread>SLIPPAGE){
        //Print("checkForOpenPN spread failed");
        return;
    }
    int ticket;
    double pn;
    double pos=0;
    double neg=0;
    double co;
    double lots;
    for(int i=0;i< NBP/2;i++){
        co=Close[i]-Open[i];
        if(co>0)      pos++;
        else if(co<0) neg++;
    }
    pn=(pos-neg)/(pos+neg);
    lots=getUnitLots();
    if(pn<-PN){
        ticket = OrderSend(Symbol(),OP_BUY,lots,Ask,SLIPPAGE,0,0,"PN",MAGICMA,0,Blue);
        PrintFormat("checkForOpenBP openOrder %f",lots);
        if(ticket==-1){
            Print("checkForBP error"+GetLastError());
        }
    }
}

void checkForClosePN(){
    double pn;
    double pos;
    double neg;
    double co;
    int i;
    int ticket;
    int res;
    
    for(i=0;i< NBP/2;i++){
        co=Close[i]-Open[i];
        if(co>0)      pos++;
        else if(co<0) neg++;
    }
    pn=(pos-neg)/(pos+neg);
    if(pn==0){
        for(i=0;i<OrdersTotal();i++){
            ticket=OrderSelect(i,SELECT_BY_POS,MODE_TRADES);
            if(ticket==-1){
                Print("checkForBP OrderSelect failed:"+GetLastError());
                continue;
            }
            double profit     = OrderProfit();
            double stoploss   = OrderStopLoss();
            double takeProfit = OrderTakeProfit();
            double openPrice  = OrderOpenPrice();
            string comment    = OrderComment();
            if(comment=="PN"){
                if(OrderProfit()>0){
                    res=OrderClose(ticket,OrderLots(),Bid,SLIPPAGE,Red);
                    if(res==-1){
                        Print("checkForClosePN OrderClose Failed:"+GetLastError());
                    }
                }
            }
        }
    }
}

void checkForOpen(){
   if(Bars<NBP/2)
      return;
   checkForOpenBP();
   checkForOpenPN();
}

void checkForClose(){
    checkForCloseBP();
    checkForClosePN();
}


/*******************************************
* system functions 
********************************************/
int OnInit() 
{ 
   return(INIT_SUCCEEDED); 
} 
uint time;
void OnTick()
{
   checkForOpen();
   checkForClose();
}


/*******************************************
* end system functions 
********************************************/

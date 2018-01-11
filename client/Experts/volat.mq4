#property copyright   "eblics"
#property link        "http://www.eblics.com"
#property description "mistake"
#include <rpcapi.mqh>

input string HOST="121.43.165.41";
input ushort PORT=8001;
input double PROB=0.5;
input double Lots=0.1;
input double TAKEPROFIT=1000;
input double SPREAD=40;
input double STOPLOSS=1000;
input double SLIPPAGE=3;
const int K=20;
const int BUFSIZE=64;
int g_hour=0;

int OnInit()
{
    rpcapi_init(HOST,PORT);
    return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   Print("停止服务器");
}

void OnTick(){
    apply_h1_vol();
}

int g_bars=0;
void apply_h1_vol(){
    if(Bars<K) return;
    if(g_bars==Bars)
        return;
    g_bars=Bars;
    
    //Print("bars is ",Bars,"g_bars is ",g_bars);
    if(OrdersTotal()>0){
        bool ret=OrderSelect(0,SELECT_BY_POS,MODE_TRADES);
        if(!ret){
        	Print("选择订单失败 "+GetLastError());
        	return;
        }
        int ticket=OrderTicket();
        close_order(ticket);
    }

    double ksteps[];
	double ropen[];double rclose[];double rhigh[];double rlow[];
	arr_reverse(ropen,Open,0,1,K);
	arr_reverse(rclose,Close,0,1,K);
	arr_reverse(rhigh,High,0,1,K);
	arr_reverse(rlow,Low,0,1,K);
    ArrayCopy(ksteps,ropen,0,0,K);
    ArrayCopy(ksteps,rclose,K,0,K);
    ArrayCopy(ksteps,rhigh,2*K,0,K);
    ArrayCopy(ksteps,rlow,3*K,0,K);
    double pred=rpcapi_predict_h1_vol(ksteps);
    Print(TimeCurrent());
    //arr_print(ropen);
    if(pred==0) return;
    if(pred>0&&pred>=SPREAD){
        send_order(OP_BUY);
        return;
    }
    if(pred<0&pred<=-SPREAD){
        send_order(OP_SELL);
        return;
    }
    

}
void close_order(int ticket){
    Print("close order",ticket);
    int orderType=OrderType();
    double price;
    price=orderType==OP_SELL?Ask:Bid;
    OrderClose(ticket,OrderLots(),price,SLIPPAGE,Red);
}
void send_order(int dir){
    Print("send order",dir);
    double price,stoploss,takeprofit;
    if(dir==OP_BUY){
        price=Ask;
        stoploss=price-STOPLOSS*Point;
        takeprofit=price+TAKEPROFIT*Point;
        OrderSend(Symbol(),dir,Lots,price,SLIPPAGE,stoploss,takeprofit,"",0,0,Green);
    }
    else{
        price=Bid;
        stoploss=price+STOPLOSS*Point;
        takeprofit=price-TAKEPROFIT*Point;
        OrderSend(Symbol(),dir,Lots,price,SLIPPAGE,stoploss,takeprofit,"",0,0,Green);
    }
}

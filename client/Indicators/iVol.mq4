#property copyright "eblics"
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict
//#property indicator_minimum    -5
//#property indicator_maximum    5
#property indicator_separate_window
#property  indicator_buffers 3
#property  indicator_color1  Blue
#property  indicator_color2  Red
#property  indicator_width1  1
#property  indicator_width2  1
#property  indicator_width3  1
#include <MovingAverages.mqh>
#include <rpcapi.mqh>
const int K=20;
//--- buffers
double ExtPBuffer[];
double ExtTBuffer[];
input string HOST="121.43.165.41";
input ushort PORT=8001;

int OnInit()
{
    string short_name;
    //--- 2 additional buffers are used for counting.
    IndicatorBuffers(3);
    SetIndexBuffer(0,ExtPBuffer);
    SetIndexBuffer(1,ExtTBuffer);
    //--- indicator line
    SetIndexStyle(0,DRAW_LINE);
    SetIndexStyle(1,DRAW_LINE);
    //--- name for DataWindow and indicator subwindow label
    short_name="P("+string(K)+")";
    IndicatorShortName(short_name);
    SetIndexLabel(0,short_name);

    short_name="T("+string(K)+")";
    IndicatorShortName(short_name);
    SetIndexLabel(1,short_name);

    SetIndexDrawBegin(0,K);
    rpcapi_init(HOST,PORT);
    //--- initialization done
    return(INIT_SUCCEEDED);
}
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
        const int prev_calculated,
        const datetime &time[],
        const double &open[],
        const double &high[],
        const double &low[],
        const double &close[],
        const long &tick_volume[],
        const long &volume[],
        const int &spread[])
{
    ArraySetAsSeries(ExtTBuffer,false);
    ArraySetAsSeries(ExtPBuffer,false);
    ArraySetAsSeries(open,false);
    ArraySetAsSeries(close,false);
	ArraySetAsSeries(high,false);
	ArraySetAsSeries(low,false);
    if(rates_total<K)
        return(0);
    int limit=0;
    if(prev_calculated==0) limit=K; else limit=prev_calculated-1;

    for(int i=limit;i<rates_total&&!IsStopped();i++){
        double ksteps[];
		ArrayCopy(ksteps,open,0,i-K,K);
		ArrayCopy(ksteps,close,K,i-K,K);
		ArrayCopy(ksteps,high,2*K,i-K,K);
		ArrayCopy(ksteps,low,3*K,i-K,K);
 
        ExtPBuffer[i]=rpcapi_predict_h1_vol(ksteps);
        ExtTBuffer[i]=(close[i]-open[i])/Point;
    }
    return(rates_total);
 }

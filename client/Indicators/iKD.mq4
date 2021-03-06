//+------------------------------------------------------------------+
//|                                                      iKD.mq4.mq4 |
//|                                                           eblics |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "eblics"
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict
//#property indicator_minimum    -5
//#property indicator_maximum    5
#property indicator_separate_window
#property  indicator_buffers 3
#property  indicator_color1  Silver
#property  indicator_color2  Blue
#property  indicator_color3  Red
#property  indicator_width1  1
#property  indicator_width2  1
#property  indicator_width3  1
#include <MovingAverages.mqh>
input int Shift=55;
input int MA1=55;
input int MA2=144;
//--- buffers
double ExtKBuffer[];
double ExtDBuffer[];
double ExtJBuffer[];
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
   string short_name;
//--- 2 additional buffers are used for counting.
   IndicatorBuffers(3);
   SetIndexBuffer(0,ExtKBuffer);
   SetIndexBuffer(1,ExtDBuffer);
   SetIndexBuffer(2,ExtJBuffer);
//--- indicator line
   SetIndexStyle(0,DRAW_LINE);
   SetIndexStyle(1,DRAW_LINE);
   SetIndexStyle(2,DRAW_LINE);
//--- name for DataWindow and indicator subwindow label
   short_name="K("+string(Shift)+")";
   IndicatorShortName(short_name);
   SetIndexLabel(0,short_name);
   
   short_name="D("+string(Shift)+")";
   IndicatorShortName(short_name);
   SetIndexLabel(1,short_name);

   SetIndexDrawBegin(0,MA1);
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
   ArraySetAsSeries(ExtDBuffer,false);
   ArraySetAsSeries(ExtKBuffer,false);
   ArraySetAsSeries(ExtJBuffer,false);
   ArraySetAsSeries(close,false);
   if(rates_total<MA1+1)
      return(0);

   //CalculateSimpleMA(rates_total,prev_calculated,MA1,close,ExtKBuffer);
   int limit=0;
   if(prev_calculated==0) limit=MA1; else limit=prev_calculated-1;
   
   for(int i=limit;i<rates_total&&!IsStopped();i++){
      ExtKBuffer[i]=K(Shift,i,close)/Point;
      ExtDBuffer[i]=D(Shift,i,close)/Point;
      ExtJBuffer[i]=ExtKBuffer[i]+ExtDBuffer[i];
      //ExtKBuffer[i]=K(Shift,i,close,ka)/Point;  
      //ExtDBuffer[i]=D(ka)/Point;
   }
   return(rates_total);
  }
//+------------------------------------------------------------------+
void CalculateSimpleMA(int rates_total,int prev_calculated,int period,const double &price[],double& buf[])
  {
   int i,limit;
//--- first calculation or number of bars was changed
   if(prev_calculated==0)
   
     {
      limit=period;
      //--- calculate first visible value
      double firstValue=0;
      for(i=0; i<limit; i++)
         firstValue+=price[i];
      firstValue/=period;
      buf[limit-1]=firstValue;
     }
   else
      limit=prev_calculated-1;
//--- main loop
   for(i=limit; i<rates_total && !IsStopped(); i++)
      buf[i]=buf[i-1]+(price[i]-price[i-period])/period;
//---
  }
double K(int shift,int index,const double& price[]){
  return (SimpleMA(index,MA1,price)-SimpleMA(index-Shift,MA1,price))/Shift;
}
   
double D(int shift,int index,const double& price[])
{
   double s=0;
   for(int i=index-shift+1;i<index;i++){
      s+=MathPow((SimpleMA(i+1,MA1,price)-SimpleMA(i,MA1,price)),2);
   } 
   return MathSqrt(s/Shift);  
}

double J(int index,double& D[])
{
   return D[index]-D[index-1];
}

double MA(int period,int shift){
   return iMA(NULL,NULL,period,0,MODE_SMA,PRICE_CLOSE,shift);
}

double sMA(int period,int index,const double& close[])
{
   double s=0;
   for(int i=index-period+1;i<=index;i++)
   {
      s+=close[i];
   }
   return s/period;
}

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
#property  indicator_color1  Red
#property  indicator_width1  1
#property  indicator_color2  Yellow
#property  indicator_width2  1
#property  indicator_color3  Yellow
#property  indicator_width3  1

#include <MovingAverages.mqh>
input int K=21;
input double Z=2;

//--- buffers

double ExtUBuffer[];
double ExtSBuffer[];

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
   string short_name;
//--- 2 additional buffers are used for counting.
   IndicatorBuffers(2);  
   IndicatorDigits(Digits);
   SetIndexBuffer(0,ExtUBuffer);
   SetIndexBuffer(1,ExtSBuffer);
   
//--- indicator line
   SetIndexStyle(0,DRAW_LINE);
   SetIndexStyle(1,DRAW_LINE);
//--- name for DataWindow and indicator subwindow label
   short_name="U";
   IndicatorShortName(short_name);
   SetIndexLabel(0,short_name);
   
   short_name="S";
   IndicatorShortName(short_name);
   SetIndexLabel(1,short_name);
//
//   short_name="X("+string(K)+")";
//   IndicatorShortName(short_name);
//   SetIndexLabel(2,short_name);
   
   SetIndexDrawBegin(0,K-1);
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
  
   ArraySetAsSeries(ExtUBuffer,false);
   ArraySetAsSeries(ExtSBuffer,false);
   ArraySetAsSeries(close,false);
   ArraySetAsSeries(open,false);
   ArraySetAsSeries(high,false);
   ArraySetAsSeries(low,false);
   if(rates_total<K+1)
      return(0);
   if(prev_calculated==0){
      ArrayInitialize(ExtUBuffer,0);
      ArrayInitialize(ExtSBuffer,0);
   }

   CalculateSU(rates_total,prev_calculated,open,close,ExtUBuffer,ExtSBuffer);
   return(rates_total);
  }
//+------------------------------------------------------------------+
void CalculateSU(int rates_total,int prev_calculated,const double &open[],const double &close[],double& ubuf[],double& sbuf[])
{
   int i;
      
   double dx=0,fu=0,fs=0,fa=0;
//--- main loop
   for(i=prev_calculated+1; i<rates_total && !IsStopped(); i++){
     //fa+=(open[i]+close[i])/2;
      dx=close[i]-open[i]; 
	  ubuf[i]=(ubuf[i-1]*i+dx)/(i+1);
	  sbuf[i]=sqrt((sbuf[i-1]*sbuf[i-1]*(i-1))+dx*dx)/(i+1);
	}
//---
  }


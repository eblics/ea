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
#property  indicator_buffers 1
#property  indicator_color1  Blue
//#property  indicator_color2  Red
//#property  indicator_color3  Silver
#property  indicator_width1  1
//#property  indicator_width2  1
//#property  indicator_width3  1
#include <MovingAverages.mqh>
input int Period=13;
input int PeriodV=5;
input int PeriodU=33;

//--- buffers
double ExtVBuffer[];
double ExtXBuffer[];
double ExtUBuffer[];
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
   string short_name;
//--- 2 additional buffers are used for counting.
   IndicatorBuffers(3);  
   IndicatorDigits(Digits);
   SetIndexBuffer(0,ExtVBuffer);
   SetIndexBuffer(1,ExtXBuffer);
   SetIndexBuffer(2,ExtUBuffer);
   
//--- indicator line
   SetIndexStyle(0,DRAW_LINE);
   //SetIndexStyle(2,DRAW_LINE);
//--- name for DataWindow and indicator subwindow label
   short_name="U";
   IndicatorShortName(short_name);
   SetIndexLabel(0,short_name);
   
   SetIndexDrawBegin(0,Period);
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
   ArraySetAsSeries(ExtVBuffer,false);
   ArraySetAsSeries(ExtUBuffer,false);
   ArraySetAsSeries(ExtXBuffer,false);
   ArraySetAsSeries(close,false);
   ArraySetAsSeries(open,false);
   if(rates_total<Period)
      return(0);

   CalculateSU(rates_total,prev_calculated,open,close);
   return(rates_total);
  }
//+------------------------------------------------------------------+
void CalculateSU(int rates_total,int prev_calculated,const double &open[],const double &close[])
  {
   int i,limit;
   double k1,k2;
//--- first calculation or number of bars was changed
   if(prev_calculated==0)  
   {
      limit=Period;
   }
   else
      limit=prev_calculated-1;
   for(i=limit; i<rates_total && !IsStopped(); i++){
	 double sum=0,var=0;
	 double sign=1;
	 for(int j=i;j>i-Period;j--){
	    sum+=close[j]-open[j];
	    var+=(close[j]-open[j])*(close[j]-open[j]);
	 }
	 if(close[i]-close[i-Period]<0) sign=-1;
	 //PrintFormat("%f %f",sum,var);
	 ExtVBuffer[i]=MathArcsin(sum/MathSqrt(var*Period));
    }
  }

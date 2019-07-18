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
#property  indicator_buffers 2
#property  indicator_color1  Blue
#property  indicator_color2  Red
#property  indicator_color3  Silver
#property  indicator_width1  1
#property  indicator_width2  1
#property  indicator_width3  1
#include <MovingAverages.mqh>
input int Period=144;

//--- buffers
//double ExtPBuffer[];
double ExtQBuffer[];
double    ExtXBuffer[];
double ExtHBuffer[];
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
   string short_name;
//--- 2 additional buffers are used for counting.
   IndicatorBuffers(2);  
   IndicatorDigits(Digits);
   SetIndexBuffer(0,ExtQBuffer);
   SetIndexBuffer(1,ExtXBuffer);
   //SetIndexBuffer(1,ExtQBuffer);
   //SetIndexBuffer(2,ExtHBuffer);
   
//--- indicator line
   SetIndexStyle(0,DRAW_LINE);
   SetIndexStyle(1,DRAW_LINE);
   //SetIndexStyle(1,DRAW_LINE);
   //SetIndexStyle(2,DRAW_LINE);
//--- name for DataWindow and indicator subwindow label
   short_name="K";
   IndicatorShortName(short_name);
   SetIndexLabel(0,short_name);
   short_name="M";
   IndicatorShortName(short_name);
   SetIndexLabel(1,short_name);
   
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
   //ArraySetAsSeries(ExtPBuffer,false);
   ArraySetAsSeries(ExtQBuffer,false);
   ArraySetAsSeries(ExtXBuffer,false);
   //ArraySetAsSeries(ExtHBuffer,false);
   ArraySetAsSeries(close,false);
   ArraySetAsSeries(open,false);
   if(rates_total<Period)
      return(0);

   CalculatePQ(rates_total,prev_calculated,open,close);
   return(rates_total);
  }
double EI=2;
//+------------------------------------------------------------------+
void CalculatePQ(int rates_total,int prev_calculated,const double &open[],const double &close[])
  {
   int i,limit;
   double w=0,l=0,co,cot;
//--- first calculation or number of bars was changed
   if(prev_calculated==0)  
   {
      ArrayInitialize(ExtXBuffer,0);
      limit=Period;
      for(i=0;i<limit;i++){
        cot+=MathAbs(close[i]-open[i]);
        co+=close[i]-open[i];
        //ExtHBuffer[i]=0.5;
      }
      //ExtPBuffer[limit-1]=p/Period;
      ExtQBuffer[limit-1]=MathAbs(co);
      ExtXBuffer[limit-1]=cot;
   }
   else
      limit=prev_calculated-1;
   for(i=limit; i<rates_total && !IsStopped(); i++){
	  cot=cot+MathAbs(close[i]-open[i])-MathAbs(close[i-Period]-open[i-Period]);
	  co=co+(close[i]-open[i])-(close[i-Period]-open[i-Period]);
	  ExtQBuffer[i]=MathAbs(co);
	  ExtXBuffer[i]=cot;
    }
  }


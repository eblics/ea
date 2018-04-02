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
#property  indicator_color2  Red
#property  indicator_color3  Silver
#property  indicator_width1  1
#property  indicator_width2  1
#property  indicator_width3  1
#include <MovingAverages.mqh>
input int Period=200;

//--- buffers
double ExtPBuffer[];
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
   IndicatorBuffers(3);  
   IndicatorDigits(Digits);
   SetIndexBuffer(0,ExtPBuffer);
   SetIndexBuffer(1,ExtQBuffer);
   SetIndexBuffer(2,ExtHBuffer);
   
//--- indicator line
   SetIndexStyle(0,DRAW_LINE);
   SetIndexStyle(1,DRAW_LINE);
   SetIndexStyle(2,DRAW_LINE);
//--- name for DataWindow and indicator subwindow label
   short_name="PQ";
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
   ArraySetAsSeries(ExtPBuffer,false);
   ArraySetAsSeries(ExtQBuffer,false);
   ArraySetAsSeries(ExtXBuffer,false);
   ArraySetAsSeries(ExtHBuffer,false);
   ArraySetAsSeries(close,false);
   ArraySetAsSeries(open,false);
   if(rates_total<Period)
      return(0);

   CalculatePQ(rates_total,prev_calculated,open,close);
   return(rates_total);
  }
//+------------------------------------------------------------------+
void CalculatePQ(int rates_total,int prev_calculated,const double &open[],const double &close[])
  {
   int i,limit;
   double p=0,q=0;
//--- first calculation or number of bars was changed
   if(prev_calculated==0)  
   {
      ArrayInitialize(ExtXBuffer,0);
      limit=Period;
      for(i=0;i<limit;i++){
        if(close[i]-open[i]>0){ 
            p+=1;
        }
        else if(close[i]-open[i]<0){
           q+=1;
        }
        //ExtHBuffer[i]=0.5;
      }
      ExtPBuffer[limit-1]=p/Period;
      ExtQBuffer[limit-1]=q/Period;
   }
   else
      limit=prev_calculated-1;
   for(i=limit; i<rates_total && !IsStopped(); i++){
	  //p=ExtPBuffer[i-1]*Period;
	  //if(i<Period+1000){
	  //  PrintFormat("%d %f %f",i,p,p+ExtXBuffer[i-Period]);
	  //}
	  p=q=0;
	  for(int j=i-1;j>=i-Period;j--){
	    if(close[j]-open[j]>0) p+=1;
	    if(close[j]-open[j]<0) q+=1;
	  }
	  //PrintFormat("p:%f q:%f",p,q);
	  ExtPBuffer[i]=p/Period;
	  ExtQBuffer[i]=q/Period;
	  ExtHBuffer[i]=0.5;
    }
  }


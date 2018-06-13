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
//#property  indicator_color3  Silver
#property  indicator_width1  1
//#property  indicator_width2  1
//#property  indicator_width3  1
#include <MovingAverages.mqh>
input int Period=144;

//--- buffers
double ExtBuffer1[];
double ExtBuffer2[];
double ExtBuffer3[];
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
   string short_name;
//--- 2 additional buffers are used for counting.
   IndicatorBuffers(3);  
   IndicatorDigits(Digits);
   SetIndexBuffer(0,ExtBuffer1);
   SetIndexBuffer(1,ExtBuffer2);
   SetIndexBuffer(2,ExtBuffer3);
   
//--- indicator line
   SetIndexStyle(0,DRAW_LINE);
   SetIndexStyle(1,DRAW_LINE);
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
   ArraySetAsSeries(ExtBuffer1,false);
   ArraySetAsSeries(ExtBuffer2,false);
   ArraySetAsSeries(ExtBuffer3,false);
   ArraySetAsSeries(close,false);
   ArraySetAsSeries(open,false);
   if(rates_total<Period)
      return(0);
   //PrintFormat("Balance:%f Enquity:%f Margin:%f Credit:%f",AccountBalance(),AccountEquity(),AccountMargin(),AccountCredit());
   CalculateSU(rates_total,prev_calculated,open,close);
   return(rates_total);
  }
double win=0,lose=0,max=0,min=12;
double wins=0,loses=0;

double CalcP(double v)
{
    return exp(MathAbs(v*10000));
}
//+------------------------------------------------------------------+
void CalculateSU(int rates_total,int prev_calculated,const double &open[],const double &close[])
  {
   int i,limit;
   double v,r,d;
//--- first calculation or number of bars was changed
   if(prev_calculated==0)  
   {
      double firstValue=0;
      limit=Period;
      for(i=0; i<limit; i++){
         v=close[i]-open[i];
         if(v>0) 
         {
           r+=CalcP(v);
         }
         if(v<0)
         {
           d+=CalcP(v);
         }
      }
      //if(d==0)d=1;
      if(d!=0) ExtBuffer1[limit-1]=r/d;
      //ExtBuffer2[limit-1]=d;
      //ExtBuffer2[limit-1]=wins/loses;
      PrintFormat("init win:%f lose:%f",r,d);
   }
   else
      limit=prev_calculated-1;
   for(i=limit; i<rates_total && !IsStopped(); i++){
     v=close[i]-open[i];
     if(v>0) 
     {
       r+=CalcP(v);
     }
     if(v<0)
     {
       d+=CalcP(v);
     }
     v=close[i-Period]-open[i-Period];
     if(v>0) 
     {
       r-=CalcP(v);
     }
     if(v<0)
     {
       d-=CalcP(v);
     }
     //if(d==0)d=1;
     if(d!=0)
        ExtBuffer1[i]=r/d;
     //ExtBuffer2[i]=d;
     //if(i<200)
     //PrintFormat("win:%f lose:%f",win,lose);
    }
  }

void CalculateSU2(int rates_total,int prev_calculated,const double &open[],const double &close[])
  {
   for(int i=prev_calculated; i<rates_total && !IsStopped(); i++){
     ExtBuffer3[i]=close[i]-open[i];   
	 if(close[i]>open[i]) { win+=1;wins+=close[i]-open[i];}
     if(close[i]==open[i]){ }
     if(close[i]<open[i]) { lose+=1;loses+=open[i]-close[i];}
     //if(close[i-Period]>open[i-Period]) {win-=1;wins-=close[i-Period]-open[i-Period];}
     //if(close[i-Period]<open[i-Period]) {lose-=1;loses-=open[i-Period]-close[i-Period];}
     if(lose!=0)
        ExtBuffer1[i]=win/lose;
     //if(i<200)
     PrintFormat("win:%f lose:%f",win,lose);
    }
  }
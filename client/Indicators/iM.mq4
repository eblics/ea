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
//#property indicator_separate_window
#property  indicator_buffers 1
#property  indicator_color1  Red
#property  indicator_color2  Yellow
//#property  indicator_color3  Silver
#property  indicator_width1  1
#property  indicator_width2  1
//#property  indicator_width3  1
#include <MovingAverages.mqh>
input int Period=200;
input int Gap=125;
input int SlipPage=3;
input int StopLoss=100;
input int TakeProfit=50;

//--- buffers
double ExtMBuffer[];
double ExtABuffer[];
//double ExtQBuffer[];
//double    ExtXBuffer[];
//double ExtHBuffer[];
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
   string short_name;
//--- 2 additional buffers are used for counting.
   IndicatorBuffers(1);  
   IndicatorDigits(Digits);
   SetIndexBuffer(0,ExtMBuffer);
   SetIndexBuffer(1,ExtABuffer);
   //SetIndexBuffer(2,ExtHBuffer);
   
//--- indicator line
   SetIndexStyle(0,DRAW_ARROW);
   SetIndexStyle(1,DRAW_LINE);
   //SetIndexStyle(2,DRAW_LINE);
//--- name for DataWindow and indicator subwindow label
   short_name="";
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
   ArraySetAsSeries(ExtMBuffer,false);
   ArraySetAsSeries(ExtABuffer,false);
   //ArraySetAsSeries(ExtQBuffer,false);
   //ArraySetAsSeries(ExtXBuffer,false);
   //ArraySetAsSeries(ExtHBuffer,false);
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
   double ma=0,gap=0,avg=0;
//--- first calculation or number of bars was changed
   if(prev_calculated==0)  
   {
      //ArrayInitialize(ExtMBuffer,0);
      limit=Period;
      //--- calculate first visible value
      double firstValue=0;
      for(i=0; i<limit; i++)
         firstValue+=close[i];
      firstValue/=Period;
      ExtABuffer[limit-1]=firstValue;
   }
   else
      limit=prev_calculated-1;
   for(i=limit; i<rates_total && !IsStopped(); i++){
	  //ma=iMA(Symbol(),0,Period,0,MODE_SMA,PRICE_CLOSE,0);
	  ma=ExtABuffer[i]=ExtABuffer[i-1]+(close[i]-close[i-Period])/Period;
	  gap=MaxGap(close,ExtABuffer,i,Period);
	  avg=AvgMa(close,ExtABuffer,i,Period);
	  //gap=AvgGap(close,ExtABuffer,i,Period)*1.2;
	  if(MathAbs(close[i]-ma)>gap &&(close[i]-ma)*avg<0){   
	  //if(MathAbs(close[i]-ma)>gap){     
	    ExtMBuffer[i]=close[i];
	    PrintFormat("p:%f  ma:%f",close[i],ma);
	  }
    }
  }
double MaxGap(const double& close[],double& ma[],int shift,int period){
    double max=0;
    double cm=0;
    for(int i=shift-1;i>=shift-period;i--){
        cm=MathAbs(close[i]-ma[i]);
        if(cm>max)
            max=cm;
    }
    return max;
}

double AvgGap(const double& close[],double& ma[],int shift,int period){
    double avg=0;
    double cm=0;
    for(int i=shift-1;i>=shift-period;i--){
        avg+=MathAbs(close[i]-ma[i]);
    }
    return avg/Period;
}

double AvgMa(const double& close[],double& ma[],int shift,int period){
    double avg=0;
    double cm=0;
    for(int i=shift-1;i>=shift-period;i--){
        avg+=close[i]-ma[i];
    }
    return avg/Period;
}
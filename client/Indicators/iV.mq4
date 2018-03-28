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
   
   SetIndexDrawBegin(0,PeriodU);
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
   if(rates_total<PeriodU)
      return(0);

   CalculateSU(rates_total,prev_calculated,open,close);
   return(rates_total);
  }
//+------------------------------------------------------------------+
void CalculateSU(int rates_total,int prev_calculated,const double &open[],const double &close[])
  {
  int i,limit;
//--- first calculation or number of bars was changed
   if(prev_calculated==0)  
     {
      ArrayInitialize(ExtUBuffer,0);
      //ArrayInitialize(ExtVBuffer,0);
      ArrayInitialize(ExtXBuffer,0);
      limit=PeriodU;
      //--- calculate first visible value
      double dx=0,fu=0,fs=0,fa=0;
      for(i=0; i<limit; i++){
         dx=close[i]-open[i];
         ExtXBuffer[i]=dx*dx;
		 fu+=ExtXBuffer[i];
	  }
      fu/=PeriodU;
      ExtUBuffer[limit-1]=sqrt(fu);
     }
   else
      limit=prev_calculated-1;
   for(i=limit; i<rates_total && !IsStopped(); i++){
     //u=(close[i]-close[i-PeriodU])/PeriodU;
     //if( i<40)
     //   Print(ExtUBuffer[i]);
     
     ExtXBuffer[i]=(close[i]-open[i])*(close[i]-open[i]);
     if(i<40){
        //Print("=====================");
        //Print(ExtUBuffer[i-1]);
        //Print(ExtUBuffer[i-1]*ExtUBuffer[i-1]+(ExtXBuffer[i]-ExtXBuffer[i-PeriodU])/PeriodU);
        //ExtUBuffer[i]=sqrt(ExtUBuffer[i-1]*ExtUBuffer[i-1]+(ExtXBuffer[i]-ExtXBuffer[i-PeriodU])/PeriodU);
        //Print((ExtXBuffer[i]-ExtXBuffer[i-PeriodU])/PeriodU);
        //Print(ExtUBuffer[i-1]*ExtUBuffer[i-1]+(ExtXBuffer[i]-ExtXBuffer[i-PeriodU])/PeriodU);
     }
     ExtUBuffer[i]=sqrt(ExtUBuffer[i-1]*ExtUBuffer[i-1]+(ExtXBuffer[i]-ExtXBuffer[i-PeriodU])/PeriodU);
     
     //return;
	 //v=(close[i]-close[i-PeriodV])/PeriodV;
	// ExtVBuffer[i]==(close[i]-close[i-PeriodV])/PeriodV;
	 
	 ExtVBuffer[i]=(close[i]-close[i-PeriodV])/PeriodV;
	 ExtVBuffer[i]=ExtVBuffer[i]/ExtUBuffer[i];
	 //ExtVBuffer[i]=ExtVBuffer[i]/close[i-PeriodV];
	 //Print(ExtVBuffer[i]);
    }
  }


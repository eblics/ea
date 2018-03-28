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
double ExtPBuffer[];
double ExtXBuffer[];
double ExtUBuffer[];
double ExtSBuffer[];
double ExtABuffer[];
double ExtTBuffer[];
double ExtBBuffer[];
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
   string short_name;
//--- 2 additional buffers are used for counting.
   IndicatorBuffers(7);  
   IndicatorDigits(Digits);
   SetIndexBuffer(0,ExtPBuffer);
   SetIndexBuffer(1,ExtTBuffer);
   SetIndexBuffer(2,ExtBBuffer);
   SetIndexBuffer(3,ExtUBuffer);
   SetIndexBuffer(4,ExtSBuffer);
   SetIndexBuffer(5,ExtXBuffer);
   SetIndexBuffer(6,ExtABuffer);
   
//--- indicator line
   SetIndexStyle(0,DRAW_LINE);
   SetIndexStyle(1,DRAW_LINE);
   SetIndexStyle(2,DRAW_LINE);
//--- name for DataWindow and indicator subwindow label
//   short_name="U("+string(K)+")";
//   IndicatorShortName(short_name);
//   SetIndexLabel(0,short_name);
//   
//   short_name="S("+string(K)+")";
//   IndicatorShortName(short_name);
//   SetIndexLabel(1,short_name);
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
   ArraySetAsSeries(ExtPBuffer,false);
   ArraySetAsSeries(ExtTBuffer,false);
   ArraySetAsSeries(ExtBBuffer,false);
   ArraySetAsSeries(ExtXBuffer,false);
   ArraySetAsSeries(ExtUBuffer,false);
   ArraySetAsSeries(ExtSBuffer,false);
   ArraySetAsSeries(ExtABuffer,false);
   ArraySetAsSeries(close,false);
   ArraySetAsSeries(open,false);
   ArraySetAsSeries(high,false);
   ArraySetAsSeries(low,false);
   if(rates_total<K+1)
      return(0);

   CalculateSU(rates_total,prev_calculated,K,open,close,high,low,ExtXBuffer,ExtUBuffer,ExtSBuffer,
        ExtPBuffer,ExtABuffer,ExtTBuffer,ExtBBuffer);
   return(rates_total);
  }
//+------------------------------------------------------------------+
void CalculateSU(int rates_total,int prev_calculated,int period,const double &open[],const double &close[],
    const double& high[],const double& low[],
    double& xbuf[],double& ubuf[],double& sbuf[],double& pbuf[],double& abuf[],double& tbuf[],double& bbuf[])
  {
   int i,limit;
//--- first calculation or number of bars was changed
   if(prev_calculated==0)  
     {
      limit=period;
      //--- calculate first visible value
      double dx=0,fu=0,fs=0,fa=0;
      for(i=0; i<limit; i++){
         fa+=close[i];
         //fa+=(open[i]+close[i])/2;
         dx=close[i]-open[i];
		 xbuf[i]=dx;
		 fu+=dx;fs+=dx*dx;
	  }
      fu/=period;fs/=period;fa=fa/period;
      ubuf[limit-1]=fu;
	  sbuf[limit-1]=sqrt(fs);
	  abuf[limit-1]=fa;
	  //Print("here  u:"+fu);
     }
   else
      limit=prev_calculated-1;
//--- main loop
   for(i=limit; i<rates_total && !IsStopped(); i++){
	  //abuf[i]=abuf[i-1]+(open[i]+close[i]-open[i-period]-close[i-period])/(2*period);
	  abuf[i]=abuf[i-1]+(close[i]-close[i-period])/period;
	  pbuf[i]=abuf[i-1]+ubuf[i-1];
	  tbuf[i]=pbuf[i]+sbuf[i-1]*Z;
	  bbuf[i]=pbuf[i]-sbuf[i-1]*Z;
	  xbuf[i]=close[i]-open[i];
      ubuf[i]=ubuf[i-1]+(xbuf[i]-xbuf[i-period])/period;
	  sbuf[i]=sqrt(sbuf[i-1]*sbuf[i-1]+(xbuf[i]*xbuf[i]-xbuf[i-period]*xbuf[i-period])/period);
	  if(i<70){
	    //Print("u:"+ubuf[i]," s:"+sbuf[i]);
	  //  Print("open:"+open[i]+"  close:"+close[i]+" period:"+period+" i-period:"+(i-period));
	  //  Print("x-:"+(xbuf[i-period])+"  x:"+xbuf[i]+"  u-:"+ubuf[i-1]+"  u:"+ubuf[i]+" s:"+sbuf[i]);
	  //  Print("x[0]:"+xbuf[0]);
	  }
	}
	//for(i=0;i<rates_total;i++){
	//    sbuf[i]=sqrt(sbuf[i]);
	//}
//---
  }


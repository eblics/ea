//+------------------------------------------------------------------+
//|                                               ePLN.mq4 |
//|                   Copyright 2019-Forever, Eblics. |
//|                                              http://www.nosite.com |
//+------------------------------------------------------------------+
#property copyright   "Eblics"
#property link        "http://www.nosite.com"
#property description "expert based on law of large numbers and probability"

/******************************************
 *   This expert based on law of large number and probabilities of prices on history, which is calculated 
 * from the historical data. 
 *   Based on LLN, I found every 240 bars of m5 data, the number of pos allways near with the number of postive,
 * in other words, it's compile with LLN, so I use the signal which postive too many or negtive too many.
 *   Based on probability, can histogramed the historical price data, you can find the brown jumps are focused on 
 * certain prices, so these prices are grativy centers. I probabelise the data to get a probability table to decide
 * which prices can buy, which prices should sell.
 *****************************************/
 
#define MAGICMA  19820211
//--- Inputs
//input int    Period=144;

//number of bars per period
int NBP=240

int OnInit() 
{ 
   //ArrayResize(arr,size); 
   double arr[]={1.01,2.02};
   int handle=FileOpen("mqarr",FILE_READ|FILE_WRITE|FILE_BIN); 
   Print("fileopend");
   Print("dsfds");
   if(handle!=INVALID_HANDLE) 
   { 
      FileSeek(handle,0,SEEK_END); 
      FileWriteArray(handle,arr,0,2); 
      FileClose(handle); 
   } 
   else{
      Print("error:"+GetLastError());
   }

   return(INIT_SUCCEEDED); 
} 



//+------------------------------------------------------------------+
//|                                               ePLN.mq4 |
//|                   Copyright 2019-Forever, Eblics. |
//|                                              http://www.nosite.com |
//+------------------------------------------------------------------+
/******************************************
 *   This expert based on law of large number and probabilities of prices on history, which is calculated 
 * from the historical data. 
 *   Based on LLN, I found every 240 bars of m5 data, the number of pos allways near with the number of postive,
 * in other words, it's compile with LLN, so I use the signal which postive too many or negtive too many.
 *   Based on probability, can histogramed the historical price data, you can find the brown jumps are focused on 
 * certain prices, so these prices are grativy centers. I probabelise the data to get a probability table to decide
 * which prices can buy, which prices should sell.
 *****************************************/

#property copyright   "Eblics"
#property link        "http://www.nosite.com"
#property description "expert based on law of large numbers and probability"

#define MAGICMA  19820211
//--- Inputs
//input int    Period=144;
//the table of probabilities, 70 to 140
double PRBT={1.00,1.00,1.00,1.00,1.00,1.00,0.96,0.94,0.90,0.86,0.84,0.82,0.80,0.80,0.79,0.79,0.79,
             0.79,0.79,0.79,0.78,0.78,0.78,0.77,0.76,0.76,0.75,0.74,0.72,0.70,0.68,0.64,0.59,0.57,
             0.54,0.53,0.51,0.49,0.46,0.41,0.35,0.29,0.23,0.18,0.16,0.16,0.15,0.13,0.11,0.08,0.06,
             0.04,0.03,0.01,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,
             0.00,0.00};
//number of bars per period
int NBP=240
//the value postive divided with negtive, this value is as a indicator to buy or sell
double PN=0.4

test for copy 


/*******************************************
* system functions 
********************************************/
int OnInit() 
{ 
   return(INIT_SUCCEEDED); 
} 

void OnTick()
{

}


/*******************************************
* end system functions 
********************************************/

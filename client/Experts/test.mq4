#include <socket.mqh>



int OnInit()
{
   return(INIT_SUCCEEDED);
}


void OnDeinit(const int reason)
{
  
   Print("停止服务器");
}
int step=0;
void OnTick(){
   if(step>2)
      return;
   step+=1;
   int a=1;
   int b=(a<<1);
   PrintFormat("%x %x",a,b);
}

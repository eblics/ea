#include <socket.mqh>

#property show_inputs
extern int port=2007;
extern string ip_address="";

string process(string to_me) {
   return(to_me+"!!!!!");
}

int start_server_loop(string id,int socket) {
    int msgsock = -1;   
    msgsock = sock_accept(socket);
    if (errno()!=0)
       return(-1);
    while(True) {
      string item = sock_receive(msgsock);
      if (errno()!=0 || IsStopped()==True) 
         return(msgsock);
         
      string response = process(item);
      sock_send(msgsock,response);  
      if (IsStopped()==True) {
        return(msgsock);
      }   
    }  
}
          
int listen_socket, msgsock;
int  start() {
   listen_socket = open_socket(port,ip_address);
   msgsock = start_server_loop("main",listen_socket);
   return(0);
}

int deinit() {
  sock_close(msgsock);
  sock_close(listen_socket);
  sock_cleanup();
  Print("DEINIT");
}
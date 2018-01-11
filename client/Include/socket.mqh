#include <winsock.mqh>
string HOST;
ushort PORT;
int    BUFSIZE=1024;
char BUFFER[1024];
string sock_send_double_array(double& arr[])
{
    string data[];
    arr_double2str(data,arr);
    string str=arr_join(data,',');
    ArrayInitialize(BUFFER,0);
    StringToCharArray(str,BUFFER);
    int sock=sock_connect(PORT,HOST);
    send(sock,BUFFER,sizeof(BUFFER),0);

    ArrayInitialize(BUFFER,0);
    recv(sock,BUFFER,sizeof(BUFFER),0);
    string msg=CharArrayToString(BUFFER);
    closesocket(sock);
    return msg;
}

int sock_init(string host,int port){
    HOST=host;
    PORT=port;
}

int sock_connect(int port,string ip_address){
    int sock,retval;
    sockaddr_in addrin;
    ref_sockaddr ref;
    char wsaData[]; // byte array of the future structure
    ArrayResize(wsaData, sizeof(WSAData)); // resize it to the size of the structure

    //int2struct(remote,sin_family,AF_INET);
    //int2struct(remote,sin_addr,inet_addr(ip_address));
    //int2struct(remote,sin_port,htons(port));
    char ch[];
    StringToCharArray(ip_address,ch);
    addrin.sin_family=AF_INET;
    addrin.sin_addr=inet_addr(ch);
    addrin.sin_port=htons(port);
    retval = WSAStartup(0x202, wsaData);
    if (retval != 0) {
        Print("Server: WSAStartup() failed with error "+ retval);
        return(-1);
    }
    sock=socket(AF_INET, SOCK_STREAM,0);
    if (sock == INVALID_SOCKET){
        Print("Server: socket() failed with error "+WSAGetLastError());
        WSACleanup();
        return(-1);
    }

    //ref.ref=addrin;
    retval=connect(sock,addrin,16);
    if (retval != 0) {
        Print("connect failed with error "+ +WSAGetLastError());
        WSACleanup();
        return(-1);
    }
    return sock;
}


int arr_double2str(string& dst[],double& src[]){
    for(int i=0;i<ArraySize(src);i++){
        dst[i]=DoubleToString(src[i]);
    }
    return 0;
}

string arr_join(string& src[],char sep){
    string str="";
    for(int i=0;i<ArraySize(src);i++){
        str+=src[i];
        if(i<ArraySize(src)-1){
            str+=",";
        }
    }
    return str;
}

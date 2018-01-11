set prj_path=%~dp0
set mql_path=C:\Users\eblics\AppData\Roaming\MetaQuotes\Terminal\F5C18A2156882613427FB4ACF0892997\MQL4\
mklink  %mql_path%\Include\WinSock.mqh %prj_path%\Include\WinSock.mqh 
mklink  %mql_path%Include\rpcapi.mqh %prj_path%Include\rpcapi.mqh 
mklink  %mql_path%\Experts\bayes.mq4 %prj_path%\Experts\bayes.mq4 
mklink  %mql_path%\Experts\maK.mq4 %prj_path%\Experts\maK.mq4 
mklink  %mql_path%\Experts\mistake.mq4 %prj_path%Experts\mistake.mq4 
mklink  %mql_path%\Experts\block.mq4 %prj_path%Experts\block.mq4 
mklink  %mql_path%\Experts\test.mq4 %prj_path%Experts\test.mq4 
mklink  %mql_path%\Experts\volat.mq4 %prj_path%Experts\volat.mq4 
mklink  %mql_path%\Indicators\iKD.mq4 %prj_path%\Indicators\iKD.mq4 
mklink  %mql_path%\Indicators\iVol.mq4 %prj_path%\Indicators\iVol.mq4 

set prj_path=%~dp0
set mql_path=C:\Users\eblics\AppData\Roaming\MetaQuotes\Terminal\F5C18A2156882613427FB4ACF0892997\MQL4\
mklink /j %mql_path%\Experts\ea %prj_path%Experts 
mklink /j %mql_path%\Indicators\ea %prj_path%\Indicators 

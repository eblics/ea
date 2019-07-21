set prj_path=%~dp0
set mql_path="C:\Program Files (x86)\MetaTrader - EXNESS\MQL4"
mklink /j %mql_path%\Experts\ea %prj_path%Experts 
mklink /j %mql_path%\Indicators\ea %prj_path%\Indicators 

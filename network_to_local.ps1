"*********************************************CATS*********************************************"






"*********************************************CATS - NETWORK TO LOCAL*********************************************"


$network_source = '//v2GBSdwesSQL01/IMPORT/OPS'



$local_dest = "C:\Users\1155449\project_home\cats\Reports\Template"



robocopy $network_source $local_dest   'Activity Mapping.txt' 'CATS_Hours.txt' 'CostCenterMapping.txt' 'Depot.csv' 'IDExceptionMapping.csv' 'Mapping 1.txt' 'WORKCENTERTOCELL.txt' 'PRISIMPoolMapping.txt' 'ProductionPegging.csv' /Z /R:5 /W:5 /MT:16 /V /LEV:1 /XO

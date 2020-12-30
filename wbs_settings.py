# %%
import os
from configparser import RawConfigParser
from common_functions import get_universal_path



def create_config(config_file):
    print('Creating New Config')
    config.add_section('wbs_settings')
    config['wbs_settings']['schema']='dbo'
    config['wbs_settings']['database']='OpsData'
    config['wbs_settings']['server']='v2GBSdwesSQL01'
    config['wbs_settings']['wbs_table']='WBS_MAPPING_UPDATES'
    config['wbs_settings']['database_folder']='//v2gbsdwessql01/IMPORT/OPS/SQL/WBS_Mapping'

    with open(config_file, 'w' ) as cfile:
        config.write(cfile)    



def init():
    global config
    global abs_executable_path
    


    abs_executable_path = get_universal_path()
    config_filename = 'config_wbs.ini'
    config_file = os.path.join( abs_executable_path, config_filename)
    config = RawConfigParser(allow_no_value=True)
    
    if not os.path.isfile(config_file):
        create_config(config_file)
    
    config.read(config_file)




# %%

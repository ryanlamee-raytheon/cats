# %%
import os
import pyodbc
import pandas as pd
import wbs_settings
from datetime import datetime
from common_functions import get_universal_path, robocopy_file

# %%
def get_sql_insert_statement(df):
    sql = "INSERT INTO [OpsData].[dbo].[WBS_MAPPING_UPDATES] ("
    sql +=  "[" + "], [".join([str(i) for i in df.columns.tolist()]) + "]) VALUES ("
    sql += "?, ".join('' for _ in df.columns.tolist()) + "?)"            
    return sql


# %%
class UpdateTable():
    def __init__(self):
        super().__init__()

        wbs_settings.init()
        wbs_settings.config
        self.db = wbs_settings.config["wbs_settings"]["database"]
        self.schema =wbs_settings.config["wbs_settings"]["schema"] 
        self.table = wbs_settings.config["wbs_settings"]["wbs_table"]
        self.cur_date = datetime.now()
        self.cnxn = pyodbc.connect(f'DSN={wbs_settings.config["wbs_settings"]["server"]}')
        self.cursor = self.cnxn.cursor()
        self.out_file = get_universal_path( os.path.join(wbs_settings.abs_executable_path, f'wbs_updates_{self.cur_date.strftime("%#d%b%Y %H%M")}.txt' ) )
        self.src_folder = get_universal_path( os.path.dirname(self.out_file) )
        
        self.database_folder = get_universal_path( wbs_settings.config['wbs_settings']['database_folder'])
        self.database_file = os.path.join( self.database_folder, os.path.basename(self.out_file) )
        self.init_df()
        print('Sucessfully Initialized UpdateTable Class')




    def init_df(self):
        self.df = pd.read_sql(f'select top 1 * from [{self.db}].[{self.schema}].[{self.table}]', con=self.cnxn)
        self.df.drop(index=0, inplace=True)
        # for i in self.df.columns:
        #     self.df.loc[0, i] = None
        print('Sucessfully Initialized DataFrame')











def update_table()
    update_tbl = UpdateTable()
    update_tbl.df = update_tbl.df.append(pd.Series(), ignore_index=True)
    update_tbl.df['Update Date'] = update_tbl.cur_date.strftime('%#m/%#d/%Y %H%M')
    update_tbl.df.to_csv(update_tbl.out_file, sep='\t', index=False)
    print('Exported Updates')
    print(robocopy_file(src=update_tbl.src_folder, dst=update_tbl.database_folder, file=os.path.basename(update_tbl.out_file), verbose=True).stdout)
    print('Saved Updates To Database Folder')
    database_df = pd.read_csv(update_tbl.database_file, sep='\t')

    sql = get_sql_insert_statement(database_df)

    database_df = database_df.astype(object).where(pd.notnull(database_df), None)

    for idx, row in database_df.iterrows():
        update_tbl.cursor.execute(sql, tuple(row))


    update_tbl.cursor.commit()
    os.remove(update_tbl.out_file)
    del update_tbl.out_file    

    print('Sucessfully Updated Database Table')


# %%
if __name__ == '__main__':
    update_table()


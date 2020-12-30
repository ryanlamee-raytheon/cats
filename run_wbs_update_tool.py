import sys
from qt_forms.qt_pandas_model import pandasModel 
# from PyQt5 import QtQml
import PySimpleGUI as sg
import pyodbc
import pandas as pd
import wbs_update_tool
from collections import defaultdict
from PyQt5.QtWidgets import QApplication, QTableView
#from qt_dataframe_model import DataFrameModel
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(QtWidgets.QMainWindow, wbs_update_tool.Ui_MainWindow):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.setMaximumSize(QtCore.QSize(388, 410))
        
        # #self.model = DataFrameModel()
        # print('self model set')
        # self.engine = QtQml.QQmlApplicationEngine()
        # self.engine.rootContext().setContextProperty("table_model", self.model)
        # self.qml_path = os.path.join(os.path.dirname(__file__), "main .qml")
        # self.engine.load(QtCore.QUrl.fromLocalFile(self.qml_path))

        self.cnxn = pyodbc.connect('DSN=v2GBSdwesSQL01')
        self.cursor = self.cnxn.cursor()
        self.df = pd.read_sql('select * from [OpsData].[dbo].[WBS_MAPPING_UPDATES]', con=self.cnxn)
        self.ocr_df = None


        self.data_update_dict = defaultdict(list)
        
        self.col_filter = None
        self.filter_value = None
        self.set_names()
        self.set_event_filters()
        self.set_db_column_names()
        self.set_lookup_dict()
        self.set_widget_lookup_dict()
        self.original_width = self.width()
        self.original_height = self.height()
        # self.set_sql_queries()

    def set_ocr_df(self, ocr_df):
        self.ocr_df = ocr_df


    def set_qapplication(self, q_application):
        self.app = q_application

    def update_dataframe(self):
        self.data_update_dict['Prj Def'].append(self.prj_def_comboBox.currentText())
        self.data_update_dict['Prj Desc'].append(self.prj_desc_comboBox.currentText())
        self.data_update_dict['Network'].append(self.network_comboBox.currentText())
        self.data_update_dict['NWA Charge Number'].append(self.nwa_comboBox.currentText())
        self.data_update_dict['WBS Element'].append(self.wbs_comboBox.currentText())
        self.data_update_dict['Alternative SP1'].append(self.alt_sp1_comboBox.currentText())
        self.data_update_dict['Alternative SP2'].append(self.alt_sp2_comboBox.currentText())
        self.data_update_dict['Alternative SP3'].append(self.alt_sp3_comboBox.currentText())
        self.data_update_dict['Phase'].append(self.phase_comboBox.currentText() )
        self.data_update_df = pd.DataFrame.from_dict(self.data_update_dict, orient='columns')



    def set_event_filters(self):
        self.prj_def_comboBox.installEventFilter(self)
        self.prj_desc_comboBox.installEventFilter(self)
        self.network_comboBox.installEventFilter(self)
        self.nwa_comboBox.installEventFilter(self)
        self.wbs_comboBox.installEventFilter(self)
        self.alt_sp1_comboBox.installEventFilter(self)
        self.alt_sp2_comboBox.installEventFilter(self)
        self.alt_sp3_comboBox.installEventFilter(self)
        self.phase_comboBox.installEventFilter(self)



    def eventFilter(self, source, event):
        if event.type() == QtCore.QEvent.FocusOut:
            if len(source.currentText()) >= 1:
                print('updating filter_value')
                self.col_filter = self.lookup_dict[source.name]
                self.filter_value = source.currentText()
        return super(Ui_MainWindow, self).eventFilter(source, event)


    def set_lookup_dict(self):
        self.lookup_dict = {
            'prj_def_comboBox': 'Prj Def',
            'prj_desc_comboBox': 'Prj Desc',
            'network_comboBox': 'Network',
            'nwa_comboBox': 'NWA Charge Number',
            'wbs_comboBox': 'WBS Element',
            'alt_sp1_comboBox': 'Alternative SP1',
            'alt_sp2_comboBox': 'Alternative SP2',
            'alt_sp3_comboBox': 'Alternative SP3',
            'phase_comboBox': 'Phase'}        


    def set_widget_lookup_dict(self):
         self.widget_lookup_dict = {
            'prj_def_comboBox': self.prj_def_comboBox.name,
            'prj_desc_comboBox': self.prj_desc_comboBox.name,
            'network_comboBox': self.network_comboBox.name,
            'nwa_comboBox': self.nwa_comboBox.name,
            'wbs_comboBox': self.wbs_comboBox.name,
            'alt_sp1_comboBox': self.alt_sp1_comboBox.name,
            'alt_sp2_comboBox': self.alt_sp2_comboBox.name,
            'alt_sp3_comboBox': self.alt_sp3_comboBox.name,
            'phase_comboBox': self.phase_comboBox.name}


    def set_names(self):
        self.prj_def_comboBox.name = 'prj_def_comboBox'
        self.prj_desc_comboBox.name = 'prj_desc_comboBox'
        self.network_comboBox.name = 'network_comboBox'
        self.nwa_comboBox.name = 'nwa_comboBox'
        self.wbs_comboBox.name = 'wbs_comboBox'
        self.alt_sp1_comboBox.name = 'alt_sp1_comboBox'
        self.alt_sp2_comboBox.name = 'alt_sp2_comboBox'
        self.alt_sp3_comboBox.name = 'alt_sp3_comboBox'
        self.phase_comboBox.name = 'phase_comboBox'


    def set_db_column_names(self):
        self.prj_def_comboBox.table_column = 'Prj Def'
        self.prj_desc_comboBox.table_column = 'Prj Desc'
        self.network_comboBox.table_column = 'Network'
        self.nwa_comboBox.table_column = 'NWA Charge Number'
        self.wbs_comboBox.table_column = 'WBS Element'
        self.alt_sp1_comboBox.table_column = 'Alternative SP1'
        self.alt_sp2_comboBox.table_column = 'Alternative SP2'
        self.alt_sp3_comboBox.table_column = 'Alternative SP3'
        self.phase_comboBox.table_column = 'Phase'        

        
    def update_table_pressed(self):
        self.update_dataframe()
        
    def show_options(self):
        if self.width() == self.original_width:
            self.setMaximumSize(QtCore.QSize(831, 410))
            self.setFixedSize(831, self.height())
        else:            
            self.setMaximumSize(QtCore.QSize(388, 410))
            self.setFixedSize(388, self.height())


    def prj_def_updated(self):
        pass
    def prj_desc_updated(self):
        pass
    def alt_sp1_updated(self):
        pass
    def alt_sp2_updated(self):
        pass
    def alt_sp3_updated(self):
        pass
    def network_updated(self):
        pass
    def nwa_updated(self):
        pass
    def wbs_element_updated(self):
        pass



    def ocr_read_pressed(self):
        # ocr_model = pandasModel( self.df[self.df[self.col_filter] == self.filter_value] )
        try:
            len(self.ocr_df.index)
        except:
            sg.PopupOK('OCR DF Not Initialized!')

        ocr_model = pandasModel(self.ocr_df[['WBS Element', 'Network', 'NWA Charge Number', 'Prj Def']] )
        self.ocr_tableView.setModel(ocr_model)
        self.ocr_tableView.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        

    # def update_data_table(self):
    #     self.ocr_tableView.setColumnCount(len(self.df.columns))
    #     self.ocr_tableView.setRowCount(len(self.df.index))
        
    #     for row_idx in self.df.iloc[0:10, :].iterrows():
    #         for col_idx, col in enumerate(self.df.columns.tolist()):
    #             self.ocr_tableView.setItem(row_idx, col_idx, QtWidgets.QTableWidgetItem(str(self.df.loc[row_idx, col] ) ) )


    def ocr_fill_pressed(self):
        self.init_table_model()

    def init_table_model(self):
        from qt_dataframe_model import DataFrameModel
        self.model = DataFrameModel(self.df.iloc[0:10, 0:10])
        # from qt_pandas_model import pandasModel
        # self.model = pandasModel(self.df.iloc[0:10, 0:10])
        self.table_view = QtWidgets.QTableView()
        self.table_view.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)        
        self.table_view.setModel(self.model)
        self.table_view.setWindowTitle('WBS Table')
        self.table_view.show()

      



    def continuous_ocr_toggled(self):
        pass
    def ocr_read_freq_changed(self):
        pass

    def populate_phase_dropdown(self):       
        self.phase_list = ['DEPOT', 'DEV', 'OTHER', 'PROD']
        self.phase_comboBox.addItems(self.phase_list)





    def update_button_pressed(self):
        self.update_dropdowns()

    def update_dropdowns(self):
        if not self.col_filter and self.filter_value:
            return

        t_df = self.df[self.df[self.col_filter] == self.filter_value]

        for widget_name, col_name in self.lookup_dict.items():
            t_widget = getattr(self, widget_name)
            t_value = ''

            if col_name != self.col_filter and col_name == 'Phase':
                try:
                    t_data = list(set(t_df.loc[:, col_name].tolist()))
                    
                    msg = '|'.join(_ for _ in t_data)

                    if len(t_data) == 1:
                        t_value = t_data
                    elif len(t_data) >= 2:
                        t_value = t_data[0]
                        
                    t_widget.setCurrentIndex(self.phase_list.index(t_value)) 
                except Exception as e:
                    print('Exception encountered adding items to dropdowns')
            elif col_name != self.col_filter:
                t_widget.clear()
                try:
                    t_data = list(set(t_df.loc[:, col_name].tolist()))
                    msg = '|'.join(_ for _ in t_data)
                    
                    if len(t_data) == 1:
                        t_value = t_data
                        print('Adding 1')
                    elif len(t_data) >= 2 <= 9:
                        t_value = t_data
                        print('Adding Between 2 And 10')
                    elif len(t_data) >= 10:
                        t_value = t_data[:10]
                        print('Adding 10')
                    elif len(t_data) == 0:
                        t_value = None
                        print('Adding None')

                    t_widget.addItems(t_value)
                    t_widget.setCurrentIndex(0)
                except:
                    pass




_cache = []

def show(title=''):
    app = QApplication(sys.argv)

    if QApplication.instance() is None:
        _cache.append(app)
    
    wbs_update_form = Ui_MainWindow()


    wbs_update_form.set_qapplication(app)
    wbs_update_form.setWindowTitle(title)
    wbs_update_form.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    _cache.append(wbs_update_form)
    wbs_update_form.populate_phase_dropdown()
    wbs_update_form.show()
    app.exec_()



# def init_table_model(wbs_update_form):
#     import numpy as np
#     df_app = QtGui.QGuiApplication(sys.argv)
#     df = pd.DataFrame(np.random.randint(0, 100, size=(6, 7)), columns=list('ABCDEFG'))
##     model = DataFrameModel(df)
#     engine = QtQml.QQmlApplicationEngine()
#     engine.rootContext().setContextProperty("table_model", model)
#     qml_path = os.path.join(os.path.dirname(__file__), "main.qml")
#     engine.load(QtCore.QUrl.fromLocalFile(qml_path))
#     if not engine.rootObjects():
#         sys.exit(-1)
#     engine.quit.connect(df_app.quit)
#     model.show()


if __name__ == '__main__':
    show('WBS Update Tool')

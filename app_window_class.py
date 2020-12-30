import os
import time
import pyautogui
import PySimpleGUI as sg
import pygetwindow as gw
from PyQt5 import QtWidgets
from PIL import ImageGrab, Image
from common_functions import get_universal_path



def set_image_dpi(file_path):
    import tempfile
    from PIL import Image

    im = Image.open(file_path)
    length_x, width_y = im.size
    factor = min(1, float(1024.0 / length_x))
    size = int(factor * length_x), int(factor * width_y)
    im_resized = im.resize(size, Image.ANTIALIAS)
    temp_file = tempfile.NamedTemporaryFile(delete=False,   suffix='.png')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(300, 300))
    return temp_filename





class SapApplicationWindowReader():
    def __init__(self):
        self.abs_executable_path = get_universal_path()        
        self.init_tmp_screenshot_folder()
        self.target_window = None
        self.window_found = False
        self.search_str = None
        self.set_screenshot_lookup_dict()
 
    def init_tmp_screenshot_folder(self):
        self.tmp_screenshot_folder = os.path.join(self.abs_executable_path, 'tmp_sap_screenshots')        
        if not os.path.isdir(self.tmp_screenshot_folder):
            os.mkdir(self.tmp_screenshot_folder)
 
 
    def set_search_str(self, search_str):
        self.search_str = search_str


    def find_target_window_by_search_str(self):
        self.set_output_file()
        window_titles = gw.getAllTitles()
        while True:
            try:
                print(f'Searching For Windows With The Title | {self.search_str}')
                self.target_window = gw.getWindowsWithTitle( [_ for _ in window_titles if self.search_str.lower() in _.lower()][0] )[0]
                self.window_found = True
                break
            except Exception as e:
                self.window_found = False
                if sg.PopupYesNo(f'No Window With The Title {self.search_str} Was Found.\n\nSearch Again?', keep_on_top=True, no_titlebar=True) == 'Yes':
                    self.search_str = sg.PopupGetText('Enter The Search Word:', keep_on_top=True, no_titlebar=True, default_text='project')
                    continue
                else:
                    raise(e)


    def set_output_file(self, file_ext='png'):
        self.out_file_ext = file_ext
        self.set_outfilename()
        self.out_file = os.path.join( self.tmp_screenshot_folder, f'{self.out_filename}.{self.out_file_ext}')
    
    def set_outfilename(self):
        self.out_filename = ''.join(_ for _ in self.search_str if str.isalnum(_) )


    def set_screenshot_lookup_dict(self):
        self.screenshot_dict = {
            'pil': self.save_screenshot_with_pil,
            'pyautogui': self.save_screenshot_with_pyautogui}


    def save_screenshot(self, image_exporter_name='pil'):
        cur_window = gw.getActiveWindow()
        self.target_window.activate()
        self.target_window.maximize()
        time.sleep(2)

        self.screenshot_dict[image_exporter_name]()
        cur_window.activate()
        cur_window.maximize()
        msg = f'Screenshot Exported Using | {image_exporter_name} | Output File | {self.out_file}'
        print(msg)



    def save_screenshot_with_pil(self):
        out_img = ImageGrab.grab(self.target_window._getWindowRect(), include_layered_windows=False)
        out_img.save(self.out_file)


    def save_screenshot_with_pyautogui(self):
        out_img = pyautogui.screenshot()
        out_img.save(self.out_file)






def save_project_builder_window(search_str=None):
    app_window = SapApplicationWindowReader()
    
    if not search_str:
        search_str = sg.PopupGetText('Enter The Search Word:', keep_on_top=True, no_titlebar=True, default_text='project')
                
    app_window.set_search_str(search_str)
    app_window.find_target_window_by_search_str()
    app_window.save_screenshot(image_exporter_name='pil')
    


if __name__ == '__main__':
    save_project_builder_window()    
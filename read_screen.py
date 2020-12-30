# %%
'''
R@yth30n2020_Nov
'''

import re
import cv2
import pytesseract
import numpy as np
import pandas as pd
from PIL import Image
import PySimpleGUI as sg
from copy import deepcopy
import run_wbs_update_tool
from collections import defaultdict
from app_window_class import SapApplicationWindowReader
from common_functions import get_universal_path
# %%




def set_image_dpi(file_path):
    import tempfile

    im = Image.open(file_path)
    length_x, width_y = im.size
    factor = min(1, float(1024.0 / length_x))
    size = int(factor * length_x), int(factor * width_y)
    im_resized = im.resize(size, Image.ANTIALIAS)
    temp_file = tempfile.NamedTemporaryFile(delete=False,   suffix='.png')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(300, 300))
    return temp_filename



class ReadSapProjectImage():
    def __init__(self):
        self.img_file = None
        self.img = None
        self.text = None
        self.activity_num = None
        self.wbs_element = None
        self.nwa_charge_num = None

        self.activity_nums = []
        self.wbs_elements = []
        self.nwa_charge_nums = []

        self.wbs_regex_pattern = r"w[a-z]{1}s element [a-z0-9-]*"




    def set_pytesseract_path(self, path):
        self.pytesseract_path = path

    def init_pytesseract(self):
        while True:
            try:
                pytesseract.pytesseract.tesseract_cmd = self.pytesseract_path
                break
            except:
                self.set_pytesseract_path( get_universal_path(sg.PopupGetFile('Select The Location Of PyTesseract.exe', keep_on_top=True)))
                continue



    def set_img_file(self, img_file):
        self.img_file = img_file


    def import_image(self):
        self.img = Image.open(self.img_file)
        self.img = cv2.imread(self.img_file)


    def read_image(self):
        self.image_text = pytesseract.image_to_string(self.img, lang='eng', config='--psm 6')
        self.text = pytesseract.image_to_string(self.img, lang = 'eng')
        
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.gray_text = pytesseract.image_to_string(self.gray, lang='eng', config='--psm 6')
        
        
        self.blur = cv2.GaussianBlur(self.gray, (3,3), 0)
        self.thresh = cv2.threshold(self.blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        self.thresh_text = pytesseract.image_to_string(self.thresh, lang='eng', config='--psm 6')

        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        self.opening = cv2.morphologyEx(self.thresh, cv2.MORPH_OPEN, self.kernel, iterations=1)
        self.opening_text = pytesseract.image_to_string(self.opening, lang='eng', config='--psm 6')

        self.invert = 255 - self.opening
        self.inverted_text = pytesseract.image_to_string(self.invert, lang='eng', config='--psm 6')


    def init_extracted_text_dict(self):
        self.extracted_text = {'img_reader.inverted_text' : self.inverted_text,
                            'img_reader.thresh_text' : self.thresh_text,
                            'img_reader.image_text' : self.image_text,
                            'img_reader.opening_text' : self.opening_text,
                            'img_reader.gray_text' : self.gray_text}





    def search_img_for_text(self):
        for _ in self.text.split('\n'):

            if 'Activity'.lower() in _.lower() and 'WBS Element'.lower() in _.lower():
                wbs_idx = _.find('WBS Element')
                self.wbs_element =  _[wbs_idx:]
                self.activity_num = _[:wbs_idx]
                break



# if __name__ == '__main__':

if __name__ == '__main__':
    search_str = 'project'
    app_window = SapApplicationWindowReader()
    app_window.set_search_str(search_str)
    app_window.find_target_window_by_search_str()
    app_window.save_screenshot(image_exporter_name='pil')

    img_reader = ReadSapProjectImage()
    img_reader.set_pytesseract_path(r"C:\Users\1155449\AppData\Local\Programs\Tesseract-OCR\tesseract.exe")
    img_reader.set_img_file( app_window.out_file )
    # img_reader.img_file = set_image_dpi(img_reader.img_file)

    img_reader.import_image()
    img_reader.read_image()
    img_reader.init_extracted_text_dict()

# %%

all_text = '****************************************End****************************************\n\n\n\n****************************************Start****************************************\n'.join(f'({k}|{_}' for k, _ in img_reader.extracted_text.items())
all_text = ''.join( ltr.lower() for ltr  in all_text if ltr.isascii() )

# %%


def init_result_dict():
    ocr_result_dict = defaultdict( list )
    ocr_result_dict['Prj Def']
    ocr_result_dict['Prj Desc']
    ocr_result_dict['WBS Element']
    ocr_result_dict['WBS Desc']
    ocr_result_dict['MP ID']
    ocr_result_dict['MP Desc']
    ocr_result_dict['Network']
    ocr_result_dict['NWA Charge Number']
    ocr_result_dict['Alternative SP1']
    ocr_result_dict['Alternative SP2']
    ocr_result_dict['Alternative SP3']
    ocr_result_dict['Phase']
    ocr_result_dict['Update Date']
    ocr_result_dict['ExtraColumn1']
    ocr_result_dict['ExtraColumn2']
    ocr_result_dict['ExtraColumn3']
    ocr_result_dict['ExtraColumn4']
    ocr_result_dict['ExtraColumn5']
    ocr_result_dict['ExtraColumn6']
    ocr_result_dict['ExtraColumn7']
    ocr_result_dict['ExtraColumn8']
    ocr_result_dict['ExtraColumn9']
    ocr_result_dict['ExtraColumn10']
    return ocr_result_dict



def get_db_df():
    import pyodbc
    import pandas as pd
    import wbs_settings

    wbs_settings.init()
    wbs_settings.config
    db = wbs_settings.config["wbs_settings"]["database"]
    schema = wbs_settings.config["wbs_settings"]["schema"] 
    table = wbs_settings.config["wbs_settings"]["wbs_table"]

    cnxn = pyodbc.connect(f'DSN={wbs_settings.config["wbs_settings"]["server"]}')
    df = pd.read_sql(f'select * from [{db}].[{schema}].[{table}]', con=cnxn)
    cnxn.close()
    return df



def get_prj_def_from_ngroup(wbs):
    prj_def = 'NG'

    if len(wbs.split('-')) == 3:
        prj_def += wbs.split('-')[1].upper()
    elif len(wbs.split('-')) >= 4:
        prj_def += wbs.split('-')[2].upper()

    prj_def += wbs.split('-')[1].upper()
    prj_def += wbs.split('-')[-1].upper()
    return prj_def


def init_regex_dict():
    regex_dict = defaultdict( list )
    regex_dict['Prj Def']
    regex_dict['Prj Desc']
    regex_dict['WBS Element']
    regex_dict['Network']
    regex_dict['NWA Charge Number']
    return regex_dict

# %%
def regex_match_with_dict(cur_key, regex_dict, ocr_result_dict, search_text):
    for search_exp in regex_dict[cur_key]:
        try:
            for _ in re.findall( search_exp, search_text):
                ocr_result_dict[cur_key].append(_)
        except:
            pass
    ocr_result_dict[cur_key] = list(set( ocr_result_dict[cur_key] ))

# %%

def init_fuzzy_lookup_dict():
    fuzzy_lookup_dict = defaultdict( list )
    fuzzy_lookup_dict['Prj Def'] = ['project', 'proj def', 'pdef']
    fuzzy_lookup_dict['Prj Desc'] = ['project', 'descrtption']
    fuzzy_lookup_dict['WBS Element'] = ['wbs', 'wbs element', 'element']
    fuzzy_lookup_dict['Network'] = ['network', 'activity']
    fuzzy_lookup_dict['NWA Charge Number'] = ['network', 'activity']
    return fuzzy_lookup_dict

def init_fuzzy_dict():
    fuzzy_dict = defaultdict( list )
    fuzzy_dict['Prj Def'] = ['project', 'proj def', 'pdef']
    fuzzy_dict['Prj Desc'] = ['project', 'descrtption']
    fuzzy_dict['WBS Element'] = ['wbs', 'wbs element', 'element']
    fuzzy_dict['Network'] = ['network', 'activity']
    fuzzy_dict['NWA Charge Number'] = ['network', 'activity']
    return fuzzy_dict

def get_fuzzy_matches(fuzzy_dict, fuzzy_lookup_dict):
    from fuzzywuzzy import process

    for k, lookup_list in fuzzy_lookup_dict.items():
        for keyword in lookup_list:
            res = process.extract(keyword, all_text.split('\n'))
            for _ in res:
                fuzzy_dict[k].append(_[0])
    return fuzzy_dict


def add_regex_expressions(regex_dict):
    regex_dict['WBS Element'].append(r"(?<=element\s)[a-z0-9\-]{9,}(?=\s|\n)" )
    regex_dict['WBS Element'].append(r"[a-z0-9]{6}-.{1}-[a-z0-9]{4}")
    regex_dict['WBS Element'].append(r"(?<=wbs element )[a-z0-9]{8}(?=\s|\n])")
    regex_dict['WBS Element'].append(r"(?<=wbs\selement\s).*[a-z0-9]{8}(?=\s|\n])")
    regex_dict['WBS Element'].append(r"(?<=wbs\selement\s_)[a-z0-9]{8}(?=\s|\n])")

    regex_dict['Network'].append(r"(?<=network\s)[a-z0-9]{9}(?=\s|\n)" )
    regex_dict['Network'].append(r"(?<=network\s)[a-z0-9]{9}(?![a-z|0-9])" )
    regex_dict['Network'].append(r"(?<=network.)[a-z0-9]{9}")
    regex_dict['Network'].append(r"(?<=network.)[a-z\d.]{9}")
    regex_dict['Network'].append(r"(?<=network).*[a-z\d.]{5,9}")
    regex_dict['Network'].append(r"(?<=activity.).*.\d{5,9}")
    regex_dict['Network'].append(r"(?<=activity.).*.\d{5,9}(?=\s|\n)")
    regex_dict['Network'].append(r"\d{9}")


    regex_dict['NWA Charge Number'].append( r"[a-z\d]{9}\s[a-z\d]{4}" )
    regex_dict['NWA Charge Number'].append( r"[a-z\d]{9,}\s[a-z\d.]{4,}" )
    regex_dict['NWA Charge Number'].append(r"(?<=network).*[a-z\d.]{5,}")
    regex_dict['NWA Charge Number'].append(r"(?<=activity.).*.\d{5,}")

    regex_dict['Prj Def'].append(r"(?<=project def\.\s(\[||)+)[a-z0-9]{4}")
    regex_dict['Prj Def'].append(r"(?<=project\s)[a-z0-9]{8}(?=\s|\n)")
    regex_dict['Prj Def'].append(r"(?<=project\sdef\.\s\[)[a-z0-9]{4}")
    return regex_dict

def lcs(s1, s2):
    matrix = [["" for x in range(len(s2))] for x in range(len(s1))]
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                if i == 0 or j == 0:
                    matrix[i][j] = s1[i]
                else:
                    matrix[i][j] = matrix[i-1][j-1] + s1[i]
            else:
                matrix[i][j] = max(matrix[i-1][j], matrix[i][j-1], key=len)
    cs = matrix[-1][-1]
    return len(cs), cs




def get_random_prj_def(db_df):
    return db_df.iloc[ np.random.randint(0, len(db_df.index)) , :]['Prj Def']
    

def lcs_compare_ocr(ocr_result_dict):
    temp_ocr = deepcopy(ocr_result_dict)

    for cur_key in ['WBS Element', 'Network', 'NWA Charge Number', 'Prj Def']:
        for s1 in temp_ocr[ cur_key]:
            for s2 in temp_ocr[ cur_key]:
                t_res = lcs(s1, s2)[1]
                if cur_key == 'Network' and len(t_res) == 9:
                    ocr_result_dict[ cur_key].append(t_res)
            
        ocr_result_dict[ cur_key] = list(set(ocr_result_dict[ cur_key]))
        return ocr_result_dict
def filter_network_results(ocr_result_dict):
    res = []
    for _ in ocr_result_dict['Network']:
        if len(_) == 9:
            res.append(_)

    res = list(set(res))
    ocr_result_dict['Network'] = res
    return ocr_result_dict

def filter_nwa_results(ocr_result_dict):
    res = []
    for _ in ocr_result_dict['NWA Charge Number']:
        if len(_) == 14:
            res.append(_)

    res = list(set(res))
    ocr_result_dict['NWA Charge Number'] = res
    return ocr_result_dict

# %%

# def fuzzy_extract(qs, ls, threshold):
#     from fuzzysearch import find_near_matches
#     from fuzzywuzzy import process
#     '''fuzzy matches 'qs' in 'ls' and returns dict of results[index].[words]
#     '''
#     results = defaultdict(list)

#     for word, _ in process.extractBests(qs, (ls,), score_cutoff=threshold):
#         for match in find_near_matches(qs, word, max_l_dist=1):
#             match = word[match.start:match.end]

#             index = ls.find(match)
#             # results[index].append(match)
#             results[qs].append(ls[index:index+len(match)+10])
#     return results


# cur_key = 'WBS Element'
# for _ in fuzzy_lookup_dict[cur_key]:
#     res = fuzzy_extract(qs=_.lower(), ls=all_text, threshold=0.85)
# res

# %%

fuzzy_dict = init_fuzzy_dict()
fuzzy_lookup_dict = init_fuzzy_lookup_dict()
fuzzy_dict = get_fuzzy_matches(fuzzy_dict, fuzzy_lookup_dict)






ocr_result_dict = init_result_dict()
regex_dict = init_regex_dict()
regex_dict = add_regex_expressions(regex_dict)

cur_key = 'WBS Element'
regex_match_with_dict(cur_key=cur_key, regex_dict=regex_dict, ocr_result_dict=ocr_result_dict, search_text=all_text)
regex_match_with_dict(cur_key=cur_key, regex_dict=regex_dict, ocr_result_dict=ocr_result_dict, search_text=fuzzy_dict[cur_key])


cur_key = 'Network'
regex_match_with_dict(cur_key=cur_key, regex_dict=regex_dict, ocr_result_dict=ocr_result_dict, search_text=all_text)
regex_match_with_dict(cur_key=cur_key, regex_dict=regex_dict, ocr_result_dict=ocr_result_dict, search_text=fuzzy_dict[cur_key])




cur_key = 'NWA Charge Number'
regex_match_with_dict(cur_key=cur_key, regex_dict=regex_dict, ocr_result_dict=ocr_result_dict, search_text=all_text)
ocr_result_dict['NWA Charge Number'] = [_ for _ in ocr_result_dict['NWA Charge Number'] if _.split(' ')[0] in ocr_result_dict['Network']]
regex_match_with_dict(cur_key=cur_key, regex_dict=regex_dict, ocr_result_dict=ocr_result_dict, search_text=fuzzy_dict[cur_key])




cur_key = 'Prj Def'
regex_match_with_dict(cur_key=cur_key, regex_dict=regex_dict, ocr_result_dict=ocr_result_dict, search_text=all_text)
regex_match_with_dict(cur_key=cur_key, regex_dict=regex_dict, ocr_result_dict=ocr_result_dict, search_text=fuzzy_dict[cur_key])






for wbs in ocr_result_dict['WBS Element']:
    if 'ocr_result_dict' in wbs.lower():
        ocr_result_dict['Prj Def'].append( get_prj_def_from_ngroup(wbs) )


ocr_result_dict = lcs_compare_ocr(ocr_result_dict)
ocr_result_dict = filter_nwa_results(ocr_result_dict)
ocr_result_dict = filter_network_results(ocr_result_dict)



# %%
db_df = deepcopy( get_db_df())


# %%
ocr_df = pd.DataFrame.from_dict(ocr_result_dict, orient='index').T

# %%
import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication

from run_wbs_update_tool import Ui_MainWindow

_cache = []
title = 'WBS Update Tool'
app = QApplication(sys.argv)

if QApplication.instance() is None:
    _cache.append(app)

wbs_update_form = Ui_MainWindow()

wbs_update_form.set_ocr_df(ocr_df)
wbs_update_form.set_qapplication(app)
wbs_update_form.setWindowTitle(title)
wbs_update_form.setAttribute(QtCore.Qt.WA_DeleteOnClose)
_cache.append(wbs_update_form)
wbs_update_form.populate_dropdowns()
wbs_update_form.show()





# %%


# %%

# %%




# %%




# %%








# # %%
# gray_image = cv2.cvtColor(img_reader.img, cv2.COLOR_BGR2GRAY)

# threshold_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# cv2.imshow('threshold_img', threshold_img)


# cv2.waitKey(0)

# cv2.destroyAllWindows()


# # %%





















# # %%

# def calibrate_cv2_lower_and_upper_range(image_file):
#     import cv2
#     import numpy as np

#     def nothing(x):
#         pass

#     cv2_upper_lower_range_dict = {}
#     # Load image
#     image = cv2.imread(image_file)

#     # Create a window
#     cv2.namedWindow('image')

#     # Create trackbars for color change
#     # Hue is from 0-179 for Opencv
#     cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
#     cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
#     cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
#     cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
#     cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
#     cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

#     # Set default value for Max HSV trackbars
#     cv2.setTrackbarPos('HMax', 'image', 179)
#     cv2.setTrackbarPos('SMax', 'image', 255)
#     cv2.setTrackbarPos('VMax', 'image', 255)

#     # Initialize HSV min/max values
#     hMin = sMin = vMin = hMax = sMax = vMax = 0
#     phMin = psMin = pvMin = phMax = psMax = pvMax = 0

#     while(1):
#         # Get current positions of all trackbars
#         hMin = cv2.getTrackbarPos('HMin', 'image')
#         sMin = cv2.getTrackbarPos('SMin', 'image')
#         vMin = cv2.getTrackbarPos('VMin', 'image')
#         hMax = cv2.getTrackbarPos('HMax', 'image')
#         sMax = cv2.getTrackbarPos('SMax', 'image')
#         vMax = cv2.getTrackbarPos('VMax', 'image')

#         # Set minimum and maximum HSV values to display
#         lower = np.array([hMin, sMin, vMin])
#         upper = np.array([hMax, sMax, vMax])

#         # Convert to HSV format and color threshold
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#         mask = cv2.inRange(hsv, lower, upper)
#         result = cv2.bitwise_and(image, image, mask=mask)

#         # Print if there is a change in HSV value
#         if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
#             print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
#             phMin = hMin
#             psMin = sMin
#             pvMin = vMin
#             phMax = hMax
#             psMax = sMax
#             pvMax = vMax

#         # Display result image
#         cv2.imshow('image', result)
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break

#     cv2.destroyAllWindows()
#     cv2_upper_lower_range_dict['lower'] = (hMin, sMin, vMin)
#     cv2_upper_lower_range_dict['upper'] = (hMax, sMax, vMax)
#     return cv2_upper_lower_range_dict
    

# cv2_upper_lower_range_dict = calibrate_cv2_lower_and_upper_range(img_reader.img_file)



# # %%
# import cv2
# import numpy as np
# import pytesseract

# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\1155449\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# # Load image, convert to HSV format, define lower/upper ranges, and perform
# # color segmentation to create a binary mask
# image = cv2.imread(img_reader.img_file)
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# mask = cv2.inRange(hsv, cv2_upper_lower_range_dict['lower'], cv2_upper_lower_range_dict['upper'])

# # Create horizontal kernel and dilate to connect text characters
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
# dilate = cv2.dilate(mask, kernel, iterations=5)

# # Find contours and filter using aspect ratio
# # Remove non-text contours by filling in the contour
# cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# for c in cnts:
#     x,y,w,h = cv2.boundingRect(c)
#     ar = w / float(h)
#     if ar < 5:
#         cv2.drawContours(dilate, [c], -1, (0,0,0), -1)

# # Bitwise dilated image with mask, invert, then OCR
# result = 255 - cv2.bitwise_and(dilate, mask)
# data = pytesseract.image_to_string(result, lang='eng',config='--psm 6')
# print(data)

# cv2.imshow('mask', mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('dilate', dilate)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('result', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # %%



# # from pytesseract import Output


# # custom_config = r'--oem 3 --psm 6'
# # _
# # details = pytesseract.image_to_data(threshold_img, output_type=Output.DICT, config=custom_config, lang='eng')

# # total_boxes = len(details['text'])


# # for sequence_number in range(total_boxes):
# # 	if int(details['conf'][sequence_number]) >30:
# # 		(x, y, w, h) = (details['left'][sequence_number], details['top'][sequence_number], details['width'][sequence_number],  details['height'][sequence_number])
# # 		threshold_img = cv2.rectangle(threshold_img, (x, y), (x + w, y + h), (0, 255, 0), 2)


# # cv2.imshow('captured text', threshold_img)
# # # Maintain output window until user presses a key
# # cv2.waitKey(0)
# # # Destroying present windows on screen
# # cv2.destroyAllWindows()
# # %%

# import easyocr
# reader = easyocr.Reader(['en'], gpu=1) # need to run only once to load model into memory

# # %%



# # # %%
# # reader.model_storage_directory = r"C:\Users\1155449\EasyOCR\model"

# # result = reader.readtext(app_window.out_file, gpu)
# # result

# # # %%
# # result
# # # %%
# # result[0]
# # # %%
# # result.text_threshold 
# # # %%

# # %%

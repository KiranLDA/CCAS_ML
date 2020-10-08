#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 11:12:06 2020

@author: kiran
"""


# /media/kiran/D0-P1/animal_data/meerkat/preprocessed/test_data/pred_table/HM_VCVM001_HMB_AUDIO_R08_ file_2_(2017_08_03-06_44_59)_ASWMUX221153_PRED_TABLE.txt",
              # /home/kiran/Desktop/HM_VHMM016_LTTB_R29_20190707-20190719_file_7_(2019_07_13-11_44_59)_135944_PRED_TABLE.txt
import pandas as pd

# test = pd.read_table("/media/kiran/D0-P1/animal_data/meerkat/preprocessed/test_data/label_table/HM_VCVM001_HMB_AUDIO_R08_ file_2_(2017_08_03-06_44_59)_ASWMUX221153_LABEL_TABLE.txt",sep=";")
# test.loc[254,]
# booli = test["cc"]
# booli == True # works
# booli == "True" #doesn't work, so its not importing these as strings


# test = pd.read_table("/media/kiran/D0-P1/animal_data/meerkat/preprocessed/test_data/label_table/HM_VCVM001_HMB_AUDIO_R08_ file_2_(2017_08_03-06_44_59)_ASWMUX221153_LABEL_TABLE.txt",sep=";")
# test = test.drop_duplicates()
# test = test[:-1]

# test["cc"].astype('bool')




folder = "/media/kiran/D0-P1/animal_data/meerkat/preprocessed/test_data/label_table"
for file in glob.glob(os.path.join(folder, "*.txt")):
    # print(file) 
    # file=glob.glob(os.path.join(folder, "*.txt"))[0]
    try:
        test = pd.read_table(file, sep=";")
        test = test.drop_duplicates()
        test = test[:-1]
        file = os.path.basename(file)
        test.to_csv(folder+"/new/"+file, header=True, index=None, sep=';')
    except:
       # pass
        try:
            test = pd.read_table(file, sep=" ")
            test = test.drop_duplicates()
            # test = test[:-1]
            file = os.path.basename(file)
            test.to_csv(folder+"/new/"+file, header=True, index=None, sep=';')
        except:
            pass


folder = "/media/kiran/D0-P1/animal_data/meerkat/preprocessed/test_data/pred_table"
for file in glob.glob(os.path.join(folder, "*.txt")):
    # print(file) 
    # file=glob.glob(os.path.join(folder, "*.txt"))[0]
    try:
        test = pd.read_table(file, sep=";")
        test = test.drop_duplicates()
        test = test[:-1]
        file = os.path.basename(file)
        test.to_csv(folder+"/new/"+file, header=True, index=None, sep=';')
    except:
       # pass
        try:
            test = pd.read_table(file, sep=" ")
            test = test.drop_duplicates()
            # test = test[:-1]
            file = os.path.basename(file)
            test.to_csv(folder+"/new/"+file, header=True, index=None, sep=';')
        except:
            pass

        
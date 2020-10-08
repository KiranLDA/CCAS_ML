import os
import pandas as pd
import shutil


def fetch_files_with_numcalls(base_dir, min_num_of_calls):
    """
    Returns a list of files from the base audio directory with given minimum number of hyena calls
    The file list consists of 'file_identifier', 'filename', 'file_type' (wav, text).
    """
    file_list = pd.DataFrame(columns=['file_identifier', 'filename', 'file_type'])
    for dir, _subdirs, files in os.walk(base_dir):
        for fname in files:
            try:
                [file_id, file_type] = fname.rsplit(sep="_", maxsplit=1)
                file_type = file_type.split(".")[0]
                file_list.loc[len(file_list)] = [file_id, fname, file_type]
            except:
                print(fname + "cannot be split into 2")
    file_list = file_list.pivot(index='file_identifier', columns='file_type', values='filename')
    file_list['num_of_calls'] = file_list.apply(lambda row: len(open(base_dir + '/' + row.labels, 'rb').readlines()), axis=1)
    return file_list[file_list['num_of_calls'] >= min_num_of_calls]



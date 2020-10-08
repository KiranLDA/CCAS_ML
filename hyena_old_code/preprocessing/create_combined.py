import os
import numpy as np
import sys

LOAD_PATH = ['/cache/rmishra/cc16_366a_converted/spectro',
             '/cache/rmishra/cc16_352a_converted/spectro',
             '/cache/rmishra/cc16_352b_converted/spectro']

SAVE_PATH = '/cache/rmishra/combined'

if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

for dir in LOAD_PATH:
    for f in os.listdir(dir):
        if 'LABEL' in f and os.path.splitext(f)[-1].lower() == ".npy":
            label = np.load(os.path.join(dir,f))
            if 1 in label[0:8,:]:
                spectro = np.load(os.path.join(dir, f.split('LABEL')[0]+'.npy'))
                combined = np.array((spectro, label))
                np.save(os.path.join(SAVE_PATH, f.split('LABEL')[0]+'SPEC_LAB.npy'), combined)


LOAD_PATH = '/cache/rmishra/combined'
SAVE_PATH = '/cache/rmishra/datasets'
SAVE_FILE = '1plus_dataset.npy'
files = os.listdir(LOAD_PATH)

dataset = []
for i in range(len(files)):
    current_file = np.load(os.path.join(LOAD_PATH, files[i]))
    dataset.append([files[i],current_file])

dataset = np.array(dataset)
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)
np.save(os.path.join(SAVE_PATH, SAVE_FILE), dataset)
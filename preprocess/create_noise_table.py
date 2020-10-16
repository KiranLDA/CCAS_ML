#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:54:48 2020

@author: kiran
"""

import pandas as pd

def create_noise_table(label_table):
    noise_table = pd.DataFrame()
    noise_table["Start"] = label_table.loc["Stop"][0:(len(label_table["Stop"])-1)]
    noise_table["Stop"] = label_table.loc["Start"][1:len(label_table["Stop"])]
    noise_table["Label"] = "Noise"
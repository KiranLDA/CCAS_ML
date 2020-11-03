#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 14:49:17 2020

@author: kiran
"""


import json


class InitialiseParams():
    def __init__(self, file):
        self.file = file
        try:
            with open(self.file) as f:
                variables = json.load(f)
        except json.JSONDecodeError as err:
            raise ValueError(f"Error is occuring on line {err.lineno}, column {err.colno} in file {self.file} \nError:  {err.msg}")
        for key, value in variables.items():
            setattr(self, key, value)
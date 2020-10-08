#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 14:45:26 2020

@author: mathieu
"""


class Evaluate:
    def __init__(self, model_to_evaluate, dataset_dir, species, is_three_axis, batch_size, threshold):
        self.model = load_model(model_to_evaluate)

def main(self):
        if self.is_three_axis:
            preds_w_threshold = self.model.predict([self.x_test_aud, self.x_test_acc_ch0, self.x_test_acc_ch1, self.x_test_acc_ch2], self.batch_size)
        else:
            preds_w_threshold = self.model.predict([self.x_test_aud, self.x_test_acc_ch2], self.batch_size)
# -*- coding: utf8 -*-

import csv

import numpy as np
import pandas as pd
import cv2
from PIL import Image
from random import randint

from sys_config import SysConfig

class DataProcessing(object):

	def __init__(self):
		self.sys_obj = SysConfig()
		# pass
		## print(tf.__version__)


	def cast_all_to_numeric(self, input_df):
		'''
		Input:
			input_df: dataframe as input data.
		Output:
			all_filter_flag: [true:meet filter, false:not meet filter]
		'''

		input_df = input_df.apply(pd.to_numeric, errors='coerce')
		# ## good performance
		input_df = input_df.fillna(0)
		return input_df
	def cast_all_to_numeric_bak(self, input_df):
		'''
		Input:
			input_df: dataframe as input data.
		Output:
			all_filter_flag: [true:meet filter, false:not meet filter]
		'''

		field_ary = list(input_df.columns.values)
		# #### column type casting
		temp_count = 1
		for temp_field in field_ary:
			input_df[temp_field]= pd.to_numeric(input_df[temp_field], errors='coerce')

			temp_count +=1

			if temp_count % 1000 == 1:
				print("field_count= %s"%(temp_count))
			## bad performance
			# # input_df[temp_field] = input_df[temp_field].fillna(0)

		## good performance
		input_df = input_df.fillna(0)
		return input_df

	def cast_to_numeric(self, input_df, field_ary):
		'''
		Input:
			input_df: dataframe as input data.
			field_ary: target field array.
		Output:
			all_filter_flag: [true:meet filter, false:not meet filter]
		'''

		#### column type casting
		for temp_field in field_ary:
			input_df[temp_field]= pd.to_numeric( input_df[temp_field], errors='coerce')
			input_df[temp_field] = input_df[temp_field].fillna(0)

		return input_df

	def img_balance_augmentation(self, data_ary):
		'''
		Rotate the img randomly for data augmentation, base on the label for class balance.
		'''
		y_list = [each_data[1] for each_data in data_ary]
		y_ary = np.array(y_list)
		unique, counts = np.unique(y_ary, return_counts = True)
		min_index = np.argmin(counts)
		max_index = np.argmax(counts)
		# print('min_index:', min_index)
		minor_class = unique[min_index]
		major_class = unique[max_index]
		print('Major class:', major_class)
		print('minor class:', minor_class)
		print(dict(zip(unique, counts)))
		r = counts[max_index]/counts[min_index]
		print('r:',r)
		
		augmented_data_ary = []
		for each_data in data_ary:
			pixel_ary = each_data[0]
			label = each_data[1]

			img_height = pixel_ary.shape[0]
			img_width = pixel_ary.shape[1]
			# print('img w/h', img_width, img_height)
			M = cv2.getRotationMatrix2D((img_width/2, img_height/2), randint(-5, 5), 1) # ((center_x, center_y), angle, scale)
			rotated_pixel_ary = cv2.warpAffine(pixel_ary, M, (img_width, img_height))

			augmented_data_ary.append([rotated_pixel_ary, label])
			if label == minor_class:
				coda = r
				while(coda>2): # Check rotated物件 是否有被overwrite
					rotated_pixel_ary_v = cv2.warpAffine(pixel_ary, M, (img_width, img_height))
					augmented_data_ary.append([rotated_pixel_ary_v, label])
					coda -= 1
		augmented_data_ary = np.array(augmented_data_ary)
		
		return augmented_data_ary

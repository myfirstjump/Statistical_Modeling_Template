# -*- coding: utf8 -*-
# import re
# import time
# import csv
# import sys
# import math
# import os
# import itertools
import unittest
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sys_config import SysConfig
from CytoOA_data_main import CytoOADataMain
from CytoOA_training_main import CytoOATrainingMain
from data_processing import DataProcessing

class UnitTestSeqTools(unittest.TestCase):

	def setUp(self):
		# print("In setUp ...")
		self.data_obj = CytoOADataMain()
		self.env_obj = SysConfig()
		self.train_obj = CytoOATrainingMain()
		self.process_obj = DataProcessing()

	def tearDown(self):
		# print("In tearDown ...")
		pass

	# def test_cyto_keras_training_pipeline(self):
	# 	print("In test_cyto_keras_training_pipeline ... ")

	# 	### probe bind directory
	# 	# # data_dir = '/home/ryan/src_dir/CytoCloudR/tmp/gene/output'
	# 	# # data_dir = '/home/ryan/src_dir/CytoCloudR/tmp/gene/test_output'
	# 	# data_dir = self.env_obj.get_cnv_output_dir()

	# 	### report excel
	# 	# outcome_file = '/home/ryan/src_dir/CytoOA_AI/data/Cyto_Report_summary2.xlsx'
	# 	# outcome_file = self.env_obj.get_outcome_file()

	# 	## read data
	# 	# cnv_df = self.data_obj.build_cnv_training_data(data_dir, outcome_file)
	# 	# print(cnv_df)

	# 	# print(cnv_df.describe())
	# 	# print(cnv_df.duplicated())
	# 	### DNN
	# 	# self.train_obj.cyto_dnn_balance_training_pipeline(cnv_df)
	
	# def test_cyto_tif_training_pipeline(self):
	# 	# # read tif path and cnv_outcome
	# 	# img_data_dir = self.env_obj.get_tif_data_dir()
	# 	# img_path_df = self.data_obj.get_tif_data_table(img_data_dir, 'local_tiff_match_file_list.txt')

	# 	# with open('data_df.pickle','rb') as file: # pickle for fast testing
	# 	# 	cnv_df = pickle.load(file)
	# 	# img_path_df = pd.merge(left=img_path_df, right=cnv_df.loc[:, ['Array_ID', 'cnv_outcome']]) ## img_path_df 包含X路徑、Y
		
	# 	# print(img_path_df.head())
	# 	# print(img_path_df.shape)
	# 	# # # read tif pixel value
	# 	# data_ary = self.data_obj.get_img_ary(img_path_df)
	# 	# np.save('tif_ary_(1220x432)_0625.npy', data_ary)
	# 	data_ary = np.load('tif_ary_(1220x432)_0625.npy') # for fast testing 'tif_ary_(610x216)_0615.npy'
	# 	(train_data_ary, test_data_ary) = train_test_split(data_ary, test_size=0.3)
	# 	# augmented_data = self.process_obj.img_balance_augmentation(train_data_ary)
	# 	# data = data[:50]

	# 	### CNN
	# 	self.train_obj.cyto_cnn_training_pipeline(train_data_ary, test_data_ary)

	# 	# ### Vgg-net
	
	def test_cyto_R_CBS_training_pipeline(self):
		#	Read CBS plot and cnv_outcome
		img_data_dir = self.env_obj.get_tif_data_dir()
		img_path_df = self.data_obj.get_tif_data_table(img_data_dir, 'cbs_all_file_list.txt')
		print("CBS all data is loaded. \n")
		with open('data_df.pickle','rb') as file: # pickle for fast testing
			cnv_df = pickle.load(file)
		img_path_df = pd.merge(left=img_path_df, right=cnv_df.loc[:, ['Array_ID', 'cnv_outcome']]) ## img_path_df 包含X路徑、Y
		
		print('Shape of the raw data records:', img_path_df.shape, '\n')

		data_ary = self.data_obj.get_img_ary(img_path_df)
		np.save('CBS_ary_(1150x400)_0801.npy', data_ary)
	# def test_cyto_ai_training_pipeline(self):
	# 	print("In cyto_ai_training_pipeline ... ")

	# 	### probe bind directory
	# 	# # data_dir = '/home/ryan/src_dir/CytoCloudR/tmp/gene/output'
	# 	# # data_dir = '/home/ryan/src_dir/CytoCloudR/tmp/gene/test_output'
	# 	data_dir = self.env_obj.get_cnv_output_dir()

	# 	### report excel
	# 	# outcome_file = '/home/ryan/src_dir/CytoOA_AI/data/Cyto_Report_summary2.xlsx'
	# 	outcome_file = self.env_obj.get_outcome_file()

	# 	### read data
	# 	# cnv_df = self.data_obj.build_cnv_training_data(data_dir, outcome_file)

	# 	### training main pipeline
	# 	## normal version
	# 	# self.train_obj.cyto_ai_training_pipeline(cnv_df)

	# 	## randomforest
	# 	# self.train_obj.cyto_ai_balance_training_pipeline(cnv_df)

	# 	### log df version
	# 	# # self.train_obj.cyto_ai_training_pipeline_log_df(cnv_df)


if __name__ == "__main__":
	unittest.main()

# -*- coding: utf8 -*-
import re
import time
import csv
import sys
import math
import os

import pandas as pd
import numpy as np

from sklearn.externals import joblib
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import pickle

from data_processing import DataProcessing
from data_sampling import DataSampling
from sys_config import SysConfig
from data_trainning import DataTrainning

# from file_list import FileList
# from Excel_Reader import ExcelReader
# from data_reader import DataReader

class CytoOATrainingMain(object):

	def __init__(self):
		pass

	def cyto_xgboost_balance_training_pipeline(self, data_df):

		data_processing_obj = DataProcessing()
		sampling_obj = DataSampling()
		sys_obj = SysConfig()
		train_obj = DataTrainning()

		test_set_ratio = sys_obj.get_test_set_ratio()
		y_key = sys_obj.get_y_key()

		model_output_dir = sys_obj.get_model_output_dir()
		log_file = sys_obj.get_log_file()

		model_threshold = 200
		model_count = 1
		print(data_df)
		#### data preprocessing
		print("start cast_all_to_numeric.")
		data_df = data_processing_obj.cast_all_to_numeric(data_df)
		print("end cast_all_to_numeric.")
		# print("########################")
		# print(data_df)

		### convert to category
		# data_df[y_key] = data_df[y_key].astype('category')

		### sampling

		disease_df = data_df.loc[data_df[y_key]==1,:]
		normal_df = data_df.loc[data_df[y_key]==0,:]

		# # log_file = '/app/data/model/RF_3000_log.txt'
		# fh_writer = open(log_file, 'w')

		# while model_count < model_threshold:
			# (train_set, train_label, test_set, test_label) = sampling_obj.category2_sampling_pipeline(normal_df, disease_df,y_key, test_set_ratio)
		(train_set, test_set) = sampling_obj.category2_simple_sampling_pipeline(normal_df, disease_df,y_key, test_set_ratio)

		print("###############################")
		print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
		print(model_count)
		# print("######################## train_set")
		# print(train_set)

		# print("######################## test_set")
		# print(test_set)

		# #### temp data
		test_set_bak = test_set.copy()
		# # test_set_bak[y_key] = test_set_bak[y_key].astype('int')

		### get x,y
		(train_set_x, train_set_y) = sampling_obj.get_x_y_from_dataframe(train_set,y_key)
		(test_set_x, test_set_y) = sampling_obj.get_x_y_from_dataframe(test_set,y_key)

		# print("######################## train_set_x")
		# print(train_set_x)
		# print("######################## train_set_y")
		# print(train_set_y)

		# print("######################## test_set_x")
		# print(test_set_x)
		# print("######################## test_set_y")
		# print(test_set_y)

		### seperate by disease
		test_set_disease = test_set_bak.loc[test_set_bak[y_key]==1,:]
		test_set_normal = test_set_bak.loc[test_set_bak[y_key]==0,:]

		(test_set_disease_x, test_set_disease_y) = sampling_obj.get_x_y_from_dataframe(test_set_disease,y_key)
		(test_set_normal_x, test_set_normal_y) = sampling_obj.get_x_y_from_dataframe(test_set_normal,y_key)

		# print("######################## test_set_disease")
		# print(test_set_disease)
		# print("######################## test_set_disease_x")
		# print(test_set_disease_x)
		# print("######################## test_set_disease_y")
		# print(test_set_disease_y)

		# print("######################## test_set_normal")
		# print(test_set_normal)
		# print("######################## test_set_normal_x")
		# print(test_set_normal_x)
		# print("######################## test_set_normal_y")
		# print(test_set_normal_y)

		test_set_disease_len = len(test_set_disease_y)
		test_set_normal_len = len(test_set_normal_y)

		print("######################## test_set_disease_len, test_set_normal_len")
		print("{}, {}".format(test_set_disease_len, test_set_normal_len))
		# ### feature list
		# # feature_dict = {}
		# # feature_dict['numeric'] = list(train_set_x.columns.values)

		### training the model
		my_model = train_obj.xgboot_training(train_set_x, train_set_y, test_set_x, test_set_y)

		# # test_set_x = preprocessing.scale(test_set_x)
		# # test_set_x = scaler.scale(test_set_x)
		# test_set_x = scaler.transform(test_set_x)


		# # test_set_y = to_categorical(test_set_y)
		# test_score = my_model.evaluate(test_set_x, test_set_y)
		# # test_set_diseas_score = my_model.score(test_set_disease_x, test_set_disease_y)
		# # test_set_normal_score = my_model.score(test_set_normal_x, test_set_normal_y)

		# print("\ntest data set, %s: %.2f%%" % (my_model.metrics_names[1], test_score[1]*100))

		# print("Test score:", test_score[0])
		# print('Test accuracy:', test_score[1])

		return data_df


	def cyto_gene_dnn_balance_training_pipeline(self, data_df):

		data_processing_obj = DataProcessing()
		sampling_obj = DataSampling()
		sys_obj = SysConfig()
		train_obj = DataTrainning()

		test_set_ratio = sys_obj.get_test_set_ratio()
		y_key = sys_obj.get_y_key()

		model_output_dir = sys_obj.get_model_output_dir()
		log_file = sys_obj.get_log_file()

		model_threshold = 10
		model_count = 1
		print(data_df)
		#### data preprocessing
		print("start cast_all_to_numeric.")
		data_df = data_processing_obj.cast_all_to_numeric(data_df)
		print("end cast_all_to_numeric.")
		# print("########################")
		# print(data_df)

		### convert to category
		# data_df[y_key] = data_df[y_key].astype('category')

		### sampling

		disease_df = data_df.loc[data_df[y_key]==1,:]
		normal_df = data_df.loc[data_df[y_key]==0,:]

		# # log_file = '/app/data/model/RF_3000_log.txt'
		# fh_writer = open(log_file, 'w')

		# while model_count < model_threshold:
			# (train_set, train_label, test_set, test_label) = sampling_obj.category2_sampling_pipeline(normal_df, disease_df,y_key, test_set_ratio)
		(train_set, test_set) = sampling_obj.category2_simple_sampling_pipeline(normal_df, disease_df,y_key, test_set_ratio)

		print("###############################")
		print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
		print(model_count)
		# print("######################## train_set")
		# print(train_set)

		# print("######################## test_set")
		# print(test_set)

		# #### temp data
		test_set_bak = test_set.copy()
		# # test_set_bak[y_key] = test_set_bak[y_key].astype('int')

		### get x,y
		(train_set_x, train_set_y) = sampling_obj.get_x_y_from_dataframe(train_set,y_key)
		(test_set_x, test_set_y) = sampling_obj.get_x_y_from_dataframe(test_set,y_key)

		# print("######################## train_set_x")
		# print(train_set_x)
		# print("######################## train_set_y")
		# print(train_set_y)

		# print("######################## test_set_x")
		# print(test_set_x)
		# print("######################## test_set_y")
		# print(test_set_y)

		### seperate by disease
		test_set_disease = test_set_bak.loc[test_set_bak[y_key]==1,:]
		test_set_normal = test_set_bak.loc[test_set_bak[y_key]==0,:]

		(test_set_disease_x, test_set_disease_y) = sampling_obj.get_x_y_from_dataframe(test_set_disease,y_key)
		(test_set_normal_x, test_set_normal_y) = sampling_obj.get_x_y_from_dataframe(test_set_normal,y_key)

		# print("######################## test_set_disease")
		# print(test_set_disease)
		# print("######################## test_set_disease_x")
		# print(test_set_disease_x)
		# print("######################## test_set_disease_y")
		# print(test_set_disease_y)

		# print("######################## test_set_normal")
		# print(test_set_normal)
		# print("######################## test_set_normal_x")
		# print(test_set_normal_x)
		# print("######################## test_set_normal_y")
		# print(test_set_normal_y)

		test_set_disease_len = len(test_set_disease_y)
		test_set_normal_len = len(test_set_normal_y)

		print("######################## test_set_disease_len, test_set_normal_len")
		print("{}, {}".format(test_set_disease_len, test_set_normal_len))
		# ### feature list
		# # feature_dict = {}
		# # feature_dict['numeric'] = list(train_set_x.columns.values)

		### training the model
		my_input_dim = 3762
		my_model,scaler = train_obj.keras_test(train_set_x, train_set_y, my_input_dim)

		# test_set_x = preprocessing.scale(test_set_x)
		# test_set_x = scaler.scale(test_set_x)
		test_set_x = scaler.transform(test_set_x)


		# test_set_y = to_categorical(test_set_y)
		test_score = my_model.evaluate(test_set_x, test_set_y)
		# test_set_diseas_score = my_model.score(test_set_disease_x, test_set_disease_y)
		# test_set_normal_score = my_model.score(test_set_normal_x, test_set_normal_y)

		print("\ntest data set, %s: %.2f%%" % (my_model.metrics_names[1], test_score[1]*100))

		print("Test score:", test_score[0])
		print('Test accuracy:', test_score[1])

		# print("Test score = {}".format(test_score))
		# print("test_set_diseas_score score = {}".format(test_set_diseas_score))
		# print("test_set_normal_score score = {}".format(test_set_normal_score))

		### log file
		# fh_writer.write("{}\t{}\t{}\n".format(test_score, test_set_diseas_score, test_set_normal_score))

		### model
		# model_file = model_output_dir + str(model_count) + '.pkl'
		# joblib.dump(my_model, model_file)
		# model_count += 1
		# fh_writer.close()

		return data_df

	def cyto_dnn_balance_training_pipeline(self, data_df):

		data_processing_obj = DataProcessing()
		sampling_obj = DataSampling()
		sys_obj = SysConfig()
		train_obj = DataTrainning()

		test_set_ratio = sys_obj.get_test_set_ratio()
		y_key = sys_obj.get_y_key()

		model_output_dir = sys_obj.get_model_output_dir()
		log_file = sys_obj.get_log_file()

		model_threshold = 200
		model_count = 1
		print(data_df)
		#### data preprocessing
		print("start cast_all_to_numeric.")
		data_df = data_processing_obj.cast_all_to_numeric(data_df)
		print("end cast_all_to_numeric.")
		# print("########################")
		# print(data_df)

		### convert to category
		# data_df[y_key] = data_df[y_key].astype('category')

		### sampling

		disease_df = data_df.loc[data_df[y_key]==1,:]
		normal_df = data_df.loc[data_df[y_key]==0,:]

		# # log_file = '/app/data/model/RF_3000_log.txt'
		# fh_writer = open(log_file, 'w')

		# while model_count < model_threshold:
			# (train_set, train_label, test_set, test_label) = sampling_obj.category2_sampling_pipeline(normal_df, disease_df,y_key, test_set_ratio)
		(train_set, test_set) = sampling_obj.category2_simple_sampling_pipeline(normal_df, disease_df,y_key, test_set_ratio)

		print("###############################")
		print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
		print(model_count)
		# print("######################## train_set")
		# print(train_set)

		# print("######################## test_set")
		# print(test_set)

		# #### temp data
		test_set_bak = test_set.copy()
		# # test_set_bak[y_key] = test_set_bak[y_key].astype('int')

		### get x,y
		(train_set_x, train_set_y) = sampling_obj.get_x_y_from_dataframe(train_set,y_key)
		(test_set_x, test_set_y) = sampling_obj.get_x_y_from_dataframe(test_set,y_key)

		# print("######################## train_set_x")
		# print(train_set_x)
		# print("######################## train_set_y")
		# print(train_set_y)

		# print("######################## test_set_x")
		# print(test_set_x)
		# print("######################## test_set_y")
		# print(test_set_y)

		### seperate by disease
		test_set_disease = test_set_bak.loc[test_set_bak[y_key]==1,:]
		test_set_normal = test_set_bak.loc[test_set_bak[y_key]==0,:]

		(test_set_disease_x, test_set_disease_y) = sampling_obj.get_x_y_from_dataframe(test_set_disease,y_key)
		(test_set_normal_x, test_set_normal_y) = sampling_obj.get_x_y_from_dataframe(test_set_normal,y_key)

		# print("######################## test_set_disease")
		# print(test_set_disease)
		# print("######################## test_set_disease_x")
		# print(test_set_disease_x)
		# print("######################## test_set_disease_y")
		# print(test_set_disease_y)

		# print("######################## test_set_normal")
		# print(test_set_normal)
		# print("######################## test_set_normal_x")
		# print(test_set_normal_x)
		# print("######################## test_set_normal_y")
		# print(test_set_normal_y)

		test_set_disease_len = len(test_set_disease_y)
		test_set_normal_len = len(test_set_normal_y)

		print("######################## test_set_disease_len, test_set_normal_len")
		print("{}, {}".format(test_set_disease_len, test_set_normal_len))
		# ### feature list
		# # feature_dict = {}
		# # feature_dict['numeric'] = list(train_set_x.columns.values)

		### training the model
		my_model,scaler = train_obj.keras_test(train_set_x, train_set_y)

		# test_set_x = preprocessing.scale(test_set_x)
		# test_set_x = scaler.scale(test_set_x)
		test_set_x = scaler.transform(test_set_x)


		# test_set_y = to_categorical(test_set_y)
		test_score = my_model.evaluate(test_set_x, test_set_y)
		# test_set_diseas_score = my_model.score(test_set_disease_x, test_set_disease_y)
		# test_set_normal_score = my_model.score(test_set_normal_x, test_set_normal_y)

		print("\ntest data set, %s: %.2f%%" % (my_model.metrics_names[1], test_score[1]*100))

		print("Test score:", test_score[0])
		print('Test accuracy:', test_score[1])

		# print("Test score = {}".format(test_score))
		# print("test_set_diseas_score score = {}".format(test_set_diseas_score))
		# print("test_set_normal_score score = {}".format(test_set_normal_score))

		### log file
		# fh_writer.write("{}\t{}\t{}\n".format(test_score, test_set_diseas_score, test_set_normal_score))

		### model
		# model_file = model_output_dir + str(model_count) + '.pkl'
		# joblib.dump(my_model, model_file)
		# model_count += 1
		# fh_writer.close()

		return data_df

	def cyto_ai_training_pipeline_log_df(self, data_df):

		data_processing_obj = DataProcessing()
		sampling_obj = DataSampling()
		sys_obj = SysConfig()
		train_obj = DataTrainning()

		test_set_ratio = sys_obj.get_test_set_ratio()
		y_key = sys_obj.get_y_key()

		df_output_file = sys_obj.get_df_output_file()
		print(data_df)
		#### data preprocessing
		print("start cast_all_to_numeric.")
		data_df = data_processing_obj.cast_all_to_numeric(data_df)
		print(data_df)
		print("end cast_all_to_numeric.")

		###
		data_df.to_csv(df_output_file,index=False)
		print("DF output file = %s"%(df_output_file))

		return data_df

	def cyto_ai_balance_training_pipeline(self, data_df):

		data_processing_obj = DataProcessing()
		sampling_obj = DataSampling()
		sys_obj = SysConfig()
		train_obj = DataTrainning()

		test_set_ratio = sys_obj.get_test_set_ratio()
		y_key = sys_obj.get_y_key()

		model_output_dir = sys_obj.get_model_output_dir()
		log_file = sys_obj.get_log_file()

		model_threshold = 200
		model_count = 1
		print(data_df)
		#### data preprocessing
		print("start cast_all_to_numeric.")
		data_df = data_processing_obj.cast_all_to_numeric(data_df)
		print("end cast_all_to_numeric.")
		# print("########################")
		# print(data_df)

		### convert to category
		data_df[y_key] = data_df[y_key].astype('category')

		### sampling

		disease_df = data_df.loc[data_df[y_key]==1,:]
		normal_df = data_df.loc[data_df[y_key]==0,:]

		print("### disease_df")
		print(disease_df)

		print("### normal_df")
		print(normal_df)

		print(len(disease_df), len(normal_df))
		# # log_file = '/app/data/model/RF_3000_log.txt'
		fh_writer = open(log_file, 'w')

		while model_count < model_threshold:
			# (train_set, train_label, test_set, test_label) = sampling_obj.category2_sampling_pipeline(normal_df, disease_df,y_key, test_set_ratio)
			(train_set, test_set) = sampling_obj.category2_simple_sampling_pipeline(normal_df, disease_df,y_key, test_set_ratio)

			print("###############################")
			print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
			print(model_count)
			# print("######################## train_set")
			# print(train_set)

			# print("######################## test_set")
			# print(test_set)

			# #### temp data
			test_set_bak = test_set.copy()
			# # test_set_bak[y_key] = test_set_bak[y_key].astype('int')

			### get x,y
			(train_set_x, train_set_y) = sampling_obj.get_x_y_from_dataframe(train_set,y_key)
			(test_set_x, test_set_y) = sampling_obj.get_x_y_from_dataframe(test_set,y_key)

			# print("######################## train_set_x")
			# print(train_set_x)
			# print("######################## train_set_y")
			# print(train_set_y)

			# print("######################## test_set_x")
			# print(test_set_x)
			# print("######################## test_set_y")
			# print(test_set_y)

			### seperate by disease
			test_set_disease = test_set_bak.loc[test_set_bak[y_key]==1,:]
			test_set_normal = test_set_bak.loc[test_set_bak[y_key]==0,:]

			(test_set_disease_x, test_set_disease_y) = sampling_obj.get_x_y_from_dataframe(test_set_disease,y_key)
			(test_set_normal_x, test_set_normal_y) = sampling_obj.get_x_y_from_dataframe(test_set_normal,y_key)

			# print("######################## test_set_disease")
			# print(test_set_disease)
			# print("######################## test_set_disease_x")
			# print(test_set_disease_x)
			# print("######################## test_set_disease_y")
			# print(test_set_disease_y)

			# print("######################## test_set_normal")
			# print(test_set_normal)
			# print("######################## test_set_normal_x")
			# print(test_set_normal_x)
			# print("######################## test_set_normal_y")
			# print(test_set_normal_y)

			test_set_disease_len = len(test_set_disease_y)
			test_set_normal_len = len(test_set_normal_y)

			print("######################## test_set_disease_len, test_set_normal_len")
			print("{}, {}".format(test_set_disease_len,test_set_normal_len))
			# ### feature list
			# # feature_dict = {}
			# # feature_dict['numeric'] = list(train_set_x.columns.values)

			### training the model
			my_model = train_obj.SKRandomForest_Category(train_set_x, train_set_y)

			test_score = my_model.score(test_set_x,test_set_y)
			test_set_diseas_score = my_model.score(test_set_disease_x,test_set_disease_y)
			test_set_normal_score = my_model.score(test_set_normal_x,test_set_normal_y)

			print("Test score = {}".format(test_score))
			print("test_set_diseas_score score = {}".format(test_set_diseas_score))
			print("test_set_normal_score score = {}".format(test_set_normal_score))

			### log file
			fh_writer.write("{}\t{}\t{}\n".format(test_score,test_set_diseas_score,test_set_normal_score))

			### model
			model_file = model_output_dir +str(model_count) +'.pkl'
			joblib.dump(my_model, model_file)
			model_count += 1
		fh_writer.close()

		return data_df
	def cyto_ai_training_pipeline(self, data_df):

		data_processing_obj = DataProcessing()
		sampling_obj = DataSampling()
		sys_obj = SysConfig()
		train_obj = DataTrainning()

		test_set_ratio = sys_obj.get_test_set_ratio()
		y_key = sys_obj.get_y_key()

		print(data_df)
		#### data preprocessing
		print("start cast_all_to_numeric.")
		data_df = data_processing_obj.cast_all_to_numeric(data_df)
		print("end cast_all_to_numeric.")
		# print("########################")
		# print(data_df)

		### convert to category
		data_df[y_key] = data_df[y_key].astype('category')

		### sampling
		(train_set, test_set) = sampling_obj.sk_sampling(data_df, test_set_ratio)

		# print("######################## train_set")
		# print(train_set)

		# print("######################## test_set")
		# print(test_set)

		#### temp data
		test_set_bak = test_set.copy()
		# test_set_bak[y_key] = test_set_bak[y_key].astype('int')

		### get x,y
		(train_set_x, train_set_y) = sampling_obj.get_x_y_from_dataframe(train_set,y_key)
		(test_set_x, test_set_y) = sampling_obj.get_x_y_from_dataframe(test_set,y_key)

		# print("######################## train_set_x")
		# print(train_set_x)
		# print("######################## train_set_y")
		# print(train_set_y)

		# print("######################## test_set_x")
		# print(test_set_x)
		# print("######################## test_set_y")
		# print(test_set_y)

		### seperate by disease
		test_set_disease = test_set_bak.loc[test_set_bak[y_key]==1,:]
		test_set_normal = test_set_bak.loc[test_set_bak[y_key]==0,:]

		(test_set_disease_x, test_set_disease_y) = sampling_obj.get_x_y_from_dataframe(test_set_disease,y_key)
		(test_set_normal_x, test_set_normal_y) = sampling_obj.get_x_y_from_dataframe(test_set_normal,y_key)

		# print("######################## test_set_disease")
		# print(test_set_disease)
		# print("######################## test_set_disease_x")
		# print(test_set_disease_x)
		# print("######################## test_set_disease_y")
		# print(test_set_disease_y)

		# print("######################## test_set_normal")
		# print(test_set_normal)
		# print("######################## test_set_normal_x")
		# print(test_set_normal_x)
		# print("######################## test_set_normal_y")
		# print(test_set_normal_y)

		test_set_disease_len = len(test_set_disease_y)
		test_set_normal_len = len(test_set_normal_y)

		print("######################## test_set_disease_len, test_set_normal_len")
		print("{}, {}".format(test_set_disease_len,test_set_normal_len))
		# ### feature list
		# # feature_dict = {}
		# # feature_dict['numeric'] = list(train_set_x.columns.values)

		### training the model
		my_model = train_obj.SKRandomForest_Category(train_set_x, train_set_y)

		test_score = my_model.score(test_set_x,test_set_y)
		test_set_diseas_score = my_model.score(test_set_disease_x,test_set_disease_y)
		test_set_normal_score = my_model.score(test_set_normal_x,test_set_normal_y)

		print("Test score = {}".format(test_score))
		print("test_set_diseas_score score = {}".format(test_set_diseas_score))
		print("test_set_normal_score score = {}".format(test_set_normal_score))

		return data_df
		
	def cyto_cnn_training_pipeline(self, data_ary, test_ary):
		# data_processing_obj = DataProcessing()
		sampling_obj = DataSampling()
		sys_obj = SysConfig()
		train_obj = DataTrainning()
		process_obj = DataProcessing()
		
		TEST_VALIDATION_SPLIT = 0.33

		# model_output_dir = sys_obj.get_model_output_dir()

		#### data preprocessing
		np.random.shuffle(data_ary)
		train_set_x, train_set_y = sampling_obj.get_x_y_from_data_ary(data_ary)
		print('train set shape', train_set_x.shape)
		train_set_x = train_set_x.reshape(train_set_x.shape[0], 432, 1220 ,1)
		
		np.random.shuffle(test_ary)
		test_X_ary, test_Y_ary = sampling_obj.get_x_y_from_data_ary(test_ary)
		test_X_ary = test_X_ary.reshape(test_X_ary.shape[0], 432, 1220 ,1)
		(test_set_x, valid_set_x, test_set_y, valid_set_y) = train_test_split(test_X_ary, test_Y_ary, test_size=TEST_VALIDATION_SPLIT)
		# train_set_x, train_set_y = sampling_obj.binary_ary_data_over_sampling(train_set_x, train_set_y)

		### sampling

		my_model = train_obj.cnn_apply(train_set_x, train_set_y, valid_set_x, valid_set_y, test_X_ary, test_Y_ary)

		# test_score = my_model.evaluate(test_set_x, test_set_y)

		# test_set_diseas_score = my_model.score(test_set_disease_x, test_set_disease_y)
		# test_set_normal_score = my_model.score(test_set_normal_x, test_set_normal_y)

		# print("\ntest data set, %s: %.2f%%" % (my_model.metrics_names[1], test_score[1]*100))

		# print("Test score:", test_score[0])
		# print('Test accuracy:', test_score[1])

		return my_model


# -*- coding: utf8 -*-

import csv
import hashlib

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

class DataSampling(object):

	def __init__(self):
		# self.sys_obj = SysConfig()
		pass
		## print(tf.__version__)

	def category2_simple_sampling_pipeline(self, large_set_df, small_set_df, target_name, test_ratio=0.2):
		"""
		Sampling Interface.

		"""
		(large_set_train, large_set_test, small_set_train, small_set_test) = self.balance_sampling(large_set_df, small_set_df, test_ratio)

		# (large_set_train_x, large_set_train_y) = self.get_x_y_from_dataframe(large_set_train, target_name)
		# (large_set_test_x, large_set_test_y) = self.get_x_y_from_dataframe(large_set_test, target_name)

		# (small_set_train_x, small_set_train_y) = self.get_x_y_from_dataframe(small_set_train, target_name)
		# (small_set_test_x, small_set_test_y) = self.get_x_y_from_dataframe(small_set_test, target_name)

		#### merge data
		train_set = large_set_train.append(small_set_train)
		test_set = large_set_test.append(small_set_test)

		return (train_set, test_set)
	def category2_sampling_pipeline(self, large_set_df, small_set_df, target_name, test_ratio=0.2):
		"""
		Sampling Interface.

		"""
		(large_set_train, large_set_test, small_set_train, small_set_test) = self.balance_sampling(large_set_df, small_set_df, test_ratio)

		(large_set_train_x, large_set_train_y) = self.get_x_y_from_dataframe(large_set_train, target_name)
		(large_set_test_x, large_set_test_y) = self.get_x_y_from_dataframe(large_set_test, target_name)

		(small_set_train_x, small_set_train_y) = self.get_x_y_from_dataframe(small_set_train, target_name)
		(small_set_test_x, small_set_test_y) = self.get_x_y_from_dataframe(small_set_test, target_name)

		#### merge data
		train_set = large_set_train_x.append(small_set_train_x)
		train_label = large_set_train_y.append(small_set_train_y)

		test_set = large_set_test_x.append(small_set_test_x)
		test_label = large_set_test_y.append(small_set_test_y)

		return (train_set, train_label, test_set, test_label)
	def balance_sampling(self, large_set_df, small_set_df, test_ratio=0.8):
		'''
		'''

		# print("small_set_df = {0}".format(len(small_set_df)))

		sample_count = len(small_set_df)

		### random select some data from large data set
		large_set_random = self.get_random_sample(large_set_df, sample_count)

		# blast_count_train,blast_count_test,no_blast_count_train, no_blast_count_test = train_test_split(small_set_df, large_set_random, test_size=test_ratio, random_state=42)
		blast_count_train,blast_count_test,no_blast_count_train, no_blast_count_test = train_test_split(small_set_df, large_set_random, test_size=test_ratio)

		# print(len(blast_count_train), len(blast_count_test))
		# print(len(no_blast_count_train), len(no_blast_count_test))
		# print(type(blast_count_test))

		return (no_blast_count_train, no_blast_count_test,blast_count_train,blast_count_test)

	def get_x_y_from_dataframe(self, input_df, target_name='cnv_outcome'):
		feature_x, target_y = input_df, input_df.pop(target_name)
		return (feature_x, target_y)
	def cbc_sampling_pipeline(self, df_1, df_2, target_name, sample_ratio=0.8):
		"""
		Sampling Interface.

		"""
		(no_blast_count_train, no_blast_count_test, blast_count_train, blast_count_test) = self.cbc_blast_count_balance_sampling(df_1, df_2, sample_ratio)

		(no_blast_count_train_x, no_blast_count_train_y) = self.cbc_load_data(no_blast_count_train, target_name)
		(no_blast_count_test_x, no_blast_count_test_y) = self.cbc_load_data(no_blast_count_test, target_name)

		(blast_count_train_x, blast_count_train_y) = self.cbc_load_data(blast_count_train, target_name)
		(blast_count_test_x, blast_count_test_y) = self.cbc_load_data(blast_count_test, target_name)

		#### merge data
		train_set = no_blast_count_train_x.append(blast_count_train_x)
		train_label = no_blast_count_train_y.append(blast_count_train_y)

		test_set = no_blast_count_test_x.append(blast_count_test_x)
		test_label = no_blast_count_test_y.append(blast_count_test_y)

		return (train_set, train_label, test_set, test_label)

	def cbc_load_data(self, input_df, target_name):
		feature_x, target_y = input_df, input_df.pop(target_name)
		return (feature_x, target_y)

	def cbc_blast_count_balance_sampling(self, noblast_count_df, blast_count_df, sample_ratio=0.8):

		print("blast_count_df = {0}".format(len(blast_count_df)))

		sample_count = len(blast_count_df)

		noblast_count_random = self.get_random_sample(noblast_count_df, sample_count)

		# blast_count_train,blast_count_test = train_test_split(blast_count_df, test_size=sample_ratio, random_state=42)

		# no_blast_count_train, no_blast_count_test = train_test_split(noblast_count_random, test_size=sample_ratio, random_state=42)

		blast_count_train,blast_count_test,no_blast_count_train, no_blast_count_test = train_test_split(blast_count_df,noblast_count_random, test_size=sample_ratio, random_state=42)

		print(len(blast_count_train), len(blast_count_test))
		print(len(no_blast_count_train), len(no_blast_count_test))
		print(type(blast_count_test))

		# self.sk_sampling(blast_count_df,sample_ratio,)
		return (no_blast_count_train, no_blast_count_test,blast_count_train,blast_count_test)
	def sk_sampling(self, data, test_ratio):
		'''
		Input:
			data: data set.
			test_ratio: test set ratio.
		'''
		train_set, test_set = train_test_split(data, test_size=test_ratio, random_state=42)

		return train_set, test_set

	def test_set_check(self, identifier, test_ratio, hash_func):
		'''
		根據hash digest的最後一bytes來決定test set，
		1 byte = 8 bits = 256可能性
		'''
		return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

	def split_train_test_by_id(self, data, test_ratio, id_column, hash_func=hashlib.md5):
		ids = data[id_column]
		in_test_set = ids.apply(lambda id_: self.test_set_check(id_, test_ratio, hash_func))
		return data.loc[~in_test_set], data.loc[in_test_set]

	def get_random_sample(self, data, test_size):
		shuffled_indices = np.random.permutation(len(data))
		test_indices = shuffled_indices[:test_size]
		return (data.iloc[test_indices])

	def split_train_test(self, data, test_ratio):
		shuffled_indices = np.random.permutation(len(data))
		test_size = int(len(data) * test_ratio)
		test_indices = shuffled_indices[:test_size]
		train_indices = shuffled_indices[test_size:]
		return (data.iloc[train_indices], data.iloc[test_indices])
	def get_x_y_from_data_ary(self, input_ary):
		'''
		Get X_ary and Y_ary from input_ary data.
		'''
		data_X = []
		data_Y = []
		for count, each_data in enumerate(input_ary):
			pixel_ary = each_data[0]
			label = each_data[1]
			
			data_X.append(pixel_ary)
			data_Y.append(label)

		data_X = np.array(data_X)
		data_Y = np.array(data_Y)

		return (data_X, data_Y)
	
	def binary_ary_data_over_sampling(self, data_X, data_Y):
		'''
		Add copies of instances from the under-represented class.
		'''
		print('Execute over sampling, directly copy the under-represented class.')
		print('Original binary counts:')
		unique, counts = np.unique(data_Y, return_counts = True)
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
		resample_X = []
		resample_Y = []
		for record, each_Y in enumerate(data_Y):
			resample_X.append(data_X[record])
			resample_Y.append(each_Y)
			if each_Y == minor_class:
				coda = r
				while(coda>1):
					resample_X.append(data_X[record])
					resample_Y.append(each_Y)
					coda -= 1
		resample_X = np.array(resample_X)
		resample_Y = np.array(resample_Y)
		return (resample_X, resample_Y)


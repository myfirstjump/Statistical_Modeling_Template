
# -*- coding: utf8 -*-

import csv
import re

import numpy as np
import pandas as pd
from PIL import Image
import cv2

from file_list import FileList

class DataReader(object):

	def __init__(self):
		self.file_obj = FileList()
		pass
		## print(tf.__version__)
	def build_array_with_cnv_to_gene(self, cnv_df, array_2_gene, gene_2_array, array_id_key='Array_ID'):
		'''
		Function:
			只保留R分析後，有CNV的資料。
			r analysis cnv result to gene symbol.
		'''
		print("In build_array_with_cnv_to_gene ...")
		# probe_id_list = list(cnv_df.columns.values)

		cnv_ary = cnv_df.to_dict(orient='records')

		# for temp_index,row in cnv_df.iterrows():
		result_ary = []
		total_count = 0
		hit_count = 0
		no_hit_count = 0
		for cnv_dict in cnv_ary:
			# print(cnv_dict)
			gene_cnv_dict = {}
			array_id = cnv_dict[array_id_key]
			gene_cnv_dict[array_id_key] = array_id

			sample_hit_gene_dict = {}
			### sample with cnv
			if array_id in array_2_gene:
				sample_hit_gene_dict = self.list_2_dict(array_2_gene[array_id])
			

				for gene_symbol,value in gene_2_array.items():
					hit_flag = 0
					if gene_symbol in sample_hit_gene_dict:
						hit_flag = 1
						# print(array_id, gene_symbol)
					else:
						hit_flag = 0

					gene_cnv_dict[gene_symbol] = hit_flag
				result_ary.append(gene_cnv_dict)

				hit_count += 1
			else:
				no_hit_count += 1
			
			total_count += 1

		gene_df = pd.DataFrame.from_dict(result_ary)

		print("Total_count = {}, hit_count = {}, no_hit_count = {}.".format(total_count, hit_count, no_hit_count))
		# probe_id_list = list(gene_df.columns.values)
		# print(probe_id_list)
		# print(gene_df)
		# print(len(cnv_df))
		# print(cnv_df[array_id_key])

		print("Out build_cnv_to_gene ...")

		return gene_df
	def get_cnv_gainloss_to_gene_table(self, input_file, array_index=3, gain_loss_index=4, gene_index=8):
		"""
		Function:
			將array_id和gene_symbol gain/loss的mapping file整理成Dict.
			回傳兩種型態.
			array_2_gene和gene_2_array

		Input:
			input_file: 為array_id對應gene_symbol的summary file.
			array_index: column number of array_id.
			gainloss_index: indec of gain/loss field.
			gene_index: column number fo gene.
		"""

		fh_input = open(input_file, 'r')
		fh_csv = csv.reader(fh_input, delimiter='\t')

		## header string
		# # row = next(fh_csv)
		# # features_ary = np.array(row)

		array_2_gene = {}
		array_id = ''
		gene_symbol = ''
		gene_2_array = {}
		for row in fh_csv:
			# print(row)
			array_id = row[array_index]
			gene_symbol = row[gene_index]
			gain_loss_int = row[gain_loss_index]

			gene_symbol_with_gain_loss = ''
			if gain_loss_int == 1:
				gene_symbol_with_gain_loss = str(gene_symbol) +"_Gain"
			else:
				gene_symbol_with_gain_loss = str(gene_symbol) +"_Loss"

			if not array_id in array_2_gene:
				array_2_gene[array_id] = []
			array_2_gene[array_id].append(gene_symbol_with_gain_loss)

			if not gene_symbol_with_gain_loss in gene_2_array:
				gene_2_array[gene_symbol_with_gain_loss] = []
			gene_2_array[gene_symbol_with_gain_loss].append(array_id)


		fh_input.close()

		### remove redundant data.
		refine_array_2_gene = {}
		for array_id, gene_ary in array_2_gene.items():
			gene_ary = list(set(gene_ary))
			refine_array_2_gene[array_id] = gene_ary

		refine_gene_2_array = {}
		for gene_symbol,array_id_ary in gene_2_array.items():
			array_id_ary = list(set(array_id_ary))
			refine_gene_2_array[gene_symbol] = array_id_ary

		return refine_array_2_gene, refine_gene_2_array

	def build_region_40_summary_with_gainloss_file(self, input_path, summary_file):
		'''
		整理region40的結果，包含gain/loss資訊(1/-1).
		輸出格式為: chr,start,end,gpr_id，gain/loss
		'''
		fh_output = open(summary_file, 'w')
		fh_output.write('chr\tstart\tend\tarray_id\tgain_loss\n')

		feature_ary = ['Chromosome', 'Start', 'End', 'Gain_loss']
		region_40_dict = self.read_region_40_file_pipeline(input_path)

		cnv_count = 0
		normal_count = 0
		for gpr_id, data_df in region_40_dict.items():
			if not data_df is None:
				ary_len = len(data_df)

				feature_list = data_df.loc[:,feature_ary].values
				print(gpr_id, ary_len)
				# print(feature_list)
				for temp_chr,temp_start,temp_end,temp_gain_loss in feature_list:
					temp_gain_loss_int = 0
					temp_gain_loss = str(temp_gain_loss)
					if temp_gain_loss =="Loss":
						temp_gain_loss_int = -1
					elif temp_gain_loss =="Gain":
						temp_gain_loss_int = 1
					else:
						temp_gain_loss_int = 0

					temp_start = re.sub(r'[,]', '', temp_start)
					temp_end = re.sub(r'[,]', '', temp_end)

					if temp_chr == 23:
						temp_chr = 'X'
					elif temp_chr == 24:
						temp_chr = 'Y'
					else:
						pass
					print(temp_chr,temp_start,temp_end,temp_gain_loss,temp_gain_loss_int,)
					fh_output.write('chr{}\t{}\t{}\t{}\t{}\n'.format(temp_chr,temp_start,temp_end,gpr_id,temp_gain_loss_int))
				# print(data_df)
				cnv_count += 1
			else:
				print("Normal.  {} .....".format(gpr_id))

				normal_count += 1
		
		print("cnv_count = {}, normal_count = {}.".format(cnv_count, normal_count))

		print("Output file = {}".format(summary_file))
		fh_output.close()

	def build_region_40_summary_file(self, input_path, summary_file):
		'''
		整理region40的結果.
		輸出格式為: chr,start,end,gpr_id
		'''
		fh_output = open(summary_file, 'w')
		fh_output.write('chr\tstart\tend\tarray_id\n')

		feature_ary = ['Chromosome', 'Start', 'End']
		region_40_dict = self.read_region_40_file_pipeline(input_path)

		cnv_count = 0
		normal_count = 0
		for gpr_id, data_df in region_40_dict.items():
			if not data_df is None:
				ary_len = len(data_df)

				feature_list = data_df.loc[:,feature_ary].values
				print(gpr_id, ary_len)
				# print(feature_list)
				for temp_chr,temp_start,temp_end in feature_list:
					temp_start = re.sub(r'[,]', '', temp_start)
					temp_end = re.sub(r'[,]', '', temp_end)

					if temp_chr == 23:
						temp_chr = 'X'
					elif temp_chr == 24:
						temp_chr = 'Y'
					else:
						pass
					print(temp_chr,temp_start,temp_end)
					fh_output.write('chr{}\t{}\t{}\t{}\n'.format(temp_chr,temp_start,temp_end,gpr_id))
				# print(data_df)
				cnv_count += 1
			else:
				print("Normal.  {} .....".format(gpr_id))

				normal_count += 1
		
		print("cnv_count = {}, normal_count = {}.".format(cnv_count, normal_count))

		print("Output file = {}".format(summary_file))
		fh_output.close()


	def list_2_dict(self, input_list):
		result_dict = dict((tmp_key,tmp_key) for tmp_key in input_list)
		return result_dict

	def build_array_to_gene(self, cnv_df, array_2_gene, gene_2_array, array_id_key='Array_ID'):
		'''
		Function:
			r analysis cnv result to gene symbol.
		'''
		print("In build_cnv_to_gene ...")
		# probe_id_list = list(cnv_df.columns.values)

		cnv_ary = cnv_df.to_dict(orient='records')

		# for temp_index,row in cnv_df.iterrows():
		result_ary = []
		for cnv_dict in cnv_ary:
			# print(cnv_dict)
			gene_cnv_dict = {}
			array_id = cnv_dict[array_id_key]
			gene_cnv_dict[array_id_key] = array_id

			sample_hit_gene_dict = {}
			### sample with cnv
			if array_id in array_2_gene:
				sample_hit_gene_dict = self.list_2_dict(array_2_gene[array_id])
			else:
				pass

			for gene_symbol,value in gene_2_array.items():
				hit_flag = 0
				if gene_symbol in sample_hit_gene_dict:
					hit_flag = 1
					print(array_id, gene_symbol)
				else:
					hit_flag = 0

				gene_cnv_dict[gene_symbol] = hit_flag
			result_ary.append(gene_cnv_dict)

		gene_df = pd.DataFrame.from_dict(result_ary)

		# probe_id_list = list(gene_df.columns.values)
		# print(probe_id_list)
		# print(gene_df)
		# print(len(cnv_df))
		# print(cnv_df[array_id_key])

		print("Out build_cnv_to_gene ...")

		return gene_df
	def get_cnv_to_gene_table(self, input_file, array_index=3, gene_index=7):
		"""
		Function:
			將array_id和gene_symbol的mapping file整理成Dict.
			回傳兩種型態.
			array_2_gene和gene_2_array

		Input:
			input_file: 為array_id對應gene_symbol的summary file.
			array_index: column number of array_id.
			gene_index: column number fo gene.
		"""

		fh_input = open(input_file, 'r')
		fh_csv = csv.reader(fh_input, delimiter='\t')

		## header string
		# # row = next(fh_csv)
		# # features_ary = np.array(row)

		array_2_gene = {}
		array_id = ''
		gene_symbol = ''
		gene_2_array = {}
		for row in fh_csv:
			# print(row)
			array_id = row[array_index]
			gene_symbol = row[gene_index]

			if not array_id in array_2_gene:
				array_2_gene[array_id] = []
			array_2_gene[array_id].append(gene_symbol)

			if not gene_symbol in gene_2_array:
				gene_2_array[gene_symbol] = []
			gene_2_array[gene_symbol].append(array_id)


		fh_input.close()

		### remove redundant data.
		refine_array_2_gene = {}
		for array_id, gene_ary in array_2_gene.items():
			gene_ary = list(set(gene_ary))
			refine_array_2_gene[array_id] = gene_ary

		refine_gene_2_array = {}
		for gene_symbol,array_id_ary in gene_2_array.items():
			array_id_ary = list(set(array_id_ary))
			refine_gene_2_array[gene_symbol] = array_id_ary

		return refine_array_2_gene, refine_gene_2_array

	def build_cnv_to_gene(self, cnv_df, gene_2_probe, array_id_key='Array_ID'):
		print("In build_cnv_to_gene ...")
		# probe_id_list = list(cnv_df.columns.values)

		cnv_ary = cnv_df.to_dict(orient='records')

		# for temp_index,row in cnv_df.iterrows():
		result_ary = []
		for cnv_dict in cnv_ary:
			# print(cnv_dict)
			gene_cnv_dict = {}
			gene_cnv_dict[array_id_key] = cnv_dict[array_id_key]
			for gene_symbol,value in gene_2_probe.items():
				temp_sum = 0.0
				temp_count = 0
				for probe_id in value:

					if probe_id in cnv_dict:
						try:
							temp_sum += float(cnv_dict[probe_id])

							temp_count += 1
						except:
							temp_sum += 0.0

				if temp_count == 0:
					temp_count = 1
				temp_sum =  temp_sum / temp_count
				gene_cnv_dict[gene_symbol] = temp_sum
			result_ary.append(gene_cnv_dict)

		gene_df = pd.DataFrame.from_dict(result_ary)

		# probe_id_list = list(gene_df.columns.values)
		# print(probe_id_list)
		# print(gene_df)
		# print(len(cnv_df))
		# print(cnv_df[array_id_key])

		print("Out build_cnv_to_gene ...")

		return gene_df

	def get_probe_to_gene_table(self, input_file, probe_index=3, gene_index=7):
		"""
		Input:
			input_file: csv file.
			probe_index: column number of probe.
			gene_index: column number fo gene.
		"""

		fh_input = open(input_file, 'r')
		fh_csv = csv.reader(fh_input, delimiter='\t')

		## header string
		# # row = next(fh_csv)
		# # features_ary = np.array(row)

		probe_2_gene = {}
		probe_id = ''
		gene_symbol = ''
		gene_2_probe = {}
		for row in fh_csv:
			# print(row)
			probe_id = row[probe_index]
			gene_symbol = row[gene_index]
			probe_2_gene[probe_id] = gene_symbol

			if not gene_symbol in gene_2_probe:
				gene_2_probe[gene_symbol] = []

			gene_2_probe[gene_symbol].append(probe_id)


		fh_input.close()
		return probe_2_gene,gene_2_probe


	def combine_outcome_data(self, cnv_df, outcome_dict,combine_column='Array_ID'):
		'''
		'''
		outcome_df = pd.DataFrame.from_dict(outcome_dict, orient='index')
		outcome_df.columns = ['cnv_outcome']

		outcome_df[combine_column] = outcome_df.index
		merge_df = pd.merge(cnv_df,outcome_df)
		# print(merge_df)

		return merge_df

	def cnv_data_reader_pipeline(self, input_path):
		'''
		Function:
			資料來源為CNV結果，(R分析之後的結果)
			產生以probe_id為column的data frame.
			value為log2 ration.

		'''
		probe_ary = []
		### 紀錄probe_id的聯集
		columns_dict = {'Array_ID':'Array_ID'}

		cnv_all_df = None
		file_list = self.file_obj.get_all_probe_bind_file(input_path)
		gpr_code = ''
		file_count = len(file_list)
		temp_count = 0
		for temp_file in file_list:
			gpr_code = self.file_obj.get_gpr_code_from_path(temp_file)

			## probe_info_dict ={ probe_id:log2, ...}
			probe_info_dict = self.region_40_file_reader_to_dict(temp_file, gpr_code[0], columns_dict)

			probe_ary.append(probe_info_dict)

			temp_count += 1

			print("%s/%s, File = %s"%(temp_count, file_count,temp_file))
			print("gpr_code = %s"%(gpr_code[0]))
			print("columns_dict len = %s"%(len(columns_dict)))

			# ### debug info
			# if temp_count ==20:
			# 	break

		data_dict = {}
		log2_value = 0
		for probe_info_dict in probe_ary:
			for temp_key in columns_dict.keys():
				if temp_key in probe_info_dict:
					log2_value = probe_info_dict[temp_key]
				else:
					log2_value = 0

				if not temp_key in data_dict:
					data_dict[temp_key] = []

				data_dict[temp_key].append(log2_value)

		cnv_all_df = pd.DataFrame.from_dict(data_dict)
		# print(cnv_all_df)
		# print(cnv_all_df.info())
		return cnv_all_df

	def cnv_data_reader(self, input_file, array_id, colunm_tag='ID', value_tag='log2'):
		data_df = self.region_40_file_reader(input_file)
		# data_frame_t = data_df.set_index(colunm_tag).T

		### get certain row as value
		serial_log2 = data_df.loc[:,[colunm_tag,value_tag]]

		### set array_id as index and transpose as columns
		serial_log2 = serial_log2.set_index(colunm_tag).T

		### assign array_id to index label
		serial_log2 = serial_log2.rename(index={value_tag:array_id})

		### adding array_id column
		serial_log2['Array_ID'] = serial_log2.index

		return serial_log2
	def cnv_data_reader_pipeline_bak(self, input_path):
		'''
		'''
		cnv_all_df = None
		file_list = self.file_obj.get_all_probe_bind_file(input_path)
		gpr_code = ''
		file_count = len(file_list)
		temp_count = 0
		for temp_file in file_list:
			gpr_code = self.file_obj.get_gpr_code_from_path(temp_file)
			cnv_df = self.cnv_data_reader(temp_file, gpr_code[0])

			if temp_count >0:
				cnv_all_df = cnv_all_df.append(cnv_df.copy())
			else:
				cnv_all_df = cnv_df.copy()

			temp_count += 1

			# print(type(cnv_all_df))
			print("%s/%s, File = %s"%(temp_count, file_count,temp_file))
			print("gpr_code = %s"%(gpr_code[0]))

		return cnv_all_df

	def cnv_data_reader_bak(self, input_file, array_id, colunm_tag='ID', value_tag='log2'):
		data_df = self.region_40_file_reader(input_file)

		### set array_id as index and transpose as columns
		data_frame_t = data_df.set_index(colunm_tag).T

		### get certain row as value
		serial_log2 = data_frame_t.loc[value_tag,:]

		## convert serials to data_frame
		data_frame_t_log2 = serial_log2.to_frame()
		data_frame_t_log2 = data_frame_t_log2.T

		### assign array_id to index label
		data_frame_t_log2 = data_frame_t_log2.rename(index={value_tag:array_id})

		### adding array_id column
		data_frame_t_log2['Array_ID'] = data_frame_t_log2.index

		# print(type(data_frame_t_log2))

		return data_frame_t_log2

	def read_all_probe_bind_pipeline(self, input_path):
		'''
		Output:
			result_dict: {}
				key: gpr_id,
				value: cnv_df, pandas dataframe
		'''
		file_list = self.file_obj.get_all_probe_bind_file(input_path)
		gpr_code = ''
		result_dict = {}
		file_count = len(file_list)
		temp_count = 0
		for temp_file in file_list:
			print("%s/%s, File = %s"%(temp_count, file_count,temp_file))
			cnv_df = None
			gpr_code = self.file_obj.get_gpr_code_from_path(temp_file)
			cnv_df = self.region_40_file_reader(temp_file)
			result_dict[gpr_code[0]] = cnv_df

			print("gpr_code = %s"%(gpr_code))

			temp_count += 1



		# result_df = self.pd_read_csv_data(input_file)

		return result_dict
	def read_region_40_file_pipeline(self, input_path):
		'''
		Output:
			result_dict: {}
				key: gpr_id,
				value: cnv_df, pandas dataframe
		'''
		region_40_list = self.file_obj.get_region_40_file(input_path)
		gpr_code = ''
		result_dict = {}
		for temp_file in region_40_list:
			print("File = %s"%(temp_file))
			cnv_df = None
			gpr_code = self.file_obj.get_gpr_code_from_path(temp_file)
			cnv_df = self.region_40_file_reader(temp_file)
			result_dict[gpr_code[0]] = cnv_df

			print("gpr_code = %s"%(gpr_code))


		# result_df = self.pd_read_csv_data(input_file)

		return result_dict
	def region_40_file_reader_to_dict(self,input_file, array_id, record_dict):
		'''
		Function:
			讀取region_40/All_probe_bind的結果，紀錄probe_id和對應的log2_ratio。
			根據header來判斷檔案內是否有資料。
		Input:
			input_file: tab format input file.
			record_dict: 記錄所有probe_id.
		Output:
			pandas data frame.
		'''
		probe_id_index = 0
		log2_index = 5
		result_df = None
		fh_input = open(input_file,'r',encoding="latin1")
		header_str = next(fh_input)
		temp_ary = header_str.split('\t')

		probe_dict = {}
		probe_dict['Array_ID'] = array_id
		if len(temp_ary) > 1:
			for temp_str in fh_input:
				temp_ary = temp_str.split('\t')
				probe_id = temp_ary[probe_id_index]
				log2_value =  temp_ary[log2_index]
				probe_dict[probe_id] = log2_value
				record_dict[probe_id] = probe_id

		fh_input.close()
		return probe_dict
	def region_40_file_reader(self,input_file):
		'''
		Function:
			根據header來判斷檔案內是否有資料。
		Input:
			tab format input file.
		Output:
			pandas data frame.
		'''
		result_df = None
		fh_input = open(input_file,'r')
		header_str = next(fh_input)
		temp_ary = header_str.split('\t')
		fh_input.close()
		if len(temp_ary) > 1:
			result_df = self.pd_read_csv_data(input_file)

		return result_df

	def read_blast_count_data(self, sys_obj, input_dir):
		accessnum_index = sys_obj.get_accessnum_index()
		blast_count_index = sys_obj.get_blast_count_index()

		blast_count_df = self.pd_read_csv_data_from_dir(input_dir)
		blast_value_temp = blast_count_df.copy()
		blast_value_temp.iloc[:, accessnum_index] = blast_value_temp.iloc[:, accessnum_index].str.replace(",","").str.replace(".","")
		blast_value_temp = blast_value_temp.iloc[:, [accessnum_index,blast_count_index]]

		blast_count_refine = blast_value_temp
		blast_count_refine.columns = ["Specimen_ID","Blast_count"]

		# blast_count_refine = blast_value_temp.rename(columns={"Accessnum":"Specimen ID"})

		# print(blast_count_refine)
		# print("Total blast_count_refine = {0} .".format(len(blast_count_refine)))

		return blast_count_refine

	def pd_read_csv_data_from_dir(self, input_dir, extension_str='.csv'):
		'''
		Input:
			input_dir: '/input/dir'
		Output:
			feature_ary = np.array
			data_x = np.array
			data_y = np.array

		'''
		file_obj = FileList()

		file_ary = file_obj.find_file(input_dir, extension_str)

		df_list = []
		for temp_file in file_ary:
			temp_df = self.pd_read_csv_data(temp_file)

			df_list.append(temp_df)

		result_df = pd.concat(df_list, axis=0)

		## reset all index
		result_df = result_df.set_index(np.arange(result_df.shape[0]))

		return result_df

	def pd_read_csv_data(self, input_file):
		data_frame = pd.read_csv(input_file, delimiter='\t',encoding="latin1")
		# print(data_frame)

		return data_frame

	def read_csv_data(self, input_file, outcome_index=26):
		'''
		feature_ary = np.array (header)
		data_x = np.array
		data_y = np.array
		'''
		fh_input = open(input_file, 'r')
		# csv_input = csv.reader.(fh_input, delimiter=',', quotechar='"')
		fh_csv = csv.reader(fh_input, delimiter=',')

		## header string
		row = next(fh_csv)
		features_ary = np.array(row)

		data_x = []
		data_y = []
		for row in fh_csv:
			data_x.append(row)
			data_y.append(row[outcome_index])

		data_x = np.array(data_x)
		data_y = np.array(data_y)
		fh_input.close()

		return features_ary,data_x,data_y

	def get_csv_data_from_dir(self, input_dir, outcome_index=26, extension_str='.csv'):
		'''
		Input:
			input_dir: '/input/dir'
		Output:
			feature_ary = np.array
			data_x = np.array
			data_y = np.array

		'''
		file_obj = FileList()

		file_ary = file_obj.find_file(input_dir, extension_str)
		data_x = []
		data_y = []

		data_x = np.array(data_x)
		data_y = np.array(data_y)

		for temp_index,temp_file in enumerate(file_ary):
			features_ary,temp_data_x,temp_data_y = self.read_csv_data(temp_file, outcome_index)

			if temp_index == 0:
				data_x = temp_data_x
				data_y = temp_data_y
			else:
				data_x = np.concatenate((data_x,temp_data_x),axis=0)
				data_y = np.concatenate((data_y,temp_data_y),axis=0)

			print(temp_file)
			print(len(temp_data_x))
			print(len(data_x))

		return features_ary,data_x,data_y

	def pd_read_txt(self, data_dir, file_path):
		obj = pd.read_csv(data_dir + file_path, sep='\t', header = None, names=['Array_ID', 'tif_path'])

		return obj

	def tif_ary_reader(self, data_df, path_column, y_label):
		
		records_num = len(data_df.index)
		print('How many records:', records_num)
		
		data_ary = []
		for i in range(records_num):
			path = data_df.loc[i, path_column]
			y = data_df.loc[i, y_label]

			im = Image.open(path)
			img_ary = np.array(im) # scaling the input
			img_ary = np.divide(img_ary, 255).astype('uint8') # uint16 -> uint8
			# img_ary = img_ary.astype('uint8')
			if img_ary.shape[0] > img_ary.shape[1]:
				print('*** Img transpose ***:')
				print(i)
				print('Img shape origin:')
				print(img_ary.shape)
				img_ary = np.transpose(img_ary)
				print('After transpose:')
				print(img_ary.shape)

			## 為了以image augmentation 解決 imbalance，旋轉放在後面流程處理，寫在 data_processing裡面
			resized_img_ary = cv2.resize(img_ary, (1220, 432)) #(610, 216)   cv2.resize 跟 array.shape 的顯示是顛倒的 ...
			# M = cv2.getRotationMatrix2D((img_px_size/2, img_px_size/2), randint(-10, 10), 1)
			# rotated_img_ary = cv2.warpAffine(resized_img_ary, M, (img_px_size, img_px_size))
			im.close()

			data_ary.append([resized_img_ary, y])

		return data_ary

	def png_ary_reader(self, data_df, path_column, y_label):
		records_num = len(data_df.index)
		print('How many records:', records_num)
		
		data_ary = []
		crop_box = (60, 100, 2369, 800) # 左右: 切到邊框   上下: 切到 +/- 3
		for i in range(records_num):
			print('Fig ', i, ' processing... \n')
			path = data_df.loc[i, path_column]
			y = data_df.loc[i, y_label]

			im = Image.open(path)
			crop_im = im.crop(crop_box)

			img_ary = np.array(crop_im) # scaling the input
			# img_ary = np.divide(img_ary, 255).astype('uint8') # uint16 -> uint8
			# img_ary = img_ary.astype('uint8')
			# if img_ary.shape[0] > img_ary.shape[1]:
			# 	print('*** Img transpose ***:')
			# 	print(i)
			# 	print('Img shape origin:')
			# 	print(img_ary.shape)
			# 	img_ary = np.transpose(img_ary)
			# 	print('After transpose:')
			# 	print(img_ary.shape)

			## After crop -> 2309 * 800
			resized_img_ary = cv2.resize(img_ary, (1150, 400)) #(610, 216)   cv2.resize 跟 array.shape 的顯示是顛倒的 ...
			# M = cv2.getRotationMatrix2D((img_px_size/2, img_px_size/2), randint(-10, 10), 1)
			# rotated_img_ary = cv2.warpAffine(resized_img_ary, M, (img_px_size, img_px_size))
			im.close()

			data_ary.append([resized_img_ary, y])

		return data_ary

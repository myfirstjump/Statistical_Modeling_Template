# -*- coding: utf8 -*-
import re
import time
import csv
import sys
import math
import csv
import os

from file_list import FileList
from Excel_Reader import ExcelReader
from data_reader import DataReader

class CytoOADataMain(object):

	def __init__(self):
		pass

	def build_cnv_only_2_gene_training_data(self, data_dir, outcome_file, cnv_2_gene_file):
		'''
		Function:
			只抓取有CNV的資料，產生以gene symbol為feature的sample set.
		Input:
			cnv_2_gene_file: 整理好的array_id對應到gene_symbol.

		'''

		print(" ##### In build_cnv_only_2_gene_training_data ... ")

		excel_obj = ExcelReader()
		data_reader_obj = DataReader()

		outcome_dict = excel_obj.get_cyto_cnv_result(outcome_file)
		cnv_df = data_reader_obj.cnv_data_reader_pipeline(data_dir)

		#### probe mapping to gene
		(array_2_gene, gene_2_array) = data_reader_obj.get_cnv_to_gene_table(cnv_2_gene_file)

		# print(array_2_gene)
		gene_cnv = data_reader_obj.build_array_with_cnv_to_gene(cnv_df, array_2_gene, gene_2_array)

		## gene cnv
		data_df = data_reader_obj.combine_outcome_data(gene_cnv, outcome_dict)

		return data_df

	def build_cnv_gainloss_2_gene_training_data(self, data_dir, outcome_file, cnv_2_gene_file):
		'''
		Function:
			針對gene symbol增加gain/loss為feature.
		Input:
			cnv_2_gene_file: 整理好的array_id對應到gene_symbol.

		'''
		excel_obj = ExcelReader()
		data_reader_obj = DataReader()

		outcome_dict = excel_obj.get_cyto_cnv_result(outcome_file)
		cnv_df = data_reader_obj.cnv_data_reader_pipeline(data_dir)

		#### probe mapping to gene
		(array_2_gene, gene_2_array) = data_reader_obj.get_cnv_gainloss_to_gene_table(cnv_2_gene_file)

		# print(array_2_gene)
		gene_cnv = data_reader_obj.build_array_to_gene(cnv_df, array_2_gene, gene_2_array)

		## gene cnv
		data_df = data_reader_obj.combine_outcome_data(gene_cnv, outcome_dict)

		return data_df

	def build_cnv_2_gene_training_data(self, data_dir, outcome_file, cnv_2_gene_file):
		'''
		Function:
			產生以gene symbol為feature的sample set.
		Input:
			cnv_2_gene_file: 整理好的array_id對應到gene_symbol.

		'''
		excel_obj = ExcelReader()
		data_reader_obj = DataReader()

		outcome_dict = excel_obj.get_cyto_cnv_result(outcome_file)
		cnv_df = data_reader_obj.cnv_data_reader_pipeline(data_dir)

		#### probe mapping to gene
		(array_2_gene, gene_2_array) = data_reader_obj.get_cnv_to_gene_table(cnv_2_gene_file)

		# print(array_2_gene)
		gene_cnv = data_reader_obj.build_array_to_gene(cnv_df, array_2_gene, gene_2_array)

		## gene cnv
		data_df = data_reader_obj.combine_outcome_data(gene_cnv, outcome_dict)

		return data_df


	def build_probe_2_gene_training_data(self, data_dir, outcome_file, probe_2_gene_file):
		excel_obj = ExcelReader()
		data_reader_obj = DataReader()

		outcome_dict = excel_obj.get_cyto_cnv_result(outcome_file)
		cnv_df = data_reader_obj.cnv_data_reader_pipeline(data_dir)

		#### probe mapping to gene
		(probe_2_gene, gene_2_probe) = data_reader_obj.get_probe_to_gene_table(probe_2_gene_file)

		gene_cnv = data_reader_obj.build_cnv_to_gene(cnv_df, gene_2_probe)

		### probe cnv
		# # data_df = data_reader_obj.combine_outcome_data(cnv_df, outcome_dict)

		## gene cnv
		data_df = data_reader_obj.combine_outcome_data(gene_cnv, outcome_dict)

		return data_df

	def build_cnv_training_data(self, data_dir, outcome_file):
		excel_obj = ExcelReader()
		data_reader_obj = DataReader()

		outcome_dict = excel_obj.get_cyto_cnv_result(outcome_file)
		cnv_df = data_reader_obj.cnv_data_reader_pipeline(data_dir)

		data_df = data_reader_obj.combine_outcome_data(cnv_df, outcome_dict)

		return data_df
	
	def get_tif_data_table(self, data_dir, file_name):
		data_reader_obj = DataReader()
		data_df = data_reader_obj.pd_read_txt(data_dir, file_name)
		return data_df

	def get_img_ary(self, data_df):
		data_reader_obj = DataReader()
		# data_ary = data_reader_obj.tif_ary_reader(data_df, 'tif_path', 'cnv_outcome')
		data_ary = data_reader_obj.png_ary_reader(data_df, 'tif_path', 'cnv_outcome')

		return data_ary



def list_2_dict(input_list):
	# dict_data = {temp:temp for temp in input_list}

	dict_data = {}
	redundant_id = []
	for temp_id in input_list:
		if temp_id in dict_data:
			# print("redundant id = {}".format(temp_id))

			dict_data[temp_id] += 1
			redundant_id.append(temp_id)
		else:
			dict_data[temp_id] = 1

	return dict_data

def get_gpr_id_from_excel_df(excel_df):
	array_id = list(excel_df.iloc[:,0])
	return array_id

def cnv_data_reader():
	cnv_obj = CytoOADataMain()

	### probe bind directory
	data_dir = '/home/ryan/src_dir/CytoCloudR/tmp/gene/output'
	data_dir = '/home/ryan/src_dir/CytoCloudR/tmp/gene/test_output'

	### report excel
	outcome_file = '/home/ryan/src_dir/CytoOA_AI/data/Cyto_Report_summary2.xlsx'

	cnv_df = cnv_obj.build_cnv_training_data(data_dir, outcome_file)

	print(cnv_df)
	print(type(cnv_df))

def get_tiff_file():
	## banana path
	# file_path_ary = ['/home/ryan/smb_data/CytoOneArray/RD/完成報告','/home/ryan/smb_data/CytoOneArray/RD/審查中報告','/home/ryan/smb_data/brank_data/For Brank/GPR']

	## local path
	file_path_ary = ["/mnt/nas_share/ryan_data/cyto_tif"]

	# file_path = '/home/ryan/smb_data/CytoOneArray/RD/完成報告/華聯/2014'
	file_ext = 'tif'
	gpr_file_list = '/home/ryan/src_dir/CytoOA_AI/data/local_all_tiff_file_list.txt'
	missing_file_list = '/home/ryan/src_dir/CytoOA_AI/data/local_tiff_missing_file_list.txt'
	match_file_list = '/home/ryan/src_dir/CytoOA_AI/data/local_tiff_match_file_list.txt'
	file_obj = FileList()
	file_ary = []
	for file_path in file_path_ary:
		file_ary += file_obj.find_file(file_path,file_ext)


	fh_writer = open(gpr_file_list,'w')
	fh_missing = open(missing_file_list,'w')
	fh_match = open(match_file_list,'w')

	all_id = []
	array_id_2_path_dict = {}

	for temp_file in file_ary:
		# print(temp_file)

		fh_writer.write(temp_file +"\n")
		file_name = file_obj.get_gpr_code_from_tif(temp_file)
		# print(file_name)

		### recording array id to file path
		array_id_2_path_dict[file_name[0]] = temp_file
		all_id += file_name

	fh_writer.close()
	# print(all_id)
	gpr_id_dict = list_2_dict(all_id)

	###
	excel_reader = ExcelReader()
	input_file = '/home/ryan/src_dir/CytoOA_AI/data/Cyto_Report_summary2.xlsx'
	# input_file = '/home/ryan/src_dir/CytoOA_AI/data/Cyto_Report_summary.xls'
	excel_df = excel_reader.read_excel(input_file)

	excel_gpr_id = get_gpr_id_from_excel_df(excel_df)

	excel_grp_dict = list_2_dict(excel_gpr_id)

	hit_count = 0
	miss_count = 0
	total_count = 0
	miss_id = []
	for key,value in excel_grp_dict.items():

		if key in gpr_id_dict:
			hit_count +=1
			fh_match.write(str(key) +"\t" +array_id_2_path_dict[key] +"\n")
		else:
			miss_count+=1
			miss_id.append(key)
			fh_missing.write(str(key) +"\n")
		total_count +=1

	print("Missing id = ")
	print(miss_id)

	print("Hit count = {}".format(hit_count))
	print("Miss_count = {}".format(miss_count))
	print("Total_count = {}".format(total_count))

	fh_writer.close()
	fh_missing.close()
	fh_match.close()

def gpr_file_test():
	file_path_ary = ['/home/ryan/smb_data/CytoOneArray/RD/完成報告','/home/ryan/smb_data/CytoOneArray/RD/審查中報告','/home/ryan/smb_data/brank_data/For Brank/GPR']
	# file_path_ary = ["/home/ryan/smb_data/brank_data/For Brank/GPR"]

	# file_path = '/home/ryan/smb_data/CytoOneArray/RD/完成報告/華聯/2014'
	file_ext = 'gpr'
	gpr_file_list = '/home/ryan/src_dir/CytoOA_AI/data/gpr_file_list.txt'
	missing_file_list = '/home/ryan/src_dir/CytoOA_AI/data/missing_file_list.txt'
	match_file_list = '/home/ryan/src_dir/CytoOA_AI/data/match_file_list.txt'
	file_obj = FileList()
	file_ary = []
	for file_path in file_path_ary:
		file_ary += file_obj.find_file(file_path,file_ext)


	fh_writer = open(gpr_file_list,'w')
	fh_missing = open(missing_file_list,'w')
	fh_match = open(match_file_list,'w')

	all_id = []
	array_id_2_path_dict = {}

	for temp_file in file_ary:
		# print(temp_file)

		fh_writer.write(temp_file +"\n")
		file_name = file_obj.get_gpr_code(temp_file)
		# print(file_name)

		### recording array id to file path
		array_id_2_path_dict[file_name[0]] = temp_file
		all_id += file_name

	fh_writer.close()
	# print(all_id)
	gpr_id_dict = list_2_dict(all_id)

	###
	excel_reader = ExcelReader()
	input_file = '/home/ryan/src_dir/CytoOA_AI/data/Cyto_Report_summary2.xlsx'
	# input_file = '/home/ryan/src_dir/CytoOA_AI/data/Cyto_Report_summary.xls'
	excel_df = excel_reader.read_excel(input_file)

	excel_gpr_id = get_gpr_id_from_excel_df(excel_df)

	excel_grp_dict = list_2_dict(excel_gpr_id)

	hit_count = 0
	miss_count = 0
	total_count = 0
	miss_id = []
	for key,value in excel_grp_dict.items():

		if key in gpr_id_dict:
			hit_count +=1
			fh_match.write(str(key) +"\t" +array_id_2_path_dict[key] +"\n")
		else:
			miss_count+=1
			miss_id.append(key)
			fh_missing.write(str(key) +"\n")
		total_count +=1

	print("Missing id = ")
	print(miss_id)

	print("Hit count = {}".format(hit_count))
	print("Miss_count = {}".format(miss_count))
	print("Total_count = {}".format(total_count))

	fh_writer.close()
	fh_missing.close()
	fh_match.close()

# if __name__ == "__main__":
# 	# file_test()
# 	# gpr_file_test()
# 	# cnv_data_reader()
# 	get_tiff_file()


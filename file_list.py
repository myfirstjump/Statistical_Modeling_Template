# -*- coding: utf8 -*-
import re
import time
import csv
import sys
import math
import csv
import os

class FileList(object):

	def get_cbs_all_file(self, file_path):
		'''
		Function:
			抓取cbs_all png檔案.
			有shift時，必須用shift檔案。
		'''
		result_ary = []
		region_file = 'CBS_All.png'
		region_shift_file = 'CBS_All_Shift.png'

		unique_file_name = 'ProbesInfo-40.txt'

		result_ary = self.find_cnv_result_file_core(file_path, region_file, region_shift_file, unique_file_name)

		return result_ary
	def get_case_code_from_path(self, file_path):
		'''
		Input:
			file_path.
		Output:
			case_code: [\w{6}-\w{2}-\w{3}-\w\w], 10 digitals
		'''
		array_id = []

		case_id = ''
		case_code = re.findall(r"(\w{6}-\d{2}-\d{3})", file_path)

		hit_count = len(case_code)
		if hit_count ==0:
			array_id.append(0)
			print("!!!! case_code no hit ..")
			print(file_path)
		elif hit_count >1:
			# print("#### case_code Multi hit ..")
			# print(file_path)
			array_id.append(case_code[0])
			case_id = array_id[0]
		else:
			array_id.append(case_code[0])
			case_id = array_id[0]

		return case_id
	def find_cnv_result_file_core(self, file_path, input_file, input_shift_file, unique_file_name):
		result_ary = []
		# region_file = input_file
		# region_shift_file = input_shift_file
		# temp_file = unique_file_name

		target_path = ''

		result_list = self.find_file_dirname(file_path, unique_file_name)
		for temp_path in result_list:
			no_shift_temp_path = os.path.join(temp_path, input_file)
			shfit_temp_path = os.path.join(temp_path, input_shift_file)

			if os.path.isfile(shfit_temp_path):
				print('##############')
				print(shfit_temp_path)
				target_path = shfit_temp_path
			else:
				target_path = no_shift_temp_path

			result_ary.append(target_path)

		return result_ary


	def get_all_probe_bind_file(self, file_path):
		result_ary = []
		region_file = 'All_Probe_Bind.txt'
		region_shift_file = 'All_Probe_Bind_Shift.txt'

		unique_file_name = 'ProbesInfo-40.txt'

		result_ary = self.find_cnv_result_file_core(file_path, region_file, region_shift_file, unique_file_name)

		return result_ary

	def get_region_40_file(self, file_path):
		result_ary = []
		region_file = 'Region-40.txt'
		region_shift_file = 'Region-40_Shift.txt'

		unique_file_name = 'ProbesInfo-40.txt'

		result_ary = self.find_cnv_result_file_core(file_path, region_file, region_shift_file, unique_file_name)

		return result_ary
	def get_region_40_file_bak(self, file_path):
		result_ary = []
		region_file = 'Region-40.txt'
		region_shift_file = 'Region-40_Shift.txt'

		temp_file = 'ProbesInfo-40.txt'

		target_path = ''

		result_list = self.find_file_dirname(file_path, temp_file)
		for temp_path in result_list:
			no_shift_temp_path = os.path.join(temp_path, region_file)
			shfit_temp_path = os.path.join(temp_path, region_shift_file)

			if os.path.isfile(shfit_temp_path):
				print('##############')
				print(shfit_temp_path)
				target_path = shfit_temp_path
			else:
				target_path = no_shift_temp_path

			result_ary.append(target_path)

		return result_ary

	def get_gpr_code_from_path(self, file_path):
		'''
		Input:
			file_path.
		Output:
			gpr_code: [], 10 digitals
		'''
		array_id = []

		gpr_code = re.findall(r"(\d{10})", file_path)

		hit_count = len(gpr_code)
		if hit_count ==0:
			array_id.append(0)
			print("!!!! no hit ..")
			print(file_path)
		elif hit_count >1:
			print("#### Multi hit ..")
			print(file_path)
		else:
			array_id.append(gpr_code[0])

		return array_id
	def parse_probe_checking_result(self,input_file):
		'''
		result_ary=[[probe_id,chr_str,int(start_str),int(end_str)],[]]
		'''
		print("Input file = {0}".format(input_file))
		with open(input_file) as input_desc:
			## not contain \n
			content = input_desc.read().splitlines()

		## remove header
		del content[0]

		result_ary = []
		row_count = 0
		for temp_str in content:
			temp_ary = temp_str.split('\t')

			probe_id = temp_ary[0]
			chr_str,start_str,end_str = temp_ary[7:10]
			start_str = int(start_str)
			end_str = int(end_str)

			## minus strand, swap start and end
			if start_str > end_str:
				start_str,end_str = end_str,start_str

			result_ary.append([probe_id,chr_str,start_str,end_str])

			# result_ary.append([chr_str,int(start_str),int(end_str)])
			# print(chr_str,start_str,end_str)

			row_count += 1

		print("Total = {0}".format(row_count))
		return result_ary

	def walk_path(self,file_path):
		result_list = []

		#file_path = '//BANANA/Service/Service(New)/Paid'
		#file_path = '//BANANA/RD_Product/rice'
		for dirname, dirnames, filenames in os.walk(file_path):
			# print path to all subdirectories first.
			#for subdirname in dirnames:
			#    print(os.path.join(dirname, subdirname))

			# print path to all filenames.
			for filename in filenames:
				print(os.path.join(dirname, filename))
				result_list.append(os.path.join(dirname, filename))


		return result_list

	def find_file_dirname(self,file_path,extension_name):
		record_dict = {}

		result_list = []

		for dirname, dirnames, filenames in os.walk(file_path):
			# print path to all filenames.
			for filename in filenames:
				if(str(filename).endswith(extension_name)):
					if not dirname in record_dict:
						result_list.append(dirname)
						record_dict[dirname] = dirname
					#print(os.path.join(dirname, filename))

		#for file_str in result_list:
		#	print(file_str)

		return result_list
	def find_file(self,file_path,extension_name):
		result_list = []

		for dirname, dirnames, filenames in os.walk(file_path):
			# print path to all filenames.
			for filename in filenames:
				if(str(filename).endswith(extension_name)):
					result_list.append(os.path.join(dirname, filename))
					#print(os.path.join(dirname, filename))

		#for file_str in result_list:
		#	print(file_str)

		return result_list

	def get_gpr_code_from_tif(self, file_path, delimeter="/"):
		'''
		'''
		temp_result = ''
		temp_ary = file_path.split(delimeter)
		array_id = []

		ary_len = len(temp_ary)
		file_index = ary_len -1
		dir_index = ary_len -2

		gpr_file = temp_ary[file_index]
		temp_file_name = gpr_file.replace('.tif','')

		# gpr_code = re.findall(r"(\d{10})", temp_file_name)
		gpr_code = re.findall(r"(\d{10})", gpr_file)
		# print(temp_ary)
		# print(temp_result)
		if len(gpr_code) ==0:
			array_id.append(0)
			print("!!!! no hit ..")
			print(gpr_file)
		elif len(gpr_code) >1:
			print("#### Multi hit ..")
			print(gpr_file)
			array_id.append(gpr_code[0])
		else:
			array_id.append(gpr_code[0])

		return array_id
	def get_gpr_code(self, file_path, delimeter="/"):
		'''
		'''
		temp_result = ''
		temp_ary = file_path.split(delimeter)
		array_id = []

		ary_len = len(temp_ary)
		file_index = ary_len -1
		dir_index = ary_len -2

		gpr_file = temp_ary[file_index]
		temp_file_name = gpr_file.replace('.gpr','')

		# gpr_code = re.findall(r"(\d{10})", temp_file_name)
		gpr_code = re.findall(r"(\d{10})", gpr_file)
		# print(temp_ary)
		# print(temp_result)
		if len(gpr_code) ==0:
			array_id.append(0)
			print("!!!! no hit ..")
			print(gpr_file)
		elif len(gpr_code) >1:
			print("#### Multi hit ..")
			print(gpr_file)
		else:
			array_id.append(gpr_code[0])

		return array_id
	def get_gpr_file_name(self, file_path, delimeter="/"):
		temp_result = ''
		temp_ary = file_path.split(delimeter)


		ary_len = len(temp_ary)
		file_index = ary_len -1
		dir_index = ary_len -2

		temp_file_name =temp_ary[file_index].replace('.gpr','')
		temp_result = temp_ary[dir_index] + "_" + temp_file_name

		# print(temp_ary)
		# print(temp_result)

		return temp_result

	def get_region_40_file_name(self, file_path, delimeter="/"):
		temp_result = ''
		temp_ary = file_path.split(delimeter)


		ary_len = len(temp_ary)
		file_index = ary_len -1
		dir_index = ary_len -2

		temp_file_name =temp_ary[file_index].replace('.txt','').split('_')[2]
		temp_result = temp_ary[dir_index] + "_" + temp_file_name

		# print(temp_ary)
		# print(temp_result)

		return temp_result


# def region_40_file_test():
# 	file_path = '/home/ryan/gpr_test/test_case'
# 	file_ext = 'txt'
# 	file_obj = FileList()
# 	file_ary = file_obj.find_file(file_path,file_ext)

# 	for temp_file in file_ary:
# 		file_name = file_obj.get_region_40_file_name(temp_file)
# 		print(file_name)


# def file_test():
# 	file_path = '/home/ryan/gpr_test'
# 	file_ext = 'gpr'
# 	file_obj = FileList()
# 	file_ary = file_obj.find_file(file_path,file_ext)

# 	for temp_file in file_ary:
# 		file_name = file_obj.get_gpr_file_name(temp_file)


# if __name__ == "__main__":
# 	# file_test()
# 	# region_40_file_test()


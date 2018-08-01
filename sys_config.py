# -*- coding: utf8 -*-

class SysConfig(object):

	def __init__(self):

		### need to modify while new run
		self.working_dir = '/app'

		### model dir
		# self.model_output_dir = self.working_dir +'/data/model/RF_3000/'
		# self.model_output_dir = self.working_dir +'/data/model/Gain_loss_RF_3000/'
		self.model_output_dir = self.working_dir +'/data/model/CNV_only_3000/'

		### log file
		# self.log_file = self.working_dir +'/data/RF_oa_gain_loss_2_3000_log.txt'
		self.log_file = self.working_dir +'/data/CNV_only_3000_log.txt'

		self.cnv_output_dir = '/home/ryan/src_dir/CytoCloudR/tmp/gene/output'

		# self.cnv_output_dir = '/home/ryan/src_dir/CytoCloudR/tmp/gene/test_output'
		self.outcome_file = '/home/ryan/src_dir/CytoOA_AI/data/Cyto_Report_summary2.xlsx'

		self.sampling_ratio = 0.8
		self.test_set_ratio = 0.2

		### y label
		self.y_key = 'cnv_outcome'

		self.tif_data_dir = './'

		###
		self.df_output_file = '/home/ryan/src_dir/CytoCloudR/tmp/gene/cnv_df.csv'
		self.df_output_file = '/home/ryan/src_dir/CytoCloudR/tmp/gene/cnv_df_all.csv'
	### method
	def get_log_file(self):
		return self.log_file

	def get_model_output_dir(self):
		return self.model_output_dir

	def get_df_output_file(self):
		return self.df_output_file
	def get_y_key(self):
		return self.y_key

	def get_test_set_ratio(self):
		return self.test_set_ratio

	def get_sampling_ratio(self):
		return self.sampling_ratio

	def get_cnv_output_dir(self):
		return self.cnv_output_dir

	def get_outcome_file(self):
		return self.outcome_file

	def get_tif_data_dir(self):
		return self.tif_data_dir
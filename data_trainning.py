# -*- coding: utf8 -*-

import csv
import hashlib

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

## AdaBoostRegressor

from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics, preprocessing
from tensorflow.python.data import Dataset
from tensorflow.python.estimator.inputs import numpy_io

### keras
# from tensorflow.keras.Sequential import Sequential
# import tensorflow.python.keras.models.Sequential as Sequential
# from tensorflow.python.keras.layers import Dense

from keras.models import Sequential
from keras import initializers, regularizers
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.optimizers import Adam, Adagrad
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.preprocessing import LabelEncoder

from sklearn.utils import shuffle

### xgboost
import xgboost
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from data_sampling import DataSampling
from data_processing import DataProcessing


class DataTrainning(object):

	def __init__(self):
		self.data_selection_obj = DataProcessing()
		# pass
		## print(tf.__version__)

	def xgboot_training(self, train_set_x, train_set_y, test_set_x, test_set_y):

		print("In xgboot_training ...")
		train_set_x, train_set_y = shuffle(train_set_x, train_set_y, random_state=0)

		train_set_x = train_set_x.as_matrix()
		train_set_y = train_set_y.as_matrix()

		# train_set_x = preprocessing.scale(train_set_x)

		### one_hot encoding
		train_set_y_one_hot = train_set_y
		# train_set_y_one_hot = to_categorical(train_set_y)

		# print("train_set_y_one_hot")
		# print(train_set_y_one_hot)

		scaler = preprocessing.StandardScaler().fit(train_set_x)
		train_set_x = scaler.transform(train_set_x)

		model = xgboost.XGBClassifier()
		model.fit(train_set_x, train_set_y)

		print(model)

		### test data
		test_set_x = scaler.transform(test_set_x)
 
		# make predictions for test data
		y_pred = model.predict(test_set_x)
		predictions = [np.round(value) for value in y_pred]
		# evaluate predictions

		# encode Y class values as integers
		# label_encoder = LabelEncoder()
		# label_encoder = label_encoder.fit(test_set_y)
		# label_encoded_y = label_encoder.transform(test_set_y)
		accuracy = accuracy_score(test_set_y, predictions)
		print("Accuracy: %.2f%%" % (accuracy * 100.0))

		return model

	def keras_test(self, train_set_x, train_set_y, my_input_dim=37627, my_epochs=50, my_batch=128):

		train_set_x, train_set_y = shuffle(train_set_x, train_set_y, random_state=0)

		train_set_x = train_set_x.as_matrix()
		train_set_y = train_set_y.as_matrix()

		# train_set_x = preprocessing.scale(train_set_x)

		### one_hot encoding
		train_set_y_one_hot = train_set_y
		# train_set_y_one_hot = to_categorical(train_set_y)

		print("train_set_y_one_hot")
		print(train_set_y_one_hot)

		scaler = preprocessing.StandardScaler().fit(train_set_x)
		train_set_x = scaler.transform(train_set_x)


		VALIDATION_SPLIT = 0.2
		model = Sequential()
		model.add(Dense(1024, input_dim=my_input_dim, activation='relu'))
		# model.add(Dense(8, input_dim=my_input_dim, activation='relu'))
		# model.add(BatchNormalization())
		# model.add(Dense(4, activation='relu'))
		# model.add(Dropout(0.5))
		# model.add(BatchNormalization())
		model.add(Dense(256, activation='relu'))
		# # model.add(BatchNormalization())
		model.add(Dense(128, activation='relu'))
		# model.add(BatchNormalization())
		model.add(Dense(32, activation='relu'))
		model.add(BatchNormalization())
		# model.add(Dense(100, activation='relu'))
		# model.add(Dense(100, activation='relu'))
		# model.add(Dense(100, activation='relu'))
		# model.add(Dense(100, activation='relu'))
		# model.add(Dense(100, activation='relu'))
		# model.add(Dense(100, activation='relu'))
		# model.add(Dense(100, activation='relu'))

		# model.add(Dense(2, activation='sigmoid'))
		model.add(Dense(1, activation='sigmoid'))
		# Compile model
		# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-6), metrics=['accuracy'])

		### SGD(lr=0.0001, momentum=0.9)
		#optimizer=Adam(lr=1e-6)

		# model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
		
		# Fit the model
		## validation_split=VALIDATION_SPLIT
		# model.fit(train_set_x, train_set_y, epochs=150, batch_size=10)
		# model.fit(train_set_x, train_set_y, epochs=20, batch_size=940)
		
		model.fit(train_set_x, train_set_y_one_hot, epochs=my_epochs, batch_size=my_batch)
		
		# evaluate the model
		scores = model.evaluate(train_set_x, train_set_y_one_hot)
		print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

		### prediction
		# probabilities = model.predict(X)
		# predictions = [float(numpy.round(x)) for x in probabilities]
		# accuracy = numpy.mean(predictions == Y)

		return model,scaler

	def SKRandomForest_Category(self, train_set_x, train_set_y):
		print("SKRandomForest_Category ...")
		# NUM_STEPS = 1000
		# MINIBATCH_SIZE = 20
		NUM_TREE = 3000
		# MAX_NODE = 1000
		NUM_FEATURES = 'auto'
		# NUM_CLASS = 1
		MAX_DEPTH = None
		sampling_obj = DataSampling()
		my_model = RandomForestClassifier(
			n_estimators = NUM_TREE,
			max_features = NUM_FEATURES,
			min_samples_split = 2,
			max_depth = MAX_DEPTH,
			oob_score = False
			)
		
		### training
		print("Before fitting")
		my_model.fit(train_set_x,train_set_y)
		train_score = my_model.score(train_set_x,train_set_y)
		print("After fitting")

		print("train_score = {} ".format(train_score))
		print("feature_importances_ = {} ".format(my_model.feature_importances_))
		print("n_features_ =  {} ".format(my_model.n_features_ ))
		# print("oob_score_ = {} ".format(my_model.oob_score_ ))
		

		# ### prediction
		# test_prediction_result = my_model.predict(test_set_feature)
		# # print("test_prediction_result = {} ".format(test_prediction_result))

		return my_model

	def SKRandomForest(self, df_1, df_2, target_name, feature_dict):

		print("SKRandomForest ...")
		# NUM_STEPS = 1000
		# MINIBATCH_SIZE = 20
		NUM_TREE = 2000
		# MAX_NODE = 1000
		NUM_FEATURES = 24
		# NUM_CLASS = 1
		MAX_DEPTH = None
		sampling_obj = DataSampling()

		### modify target
		df_1 = self.data_selection_obj.cast_to_numeric(df_1, ['Blast_count'])
		df_2 = self.data_selection_obj.cast_to_numeric(df_2, ['Blast_count'])

		### Sampling
		(train_set, train_label, test_set, test_label) = sampling_obj.cbc_sampling_pipeline(df_1, df_2, target_name, sample_ratio=0.8)

		### feature columns
		numeric_fields = feature_dict['numeric']
		category_fields = feature_dict['category']
		all_features = numeric_fields + category_fields

		train_set_feature = train_set.loc[:,all_features]
		train_set_feature = self.data_selection_obj.cast_to_numeric(train_set_feature, numeric_fields)
		train_set_feature = train_set_feature.reindex()

		test_set_feature = test_set.loc[:,all_features]
		test_set_feature = self.data_selection_obj.cast_to_numeric(test_set_feature, numeric_fields)
		test_set_feature = test_set_feature.reindex()

		train_feature_columns = self.build_feature_columns(feature_dict)

		my_model = RandomForestRegressor(
			n_estimators = NUM_TREE,
			max_features = NUM_FEATURES,
			min_samples_split = 2,
			max_depth = MAX_DEPTH,
			oob_score = True
			)
		
		### training
		print("Before fitting")
		my_model.fit(train_set_feature,train_label)
		train_score = my_model.score(train_set_feature,train_label)
		print("After fitting")

		print("feature_importances_ = {} ".format(my_model.feature_importances_))
		print("n_features_ =  {} ".format(my_model.n_features_ ))
		print("oob_score_ = {} ".format(my_model.oob_score_ ))
		print("train_score = {} ".format(train_score))

		### prediction
		test_prediction_result = my_model.predict(test_set_feature)
		# print("test_prediction_result = {} ".format(test_prediction_result))

		for (pair_1,pari2) in zip(test_label,test_prediction_result):
			print("real,predict = {},{} ".format(pair_1,pari2))

	def build_feature_columns(self, feature_dict):
		"""
		"""
		# numeric_fields = feature_dict['numeric']
		# category_fields = feature_dict['category']
		numeric_set = []
		category_set = []

		if "numeric" in feature_dict:
			numeric_set = [tf.feature_column.numeric_column(my_feature) for my_feature in feature_dict['numeric']]

		if "category" in feature_dict:
			category_set = [tf.feature_column.categorical_column_with_hash_bucket(my_feature, hash_bucket_size=10) for my_feature in feature_dict['category']]

		feature_set = set(numeric_set + category_set)

		return feature_set


	def randomforest_data_func(self, features, targets, batch_size=1, shuffle=True, num_epochs=None):
		features = {key:np.array(value) for key,value in dict(features).items()}
		# ds = Dataset.from_tensor_slices((features,targets))
		# ds = ds.batch(batch_size).repeat(num_epochs)
		
		fn_ref = numpy_io.numpy_input_fn(
			x=features,
			y=targets,
			batch_size=batch_size,
			num_epochs=None,
			shuffle=True)

		return fn_ref
	
	def data_func(self, features, targets, batch_size=1, shuffle=True, num_epochs=None):
		features = {key:np.array(value) for key,value in dict(features).items()}
		ds = Dataset.from_tensor_slices((features,targets))
		ds = ds.batch(batch_size).repeat(num_epochs)

		if shuffle:
			ds = ds.shuffle(100)

		features, labels = ds.make_one_shot_iterator().get_next()

		return (features,labels)

	def test_func(self, features, targets, batch_size=1, num_epochs=1):
		features = {key:np.array(value) for key,value in dict(features).items()}
		ds = Dataset.from_tensor_slices((features,targets))
		ds = ds.batch(batch_size).repeat(num_epochs)

		features, labels = ds.make_one_shot_iterator().get_next()

		return (features,labels)
	def standarizer(self):
		pass
		# boston.data = "df"
		# x_data = preprocessing.StandardScaler().fit_transform(boston.data)

	def DNNRegressor(self, df_1, df_2, target_name, feature_dict):

		print(" DNNRegressor ...")
		NUM_STEPS = 1000
		MINIBATCH_SIZE = 20
		NUM_TREE = 2000
		MAX_NODE = 1000
		NUM_FEATURES = 24
		NUM_CLASS = 1
		sampling_obj = DataSampling()

		### modify target
		df_1 = self.data_selection_obj.cast_to_numeric(df_1, ['Blast_count'])
		df_2 = self.data_selection_obj.cast_to_numeric(df_2, ['Blast_count'])

		### Sampling
		(train_set, train_label, test_set, test_label) = sampling_obj.cbc_sampling_pipeline(df_1, df_2, target_name, sample_ratio=0.8)

		### feature columns
		numeric_fields = feature_dict['numeric']
		category_fields = feature_dict['category']
		all_features = numeric_fields + category_fields

		train_set_feature = train_set.loc[:,all_features]
		train_set_feature = self.data_selection_obj.cast_to_numeric(train_set_feature, numeric_fields)
		train_set_feature = train_set_feature.reindex()

		test_set_feature = test_set.loc[:,all_features]
		test_set_feature = self.data_selection_obj.cast_to_numeric(test_set_feature, numeric_fields)
		test_set_feature = test_set_feature.reindex()

		train_feature_columns = self.build_feature_columns(feature_dict)

		# print(train_feature_columns)
		# print(len(train_feature_columns))

		# print("train_set_feature = {0}".format(train_set_feature.columns))

		# print("train_set_feature len = {0}".format(len(train_set_feature)))
		# print("train_label len = {0}".format(len(train_label)))

		# print("DF NA checking ...")
		# print(train_set_feature.isnull().any().any())


		reg = tf.estimator.DNNRegressor(
			feature_columns=train_feature_columns,
			# hidden_units=[100, 100, 100, 50, 20, 10],
			hidden_units=[30,30, 30, 20, 10],
			optimizer=tf.train.ProximalAdagradOptimizer(
			learning_rate=0.01,
			l1_regularization_strength=0.001
			)
		)

		train_fn = lambda: self.data_func(train_set_feature, train_label, batch_size=MINIBATCH_SIZE)

		test_fn = lambda: self.test_func(test_set_feature, test_label)

		### training
		reg.train(input_fn=train_fn, steps=NUM_STEPS)

		## evaulate
		eval_result  = reg.evaluate(input_fn=test_fn)
		average_loss = eval_result["average_loss"]
		print("average_loss = {0}".format(average_loss))
		print("\nRMSE for the test set: {:.2f}".format(average_loss**0.5))
	
		# ## prediction
		# predict_set = dict(test_set_feature.head(10))
		# predict_set = {key:np.array(value) for key,value in predict_set.items()}
		# predict_input_fn = tf.estimator.inputs.numpy_input_fn(predict_set, shuffle=False)
		# predict_results = reg.predict(input_fn=predict_input_fn)
		

	def RandomForest(self, df_1, df_2, target_name, feature_dict):

		NUM_STEPS = 1000
		MINIBATCH_SIZE = 20
		NUM_TREE = 2000
		MAX_NODE = 1000
		NUM_FEATURES = 24
		NUM_CLASS = 1
		sampling_obj = DataSampling()

		### modify target
		df_1 = self.data_selection_obj.cast_to_numeric(df_1, ['Blast_count'])
		df_2 = self.data_selection_obj.cast_to_numeric(df_2, ['Blast_count'])

		### Sampling
		(train_set, train_label, test_set, test_label) = sampling_obj.cbc_sampling_pipeline(df_1, df_2, target_name, sample_ratio=0.8)

		### feature columns
		numeric_fields = feature_dict['numeric']
		category_fields = feature_dict['category']
		all_features = numeric_fields + category_fields

		train_set_feature = train_set.loc[:,all_features]
		train_set_feature = self.data_selection_obj.cast_to_numeric(train_set_feature, numeric_fields)
		train_set_feature = train_set_feature.reindex()

		test_set_feature = test_set.loc[:,all_features]
		test_set_feature = self.data_selection_obj.cast_to_numeric(test_set_feature, numeric_fields)
		test_set_feature = test_set_feature.reindex()

		train_feature_columns = self.build_feature_columns(feature_dict)

		print(train_feature_columns)
		print(len(train_feature_columns))

		print("train_set_feature = {0}".format(train_set_feature.columns))

		print("train_set_feature len = {0}".format(len(train_set_feature)))
		print("train_label len = {0}".format(len(train_label)))

		print("DF NA checking ...")
		print(train_set_feature.isnull().any().any())

		
		params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
			num_classes = NUM_CLASS,
			num_features= NUM_FEATURES,
			num_trees = NUM_TREE,
			regression = True,			
			max_nodes = MAX_NODE)

		graph_builder_class = tf.contrib.tensor_forest.python.tensor_forest.RandomForestGraphs
		my_model = tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator(params, graph_builder_class=graph_builder_class)

		# train_fn = lambda: self.data_func(train_set_feature, train_label, batch_size=MINIBATCH_SIZE)

		# test_fn = lambda: self.test_func(test_set_feature, test_label)

		train_fn = self.randomforest_data_func(train_set_feature, train_label,batch_size=MINIBATCH_SIZE)
		### training
		print("Before fitting")
		my_model.fit(input_fn=train_fn, steps=None)
		print("After fitting")

		# ## evaulate
		# eval_result  = reg.evaluate(input_fn=test_fn)
		# average_loss = eval_result["average_loss"]
		# print("average_loss = {0}".format(average_loss))
		# print("\nRMSE for the test set: {:.2f}".format(average_loss**0.5))
	
		# ## prediction
		# predict_set = dict(test_set_feature.head(10))
		# predict_set = {key:np.array(value) for key,value in predict_set.items()}
		# predict_input_fn = tf.estimator.inputs.numpy_input_fn(predict_set, shuffle=False)
		# predict_results = reg.predict(input_fn=predict_input_fn)

	def LinearRegressor(self, df_1, df_2, target_name, feature_dict):
		# train_numeric_list = ["Age", "Blast",
		# 	"WBC", "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "RDW", "PLT", "MPV", "NE", "LY", 
		# 	"MO", "EO", "BA", "NE_c", "LY_c", "MO_c", "EO_c", "BA_c", "NRBC", "NRBC_c",
		# 	"Left Shift 3"]
		NUM_STEPS = 200
		MINIBATCH_SIZE = 20

		sampling_obj = DataSampling()

		
		(train_set, train_label, test_set, test_label) = sampling_obj.cbc_sampling_pipeline(df_1, df_2, target_name, sample_ratio=0.8)

		# train_label = train_label.to_frame()
		# print(train_label)
		# print(type(train_label))

		### feature columns
		numeric_fields = feature_dict['numeric']
		category_fields = feature_dict['category']
		all_features = numeric_fields + category_fields

		train_set_feature = train_set.loc[:,all_features]
		train_set_feature = self.data_selection_obj.cast_to_numeric(train_set_feature, numeric_fields)
		train_set_feature = train_set_feature.reindex()

		test_set_feature = test_set.loc[:,all_features]
		test_set_feature = self.data_selection_obj.cast_to_numeric(test_set_feature, numeric_fields)
		test_set_feature = test_set_feature.reindex()

		train_feature_columns = self.build_feature_columns(feature_dict)

		print(train_feature_columns)

		print("train_set_feature = {0}".format(train_set_feature.columns))

		print("train_set_feature len = {0}".format(len(train_set_feature)))
		print("train_label len = {0}".format(len(train_label)))

		print("DF NA checking ...")
		print(train_set_feature.isnull().any().any())
		# print(train_set_feature['BA_c'])

		# reg = tf.estimator.LinearRegressor(
		# 	feature_columns=train_feature_columns,
		# 	optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1)
		# )

		reg = tf.estimator.LinearRegressor(
			feature_columns=train_feature_columns
		)

		train_fn = lambda: self.data_func(train_set_feature, train_label, batch_size=MINIBATCH_SIZE)

		test_fn = lambda: self.test_func(test_set_feature, test_label)

		### training
		reg.train(input_fn=train_fn, steps=NUM_STEPS)

		## evaulate
		eval_result  = reg.evaluate(input_fn=test_fn)
		average_loss = eval_result["average_loss"]
		print("average_loss = {0}".format(average_loss))
		print("\nRMSE for the test set: {:.2f}".format(average_loss**0.5))
	
		## prediction
		predict_set = dict(test_set_feature.head(10))
		predict_set = {key:np.array(value) for key,value in predict_set.items()}
		predict_input_fn = tf.estimator.inputs.numpy_input_fn(predict_set, shuffle=False)
		predict_results = reg.predict(input_fn=predict_input_fn)
		
		# for key,value in predict_set.items():
		# 	print("\n##########\n")
		# 	print(key)
		# 	print(value)
		# print("\nPrediction results:")
		# for i, prediction in enumerate(predict_results):
		# 	# print("real = {0}, prediction = {1}".format(test_set_feature.loc[i,"Blast_count"],prediction["predictions"][0]))
		# 	print(" prediction = {0}".format(prediction["predictions"][0]))
			
		

		## 11 age group buckets (from age 17 and below, 18-24, 25-29, ..., to 65 and over)
		# age = tf.feature_column.numeric_column('age')
		# age_buckets = tf.feature_column.bucketized_column( age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
		##
		# age_buckets_x_education_x_occupation = tf.feature_column.crossed_column( [age_buckets, 'education', 'occupation'], hash_bucket_size=1000)

		# Category field
		# occupation = tf.feature_column.categorical_column_with_hash_bucket('occupation', hash_bucket_size=1000)
		# relationship = tf.feature_column.categorical_column_with_vocabulary_list( 'relationship', ['Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried','Other-relative'])

		###  numeric field
		# age = tf.feature_column.numeric_column('age')
		# education_num = tf.feature_column.numeric_column('education_num')
		# capital_gain = tf.feature_column.numeric_column('capital_gain')
		# capital_loss = tf.feature_column.numeric_column('capital_loss')
		# hours_per_week = tf.feature_column.numeric_column('hours_per_week')


		# reg = learn.LinearRegressor(
		# 	feature_columns=feature_columns,
		# 	optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1)
		# )

		# reg.fit(x_data, boston.target, steps=NUM_STEPS,batch_size=MINIBATCH_SIZE)

		# MSE = reg.evaluate(x_data, boston.target, steps=1)

	def cnn_apply(self, train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y):
		
		# convolution layer params
		INPUT_WIDTH = 1220
		INPUT_LENGTH = 432
		KERNEL_SIZE = 3 # (432 + 2*padding - kernel)/strides + 1 = 432
		POOL_SIZE = 2
		CONV_STRIDES = 1
		POOL_STRIDES = 2 # pooling strides should be same with the pooling size

		# training params
		BATCH_SIZE = 24 # 1220x432下 上限 27
		EPOCHS = 50
		LEARNING_RATE = 0.005
		LASSO_STRENGTH = 1
		RIDGE_STRENGTH = 0.5
		EARLY_STOP_PATIENCE = 6
		ACTIVATION_fnc = 'relu'

		KERNEL_MEAN = 0
		KERNEL_STD = 0.001
		BIAS_MEAN = 0
		BIAS_STD = 0.001
		SEED = 345
		FC_MEAN = 0
		FC_STD = 0.001

		# print('Data type:', type(train_set_x[0]))
		# Normalization 
		# I do this in data_reader, uint16 pixel values -> divide by 255 to form uint8 pixel values

		# self divide train/valid
		print('Valid set y counts:')
		unique, counts = np.unique(valid_set_y, return_counts = True)
		print(dict(zip(unique, counts)))
		print('Naive classifier will get: {} accuracy.'.format(counts[1]/(counts[0] + counts[1])))

		print('Train set y counts:')
		unique, counts = np.unique(train_set_y, return_counts = True)
		print(dict(zip(unique, counts)))

		model = Sequential()

		# 1st
		model.add(Conv2D(input_shape=(INPUT_LENGTH, INPUT_WIDTH, 1), filters=32, kernel_size=(KERNEL_SIZE,KERNEL_SIZE), strides=(CONV_STRIDES,CONV_STRIDES), padding='same', kernel_regularizer = regularizers.l2(RIDGE_STRENGTH)))
		model.add(BatchNormalization())
		model.add(Activation(ACTIVATION_fnc))
		model.add(MaxPooling2D(pool_size=(POOL_SIZE,POOL_SIZE), strides=(POOL_STRIDES,POOL_STRIDES), padding='same'))
		### If wanna set initializers: kernel_initializer=initializers.Identity(gain=initializers.RandomNormal(mean=KERNEL_MEAN, stddev=KERNEL_STD, seed=SEED)), bias_initializer=initializers.Identity(gain=initializers.RandomNormal(mean=BIAS_MEAN, stddev=BIAS_STD, seed=SEED)
		
		# 2nd
		model.add(Conv2D(filters=64, kernel_size=(KERNEL_SIZE,KERNEL_SIZE), strides=(CONV_STRIDES,CONV_STRIDES), padding='same', kernel_regularizer = regularizers.l2(RIDGE_STRENGTH)))
		model.add(BatchNormalization())
		model.add(Activation(ACTIVATION_fnc))
		model.add(MaxPooling2D(pool_size=(POOL_SIZE,POOL_SIZE), strides=(POOL_STRIDES,POOL_STRIDES), padding='same'))

		# 3rd
		model.add(Conv2D(filters=32, kernel_size=(KERNEL_SIZE,KERNEL_SIZE), strides=(CONV_STRIDES,CONV_STRIDES), padding='same', kernel_regularizer = regularizers.l2(RIDGE_STRENGTH)))
		model.add(BatchNormalization())
		model.add(Activation(ACTIVATION_fnc))
		model.add(MaxPooling2D(pool_size=(POOL_SIZE,POOL_SIZE), strides=(POOL_STRIDES,POOL_STRIDES), padding='same'))

		# 4th
		model.add(Conv2D(filters=16, kernel_size=(KERNEL_SIZE,KERNEL_SIZE), strides=(CONV_STRIDES,CONV_STRIDES), padding='same', kernel_regularizer = regularizers.l2(RIDGE_STRENGTH)))
		model.add(BatchNormalization())
		model.add(Activation(ACTIVATION_fnc))
		model.add(MaxPooling2D(pool_size=(POOL_SIZE,POOL_SIZE), strides=(POOL_STRIDES,POOL_STRIDES), padding='same'))
		
		# 5th
		model.add(Conv2D(filters=16, kernel_size=(KERNEL_SIZE,KERNEL_SIZE), strides=(CONV_STRIDES,CONV_STRIDES), padding='same', kernel_regularizer = regularizers.l2(RIDGE_STRENGTH)))
		model.add(BatchNormalization())
		model.add(Activation(ACTIVATION_fnc))
		model.add(MaxPooling2D(pool_size=(POOL_SIZE,POOL_SIZE), strides=(POOL_STRIDES,POOL_STRIDES), padding='same'))

		# 6th
		model.add(Flatten())
		model.add(Dense(32))#, kernel_initializer=initializers.RandomNormal(mean=FC_MEAN, stddev=FC_STD, seed=None), bias_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)))
		model.add(BatchNormalization())
		model.add(Activation(ACTIVATION_fnc))

		# # 7th
		# model.add(Dense(32))
		# model.add(BatchNormalization())
		# model.add(Activation('relu'))

		model.add(Dense(1, activation='sigmoid'))

		model.compile(loss='binary_crossentropy', optimizer=Adagrad(lr=LEARNING_RATE), metrics=['accuracy'])

		# callbacks 項目
		early_stop = EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE, mode='auto')
		# tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
		tensorboard = TensorBoard(log_dir='./graphs') # 外部執行: tensorboard --logdir='./graphs' --port 6006

		model.fit(train_set_x, train_set_y, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, validation_data=(valid_set_x, valid_set_y), callbacks=[early_stop, tensorboard])

		print('**************Prediction*****************')
		prediction = model.predict(x=test_set_x, batch_size=1, verbose=0)
		predict_df = pd.DataFrame(prediction)
		fn = np.vectorize(lambda y: 0 if y < 0.5 else 1) # apply function point-wise to ndarray
		pred = fn(prediction)

		predict_df = predict_df.assign(pred_label=pred)
		predict_df = predict_df.assign(true_label=test_set_y)
		print(predict_df)
		print(confusion_matrix(test_set_y, pred))

		scores = model.evaluate(test_set_x, test_set_y)
		print('Loss value valid: %.2f' % scores[0])
		print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

		print('Test set y counts:')
		unique, counts = np.unique(test_set_y, return_counts = True)
		print(dict(zip(unique, counts)))

		return model

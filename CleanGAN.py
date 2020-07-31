#!/usr/bin/env python

from Utils import *

import os, sys
import argparse
import glob
from six.moves import map, zip, range
import numpy as np

from tensorpack import *
from tensorpack.utils.viz import *
from tensorpack.utils.gpu import get_nr_gpu

from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.varreplace import freeze_variables
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope

import tensorpack.tfutils.symbolic_functions as symbf
import tensorflow as tf
from GAN import GANTrainer, GANModelDesc, SeparateGANTrainer



from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v2

class Model(GANModelDesc):
	def _get_inputs(self):
		return [InputDesc(tf.float32, (None, 2, DIMY, DIMX), 'inputA'),
				InputDesc(tf.float32, (None, 2, DIMY, DIMX), 'mask'), 
				InputDesc(tf.float32, (None, 2, DIMY, DIMX), 'inputB'), 
				]

	def build_losses(self, vecpos, vecneg, name="WGAN_loss"):
		with tf.name_scope(name=name):
			# the Wasserstein-GAN losses
			d_loss = tf.reduce_mean(vecneg - vecpos, name='d_loss')
			g_loss = tf.negative(tf.reduce_mean(vecneg), name='g_loss')
			# add_moving_summary(self.d_loss, self.g_loss)
			return g_loss, d_loss

	#FusionNet
	@auto_reuse_variable_scope
	def generator(self, img):
		assert img is not None
		return arch_generator(img)
		# return arch_fusionnet(img)

	@auto_reuse_variable_scope
	def discriminator(self, img):
		assert img is not None
		return arch_discriminator(img)

	@auto_reuse_variable_scope
	def expansion(self, img):
		assert img is not None
		return arch_fusionnet(img)
	


	
	def _build_graph(self, inputs):
		A, R, B = inputs

		A = cvt2tanh(A)
		R = cvt2tanh(R)
		B = cvt2tanh(B)

		A = tf.identity(A, name='A')
		R = tf.identity(R, name='R')
		B = tf.identity(B, name='B')



		# use the initializers from torch
		with argscope([Conv2D, Deconv2D, FullyConnected],
					  # W_init=tf.contrib.layers.variance_scaling_initializer(factor=.333, uniform=True),
					  W_init=tf.truncated_normal_initializer(stddev=0.02),
					  use_bias=False), \
				argscope(BatchNorm, gamma_init=tf.random_uniform_initializer()), \
				argscope([Conv2D, Deconv2D, BatchNorm], data_format='NCHW'), \
				argscope(LeakyReLU, alpha=0.2):
			with tf.name_scope('preprocessing'):
				image  = tf.identity(A, name='S01') # For PSNR
				label  = tf.identity(B, name='S02') # For PSNR
			
					
			with tf.variable_scope('gen'):
				with tf.variable_scope('recon'):
					filter1 = self.generator(image)

			filtered_image1 = image - filter1
			updated_image1  = update(filtered_image1, label, R, name='S1')
			
			
			with tf.variable_scope('discrim'):
				S1_dis_real = self.discriminator(label)
				S1_dis_fake = self.discriminator(updated_image1)
				

		with tf.name_scope('losses'):

			with tf.name_scope('Img'):

				with tf.name_scope('Recon'):
					smoothness_AA = tf.reduce_mean(tf.image.total_variation((updated_image1)), name='smoothness_AA')
					background_diff_AA = background_diff(updated_image1, label, name='background_diff_AA')
					signal_diff_AA = signal_diff(updated_image1, label, name='signal_diff_AA')
				
			with tf.name_scope('LossAA'):
				G_loss_AA, D_loss_AA = self.build_losses(S1_dis_real, S1_dis_fake, name='AA')
		
						
		DELTA = 1e-4
		ALPHA = 1e+2
		RATES = tf.count_nonzero(tf.ones_like(R), dtype=tf.float32) / 2 / tf.count_nonzero(R, dtype=tf.float32) 
		self.g_loss = tf.add_n([
								(G_loss_AA),
								(smoothness_AA) * DELTA, 
								(background_diff_AA) * ALPHA * RATES,
								(signal_diff_AA) * ALPHA * RATES
								], name='G_loss_total')
		self.d_loss = tf.add_n([
								(D_loss_AA), 
								], name='D_loss_total')

		wd_g = regularize_cost('gen/.*/W', 		l1_regularizer(1e-5), name='G_regularize')
		wd_d = regularize_cost('discrim/.*/W', 	l1_regularizer(1e-5), name='D_regularize')

		self.g_loss = tf.add(self.g_loss, wd_g, name='g_loss')
		self.d_loss = tf.add(self.d_loss, wd_d, name='d_loss')

	

		self.collect_variables()

		add_moving_summary(self.d_loss, self.g_loss)
		add_moving_summary(
			smoothness_AA,
			background_diff_AA,
			signal_diff_AA
			)

		psnr(tf_complex(cvt2imag(updated_image1)), tf_complex(cvt2imag(label)), maxp=255, name='PSNR_recon')


		def viz3(name, listTensor):
			img = tf.concat(listTensor, axis=3)
			
			out = img
			img = cvt2imag(img)
			img = tf.clip_by_value(img, 0, 255)
			
			tf.summary.image(name+'_real', tf.transpose(img[:,0:1,...], [0, 2, 3, 1]), max_outputs=50)
			tf.summary.image(name+'_imag', tf.transpose(img[:,1:2,...], [0, 2, 3, 1]), max_outputs=50)
				
			return tf.identity(out, name='viz_'+name), tf.identity(img, name='vis_'+name)

		viz_A_recon, vis_A_recon = viz3('A_recon', [R, image, label, updated_image1])
		#viz_B_recon, vis_B_recon = viz3('B_recon', [R, label, M2, S2, T2, tf.abs(S02-M2), tf.abs(S02-S2), tf.abs(S02-T2), Sn2, Sp2, Tn2, Tp2])
		

		#print(S01, R, Rh)
		#print(viz_A_recon, vis_A_recon)
		#print(M1, S1, T1)
	def _get_optimizer(self):
		lr = symbolic_functions.get_scalar_var('learning_rate', 1e-4, summary=True)
		return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)




###############################################################################	
def sample(imageDir, maskDir, labelDir, model_path, resultDir, size):
	# TODO
	print(sys.argv[0])
	pred_config = PredictConfig(
		session_init=SaverRestore(model_path), #session_init=SaverRestore(args.load)
		model=Model(),
		input_names=['inputA', 'mask', 'inputB'],
		output_names=['vis_A_recon'])


	ds_valid = ImageDataFlow(imageDir, maskDir, labelDir, size, is_training=False)
	# ds_valid = PrefetchDataZMQ(ds_valid, nr_proc=8)
	
	filenames = glob.glob(imageDir + '/*.*')
	from natsort import natsorted
	filenames = natsorted(filenames)
	print(filenames)
	print(resultDir)

	import shutil
	shutil.rmtree(resultDir, ignore_errors=True)
	shutil.rmtree(resultDir+'/mag/', ignore_errors=True)
	shutil.rmtree(resultDir+'/ang/', ignore_errors=True)

	# Make directory to hold RefineGAN result
	os.makedirs(resultDir)
	os.makedirs(resultDir+'/mag/')
	os.makedirs(resultDir+'/ang/')

	# Zero-filling for baseline
	os.makedirs(resultDir+'/M/')
	os.makedirs(resultDir+'/M/mag/')
	os.makedirs(resultDir+'/M/ang/')
	


	## Extract stack of images with SimpleDatasetPredictor
	pred = SimpleDatasetPredictor(pred_config, ds_valid)
	
	for idx, o in enumerate(pred.get_result()):
		print(pred)
		print(len(o))
		print(o[0].shape)

		outA = o[0][:, :, :, :] 

	
		colors0 = np.array(outA) #.astype(np.uint8)
		head, tail = os.path.split(filenames[idx])
		tail = tail.replace('png', 'tif')
		print(tail)
		print(colors0.shape)
		print(colors0.dtype)
		import skimage.io
	
		skimage.io.imsave(resultDir+ "/full_"+tail, np.squeeze(colors0[...,256*1:256*2])) # Zerofill
		skimage.io.imsave(resultDir+"/zfill_"+tail, np.squeeze(colors0[...,256*2:256*3])) # Zerofill
		skimage.io.imsave(resultDir+tail, np.squeeze(colors0[...,256*4:256*5])) # Zerofill

		skimage.io.imsave(resultDir+"mag/mag_"+tail, np.abs(np_complex(np.squeeze(colors0[...,256*4:256*5]))))
		skimage.io.imsave(resultDir+"ang/ang_"+tail, np.angle(np_complex(np.squeeze(colors0[...,256*4:256*5]))))


		skimage.io.imsave(resultDir+"/M/mag/mag_"+tail, np.abs(np_complex(np.squeeze(colors0[...,256*2:256*3]))))
		skimage.io.imsave(resultDir+"/M/ang/ang_"+tail, np.angle(np_complex(np.squeeze(colors0[...,256*2:256*3]))))

		# skimage.io.imsave(resultDir+"/S/mag/mag_"+tail, np.abs(np_complex(np.squeeze(colors0[...,256*3:256*4]))))
###############################################################################	
class VisualizeRunner(Callback):
	def _setup_graph(self):
		self.pred = self.trainer.get_predictor(
			input_names=['inputA', 'mask', 'inputB'],
			output_names=['vis_A_recon'])

	def _before_train(self):
		global args
		self.ds_train, self.ds_valid = get_data(args.imageDir, args.maskDir, args.labelDir, size=1)
		self.ds_train.reset_state()
		self.ds_valid.reset_state() 

	def _trigger(self):
		for lst in self.ds_train.get_data():
			vis_train =  np.array(self.pred(lst)[0])
			
			vis_train_real = np.transpose(vis_train[:,0:1,...], [0, 2, 3, 1])
			vis_train_imag = np.transpose(vis_train[:,1:2,...], [0, 2, 3, 1])
			
			self.trainer.monitors.put_image('vis_train_real', vis_train_real)
			self.trainer.monitors.put_image('vis_train_imag', vis_train_imag)
		for lst in self.ds_valid.get_data():
			vis_valid = np.array(self.pred(lst)[0])
			vis_valid_real = np.transpose(vis_valid[:,0:1,...], [0, 2, 3, 1])
			vis_valid_imag = np.transpose(vis_valid[:,1:2,...], [0, 2, 3, 1])
			
			self.trainer.monitors.put_image('vis_valid_real', vis_valid_real)
			self.trainer.monitors.put_image('vis_valid_imag', vis_valid_imag)

###############################################################################		
# if __name__ == '__main__':
def main(input_args):
	global args
	args = input_args
	np.random.seed(2018)
	tf.set_random_seed(2018)
	len_images = len(glob.glob(args.imageDir + '/*.*'))
	len_labels = len(glob.glob(args.labelDir + '/*.*'))
	assert len_images == len_labels
	#https://docs.python.org/3/library/argparse.html
	#
	if args.gpu:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
		# os.environ['TENSORPACK_TRAIN_API'] = 'v2'
	if args.sample:
		sample(args.imageDir, args.maskDir, args.labelDir, args.load, args.sample, len_images)
	else:
		logger.auto_set_dir()
		ds_train, ds_valid = get_data(args.imageDir, args.maskDir, args.labelDir, len_images)

		ds_train = PrefetchDataZMQ(ds_train, nr_proc=4)
		ds_valid = PrefetchDataZMQ(ds_valid, nr_proc=4)

		ds_train.reset_state()
		ds_valid.reset_state() 

		nr_tower = max(get_nr_gpu(), 1)
		ds_train = QueueInput(ds_train)
		model = Model()
		if nr_tower == 1:
			trainer = SeparateGANTrainer(ds_train, model, g_period=1, d_period=1)
		else:
			trainer = MultiGPUGANTrainer(nr_tower, ds_train, model)
		trainer.train_with_defaults(
			callbacks=[
				PeriodicTrigger(ModelSaver(), every_k_epochs=20),
				PeriodicTrigger(MaxSaver('validation_PSNR_recon'), every_k_epochs=20),
				VisualizeRunner(),
				InferenceRunner(ds_valid, [
										   ScalarStats('PSNR_recon'),
										   ScalarStats('losses/Img/Recon/smoothness_AA'),
					]),
				ClipCallback(),
				ScheduledHyperParamSetter('learning_rate', 
					[(0, 1e-4), (100, 3e-4), (200, 2e-5), (300, 1e-5), (400, 2e-6), (500, 1e-6)], interp='linear')
				
				],
			session_init=SaverRestore(args.load) if args.load else None, 
			steps_per_epoch=ds_train.size(),
			max_epoch=100
		)


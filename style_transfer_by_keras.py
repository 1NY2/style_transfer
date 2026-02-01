'''
（1）图像风格迁移：给定一张普通图片和一种艺术风格图片，生成一张呈现艺术风格
    和普通图片内容的迁移图片。
（2）此次实现中使用了VGG19的卷积神经网络模型，优化过程使用了scipy.optimizer
    基于L-BFGS算法的fmin_l_bfgs_b方法
（3）每次反向优化20次写出一张图片，在代码运行过程中发现超过10次loss减少量减少
    趋于平缓，所以只写出15张图片
（4）从images文件夹中选择普通图片和风格图片，并且不同风格和内容图片中间过程生成
    的图片都在results文件夹中
 (5)由于保存权值的.h5文件较大，这里给出下载地址
    https://github.com/fchollet/deep-learning-models/
    releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5
'''
import os
os.environ['KERAS_BACKEND']='tensorflow'
import time
import numpy as np
from imageio import imwrite
from scipy.optimize import fmin_l_bfgs_b
import tensorflow as tf
from keras import backend as K
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import VGG19
#from keras.applications.imagenet_utils import _obtain_input_shape
#使用tensorflow环境编程
#定义目标图像长宽
img_rows=400
img_columns=300
#读入图片文件，以数组形式展开成三阶张量，后用numpy扩展为四阶张量
#最后使用对图片进行预处理：（1）去均值,（2）三基色RGB->BGR(3)调换维度 
def read_img(filename):
	img=load_img(filename,target_size=(img_columns,img_rows))
	img=img_to_array(img)
	img=np.expand_dims(img,axis=0)
	img=preprocess_input(img)
	return img
#写入/存储图片，将输出数组转换为三维张量，量化高度层BGR,并将BGR->RGB
#经灰度大小截断在（0,255）
def write_img(x,ordering):
	x=x.reshape((img_columns,img_rows,3))
	x[:,:,0]+=103.939
	x[:,:,1]+=116.779
	x[:,:,2]+=123.68
	x=x[:,:,::-1]
	x=np.clip(x,0,255).astype('uint8')
	result_file=('results/%s'%str(ordering).zfill(2))+'.png'
	if not os.path.exists('results'):
		os.mkdir('results')
	imwrite(result_file,x)
	print(result_file)
#建立vgg19模型
def vgg19_model(input_tensor):
	img_input=Input(tensor=input_tensor,shape=(img_columns,img_rows,3))
	#Blocks 1
	x=Conv2D(64,(3,3),activation='relu',padding='same',name='block1_conv1')(img_input)
	x=Conv2D(64,(3,3),activation='relu',padding='same',name='block1_conv2')(x)
	x=MaxPooling2D((2,2),strides=(2,2),name='block1_pooling')(x)
	#Block 2
	x=Conv2D(128,(3,3),activation='relu',padding='same',name='block2_conv1')(x)
	x=Conv2D(128,(3,3),activation='relu',padding='same',name='block2_conv2')(x)
	x=MaxPooling2D((2,2),strides=(2,2),name='block2_pooling')(x)
	#Block3
	x=Conv2D(256,(3,3),activation='relu',padding='same',name='block3_conv1')(x)
	x=Conv2D(256,(3,3),activation='relu',padding='same',name='block3_conv2')(x)
	x=Conv2D(256,(3,3),activation='relu',padding='same',name='block3_conv3')(x)
	x=Conv2D(256,(3,3),activation='relu',padding='same',name='block3_conv4')(x)
	x=MaxPooling2D((2,2),strides=(2,2),name='block3_pooling')(x)
	#Block 4
	x=Conv2D(512,(3,3),activation='relu',padding='same',name='block4_conv1')(x)
	x=Conv2D(512,(3,3),activation='relu',padding='same',name='block4_conv2')(x)
	x=Conv2D(512,(3,3),activation='relu',padding='same',name='block4_conv3')(x)
	x=Conv2D(512,(3,3),activation='relu',padding='same',name='block4_conv4')(x)
	x=MaxPooling2D((2,2),strides=(2,2),name='block4_pooling')(x)
	#Block 5
	x=Conv2D(512,(3,3),activation='relu',padding='same',name='block5_conv1')(x)
	x=Conv2D(512,(3,3),activation='relu',padding='same',name='block5_conv2')(x)
	x=Conv2D(512,(3,3),activation='relu',padding='same',name='block5_conv3')(x)
	x=Conv2D(512,(3,3),activation='relu',padding='same',name='block5_conv4')(x)
	x=MaxPooling2D((2,2),strides=(2,2),name='block5_pooling')(x)
	model = Model(inputs=img_input, outputs=x, name='vgg19')
	weights_path = 'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
	model.load_weights(weights_path)
	return model
#生成输入的张量,将内容，风格和迁移图像（中间量）一起输入到vgg模型中，返回三合一张量，和中间图张量
def create_tensor(content_path,style_path):
	content_array = read_img(content_path)
	style_array = read_img(style_path)
	
	# 创建输入张量，包含内容、风格和待优化的迁移图像
	input_tensor = Input(shape=(None, img_columns, img_rows, 3), name='input_tensor')
	
	# 内容和风格张量在优化过程中是固定的，迁移图像需要优化
	content_tensor = tf.constant(content_array)
	style_tensor = tf.constant(style_array)
	
	# 初始化迁移图像为随机噪声
	transfer_array = np.random.uniform(0, 255, (1, img_columns, img_rows, 3)) - 128.0
	transfer_tensor = tf.Variable(transfer_array, trainable=True, name='transfer_tensor')
	
	return input_tensor, transfer_tensor, content_tensor, style_tensor
#设置Gram_matrix矩阵的计算图，输入为某一层的representation
def gram_matrix(x):
	features=tf.reshape(x, [-1, tf.shape(x)[-1]])  # 等同于 batch_flatten
	gram=tf.linalg.matmul(features, features, transpose_a=True)  # 等同于 dot(features, transpose(features))
	return gram
#风格loss
def style_loss(style_img_feature,transfer_img_feature):
	style=style_img_feature
	transfer=transfer_img_feature
	A=gram_matrix(style)
	G=gram_matrix(transfer)
	channels=3
	size=img_rows*img_columns
	loss=tf.reduce_sum(tf.square(A-G))/(4.*(channels**2)*(size**2))
	return loss
#内容loss
def content_loss(content_img_feature,transfer_img_feature):
	content=content_img_feature
	transfer=transfer_img_feature
	loss=tf.reduce_sum(tf.square(transfer-content))
	return loss		 
#变量loss,一段迷一样的表达式×-×，施加全局差正则表达式，全局差正则用于使生成的图片更加平滑自然
def total_variation_loss(x):
	a=tf.square(x[:,:img_columns-1,:img_rows-1,:]-x[:,1:,:img_rows-1,:])
	b=tf.square(x[:,:img_columns-1,:img_rows-1,:]-x[:,:img_columns-1,1:,:])
	loss=tf.reduce_sum(tf.pow(a+b,1.25))
	return loss
#total loss
def total_loss(model,loss_weights,transfer_tensor,content_tensor,style_tensor):
	# 构建模型来获取特征
	content_features = model(content_tensor)
	style_features = model(style_tensor)
	transfer_features = model(transfer_tensor)
	
	# 计算内容损失 - 使用block4_conv2层
	content_layer = 'block4_conv2'
	content_img_features = content_features[content_layer]
	transfer_img_features = transfer_features[content_layer]
	content_loss_val = content_loss(content_img_features, transfer_img_features)
	
	total_loss_val = loss_weights['content'] * content_loss_val
	
	# 计算风格损失 - 多层
	feature_layers=['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']
	for layer_name in feature_layers:
		style_features_layer = style_features[layer_name]
		transfer_features_layer = transfer_features[layer_name]
		style_loss_val = style_loss(style_features_layer, transfer_features_layer)
		total_loss_val += (loss_weights['style']/len(feature_layers)) * style_loss_val
	
	# 计算总变差损失
	total_loss_val += loss_weights['total'] * total_variation_loss(transfer_tensor)
	
	return total_loss_val
#通过K.gradient获取反向梯度，同时得到梯度和损失，
def create_outputs(transfer_tensor, content_tensor, style_tensor, loss_weights):
	# 重新构建模型，只使用需要的层
	model_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1', 'block4_conv2']

	# 使用本地预训练的VGG19模型
	weights_path = 'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
	base_model = VGG19(weights=weights_path, include_top=False)

	# 创建特征提取模型
	outputs_dict = dict([(layer.name, layer.output) for layer in base_model.layers[1:]])
	feature_model = Model(base_model.input, outputs_dict)

	# 设置特征模型为不可训练
	feature_model.trainable = False

	# 定义损失函数
	def loss_fn():
		return total_loss(feature_model, loss_weights, transfer_tensor, content_tensor, style_tensor)

	# 计算梯度
	with tf.GradientTape() as tape:
		loss = loss_fn()
	grads = tape.gradient(loss, transfer_tensor)

	return loss, grads
#计算输入图像的关于损失函数的倒数和对应损失值
def eval_loss_and_grads(x, transfer_tensor, content_tensor, style_tensor, loss_weights):
	x = x.reshape((1, img_columns, img_rows, 3))
	x = x.astype(np.float32)  # 确保数据类型为float32
	transfer_tensor.assign(x)
	loss, grads = create_outputs(transfer_tensor, content_tensor, style_tensor, loss_weights)
	loss_value = loss.numpy()
	grads_value = grads.numpy().flatten().astype('float64')
	return loss_value, grads_value
#获取评价程序
class Evaluator(object):
	def __init__(self, transfer_tensor, content_tensor, style_tensor, loss_weights):
		self.loss_value = None
		self.grads_value = None
		self.transfer_tensor = transfer_tensor
		self.content_tensor = content_tensor
		self.style_tensor = style_tensor
		self.loss_weights = loss_weights

	def loss(self, x):
		loss_value, grads_value = eval_loss_and_grads(x, self.transfer_tensor, self.content_tensor, self.style_tensor, self.loss_weights)
		self.loss_value = loss_value
		self.grads_value = grads_value
		return self.loss_value

	def grads(self, x):
		grads_value = np.copy(self.grads_value)
		self.loss_value = None
		self.grads_value = None
		return grads_value
#main函数
if __name__=='__main__':
	print('')
	print('Welcom!')
	path={'content':'images/Taipei101.jpg','style':'images/Cubist.jpg'}
	
	# 读取内容和风格图像
	content_array = read_img(path['content'])
	style_array = read_img(path['style'])
	
	# 创建变量
	transfer_array = np.random.uniform(0, 255, (1, img_columns, img_rows, 3)) - 128.0
	transfer_array = transfer_array.astype(np.float32)  # 确保数据类型为float32
	transfer_tensor = tf.Variable(transfer_array, trainable=True, name='transfer_tensor', dtype=tf.float32)
	content_tensor = tf.constant(content_array.astype(np.float32))
	style_tensor = tf.constant(style_array.astype(np.float32))
	
	loss_weights={'style':1.0,'content':0.025,'total':1.0}
	
	#生成处理器
	evaluator=Evaluator(transfer_tensor, content_tensor, style_tensor, loss_weights)
	
	#生成噪声
	x=np.random.uniform(0,225,(1,img_columns,img_rows,3))-128
	x = x.astype(np.float32).flatten()  # 确保数据类型为float32并展平
	
	#迭代训练15次
	for ordering in range(15):
		print('Start:',ordering)
		start_time=time.time()
		x,min_val,info=fmin_l_bfgs_b(evaluator.loss,x.flatten(),fprime=evaluator.grads,maxfun=20)
		print('Current_Loss:',min_val)
		img=np.copy(x)
		write_img(img,ordering)
		end_time=time.time()
		print('Used %ds'%(end_time-start_time))
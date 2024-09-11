#!/usr/bin/env python3

import xarray as xr
import os
from glob import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import backend as K
import pickle
import keras as k
from keras.layers import Conv2D, Input, AvgPool2D, MaxPool2D, Concatenate, Add, Dropout, BatchNormalization, Conv2DTranspose, Activation, DepthwiseConv2D, concatenate
import keras_cv
from tensorflow.keras.layers import MultiHeadAttention

kernel_initializer = 'glorot_uniform'

def conv_batchnorm_relu_block_SE_and_residual_connections(input_tensor, nb_filter, dropout_rate, kernel_norm, kernel_size=3,kernel_initializer=kernel_initializer):
    depth_multiplier=2
    
    shortcut = Conv2D(nb_filter, kernel_size=(1, 1), padding='same', strides=1, kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(input_tensor)
    shortcut = BatchNormalization()(shortcut)
    shortcut = Activation('relu')(shortcut)
    
    # x = Conv2D(nb_filter, (kernel_size, kernel_size), padding='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(input_tensor)
    x = DepthwiseConv2D(3, padding ='same', depth_multiplier=depth_multiplier, kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = tf.keras.layers.SpatialDropout2D(rate=dropout_rate, data_format='channels_last')(x,training=True)
    #x = Dropout(dropout_rate)(x, training =True) #add dropout 

    x = Conv2D(nb_filter, (kernel_size, kernel_size), padding='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(x)
    x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = Dropout(dropout_rate)(x, training =True) #add dropout 
    
    residual_SE_block = Add()([shortcut, x])
    # residual_SE_block = keras_cv.layers.SqueezeAndExcite2D(residual_SE_block.shape[-1])(residual_SE_block)
    
    return residual_SE_block

def inception_block(prevlayer, a, b,dropout_rate, kernel_norm, depth_multiplier = False):
    #shortcut = Conv2D(a,kernel_size=(1, 1), padding='same', strides=1, kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(prevlayer)
    #shortcut = BatchNormalization()(shortcut)
    #shortcut = Activation('relu')(shortcut)

    shortcut= prevlayer
    
    if depth_multiplier == True:
        depth_multiplier=2
    else:
        depth_multiplier=2
    
    conva = DepthwiseConv2D(3, padding ='same',depth_multiplier=depth_multiplier,kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(prevlayer)
    conva = BatchNormalization()(conva)
    conva = tf.keras.activations.relu(conva)
    conva = tf.keras.layers.SpatialDropout2D(rate=dropout_rate, data_format='channels_last')(conva,training=True)
    #conva = Dropout(dropout_rate)(conva, training =True) #add dropout
    
    conva = Conv2D(a,(1,1), padding ='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(conva)
    conva = BatchNormalization()(conva)
    #conva = tf.keras.activations.relu(conva)
    #conva = Dropout(dropout_rate)(conva, training =True) #add dropout

    convb = DepthwiseConv2D(5, padding ='same',depth_multiplier=depth_multiplier,kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(prevlayer)
    convb = BatchNormalization()(convb)
    convb = tf.keras.activations.relu(convb)
    convb = tf.keras.layers.SpatialDropout2D(rate=dropout_rate, data_format='channels_last')(convb,training=True)
    #convb = Dropout(dropout_rate)(convb, training =True) #add dropout
    
    convb = Conv2D(a,(1,1), padding ='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(convb)
    convb = BatchNormalization()(convb)
    #convb = tf.keras.activations.relu(convb)
    #convb = Dropout(dropout_rate)(convb, training =True) #add dropout

    convc = DepthwiseConv2D(7, padding='same',depth_multiplier=depth_multiplier,kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(prevlayer)
    convc = BatchNormalization()(convc)
    convc = tf.keras.activations.relu(convc)
    convc = tf.keras.layers.SpatialDropout2D(rate=dropout_rate, data_format='channels_last')(convc,training=True)
    #convc = Dropout(dropout_rate)(convc, training =True) #add dropout
    
    convc = Conv2D(a,(1,1), padding ='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(convc)
    convc = BatchNormalization()(convc)
    #convc = tf.keras.activations.relu(convc)
    #convc = Dropout(dropout_rate)(convc, training =True) #add dropout

    # if True == pooling:
    #     convd = MaxPooling2D(pool_size=(2, 2))(convd)
    
    #Max pool
    convd = MaxPool2D((5,5), strides=(1, 1), padding='same')(prevlayer)
    convd = Conv2D(a,(1, 1), padding='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(convd)
    convd = BatchNormalization()(convd)
    convd = tf.keras.activations.relu(convd)
    convd = tf.keras.layers.SpatialDropout2D(rate=dropout_rate, data_format='channels_last')(convd,training=True)
    #onvd = Dropout(dropout_rate)(convd, training =True) #add dropout
    
    convd = Conv2D(a,(1,1), padding ='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(convd)
    convd = BatchNormalization()(convd)
    #convd = tf.keras.activations.relu(convd)
    #convd = Dropout(dropout_rate)(convd, training =True) #add dropout

    up = concatenate([conva, convb, convc, convd])
    
    #residual_block = Concatenate()([shortcut, up])
    
    # Adjust the number of channels in the shortcut connection
    if shortcut.shape[-1] != up.shape[-1]:
        shortcut = Conv2D(up.shape[-1], kernel_size=1, strides=1, padding='same')(shortcut)

    residual_block = Add()([shortcut, up])

    # residual_SE_block = keras_cv.layers.SqueezeAndExcite2D(residual_block.shape[-1])(residual_block)
    
    return residual_block


def transformer_encoder(new_inputs, num_heads, ff_dim=0, dropout=0):
    x = MultiHeadAttention(key_dim=num_heads, num_heads=num_heads, dropout=dropout)(new_inputs, new_inputs,new_inputs)
    res = k.layers.Add()([new_inputs, x])
    x = Conv2D(list(res.shape)[-1], 1)(res)  # 1x1 convolution for channel-wise feedforward
    x - tf.keras.activations.relu(x)
    x = Add()([res, x])
    x = BatchNormalization()(x)
    
    return (x)

def model_build_func(inputs,kernel_norm, output_channels,var_name, number_of_UNET_backbone_max_pool,using_deep_supervision=True,kernel_initializer=kernel_initializer ):

    dropout_rate_initial = 0.1
    dropout_rate_later = 0.25
    nb_filter = [16,32,64,128]
    num_heads = 2
    num_transformer_loops = 2


    # Set image data format to channels first
    global bn_axis
    
    k.backend.set_image_data_format("channels_last")
    bn_axis = -1
    
    conv1_1 = inception_block(inputs, nb_filter[0], nb_filter[0],dropout_rate=dropout_rate_initial,depth_multiplier=True, kernel_norm =kernel_norm)
    pool1 = MaxPool2D((2, 2), strides=(2, 2))(conv1_1)
    
    conv2_1 = inception_block(pool1, nb_filter[1], nb_filter[1], dropout_rate = dropout_rate_later,depth_multiplier=False, kernel_norm =kernel_norm)
    # conv2_1 = conv_batchnorm_relu_block_SE_and_residual_connections(conv2_1, nb_filter=nb_filter[1], dropout_rate = dropout_rate_later, kernel_norm=kernel_norm)  
    pool2 = MaxPool2D((2, 2), strides=(2, 2))(conv2_1)
    
    up1_2 = tf.nn.depth_to_space(conv2_1, block_size=2)
    conv1_2 = concatenate([up1_2, conv1_1], axis=bn_axis)
    conv1_2 = inception_block(conv1_2, nb_filter[0], nb_filter[0], dropout_rate = dropout_rate_later,depth_multiplier=False, kernel_norm =kernel_norm)  

    #conv3_1=inception_block(pool2, nb_filter[2], nb_filter[2],dropout_rate=dropout_rate_later,depth_multiplier=True, kernel_norm =kernel_norm)
    conv3_1 = conv_batchnorm_relu_block_SE_and_residual_connections(pool2, nb_filter=nb_filter[2], dropout_rate = dropout_rate_later, kernel_norm=kernel_norm)
    pool3 = MaxPool2D((2, 2), strides=(2, 2),)(conv3_1)
    
    up2_2 =tf.nn.depth_to_space(conv3_1, block_size=2)
    conv2_2 = concatenate([up2_2, conv2_1], axis=bn_axis)
    #conv2_2 = inception_block(conv2_2, nb_filter[1], nb_filter[1], dropout_rate = dropout_rate_later,depth_multiplier=False, kernel_norm =kernel_norm)
    conv2_2 = inception_block(conv2_2, nb_filter[1], nb_filter[1], dropout_rate = dropout_rate_later,depth_multiplier=False, kernel_norm =kernel_norm)
    # conv2_2 = conv_batchnorm_relu_block_SE_and_residual_connections(conv2_2, nb_filter=nb_filter[2], dropout_rate = dropout_rate_later, kernel_norm=kernel_norm)  
    
    up1_3 = tf.nn.depth_to_space(conv2_2, block_size=2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], axis=bn_axis)
    for i in range(num_transformer_loops):
        conv1_3 = transformer_encoder(conv1_3, num_heads=num_heads, ff_dim=0, dropout=dropout_rate_later)
    # conv1_3 = conv_batchnorm_relu_block_SE_and_residual_connections(conv1_3, nb_filter=nb_filter[0], dropout_rate = dropout_rate_later, kernel_norm=kernel_norm)  
    
    #conv4_1 = inception_block(pool3, nb_filter[3], nb_filter[3], dropout_rate = dropout_rate_later,depth_multiplier=False, kernel_norm =kernel_norm)
    conv4_1 = conv_batchnorm_relu_block_SE_and_residual_connections(pool3, nb_filter=nb_filter[3], dropout_rate = dropout_rate_later, kernel_norm=kernel_norm)
    # conv4_1 = conv_batchnorm_relu_block_SE_and_residual_connections(conv4_1, nb_filter=nb_filter[3], dropout_rate = dropout_rate_later, kernel_norm=kernel_norm)
    
    up3_2 = tf.nn.depth_to_space(conv4_1, block_size=2)
    conv3_2 = concatenate([up3_2, conv3_1], axis=bn_axis)
    conv3_2 = inception_block(conv3_2, nb_filter[2], nb_filter[2], dropout_rate = dropout_rate_later,depth_multiplier=False, kernel_norm =kernel_norm)  
    # conv_batchnorm_relu_block_SE_and_residual_connections(conv3_2, nb_filter=nb_filter[2], dropout_rate = dropout_rate_later, kernel_norm=kernel_norm)
    # conv3_2 = conv_batchnorm_relu_block_SE_and_residual_connections(conv3_2, nb_filter=nb_filter[2], dropout_rate = dropout_rate_later, kernel_norm=kernel_norm)
    #conv3_2 = inception_block(conv3_2, nb_filter[2], nb_filter[2], dropout_rate = dropout_rate_later,depth_multiplier=True, kernel_norm=kernel_norm)
    
    up2_3 = tf.nn.depth_to_space(conv3_2, block_size=2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2],axis=bn_axis)
    conv2_3 =  inception_block(conv2_3, nb_filter[2], nb_filter[2], dropout_rate = dropout_rate_later,depth_multiplier=False, kernel_norm =kernel_norm) 
    # conv_batchnorm_relu_block_SE_and_residual_connections(conv2_3, nb_filter=nb_filter[2], dropout_rate = dropout_rate_later, kernel_norm=kernel_norm) 
    # conv2_3 = conv_batchnorm_relu_block_SE_and_residual_connections(conv2_3, nb_filter=nb_filter[2], dropout_rate = dropout_rate_later, kernel_norm=kernel_norm) 
    
    up1_4 = tf.nn.depth_to_space(conv2_3, block_size=2)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], axis=bn_axis)
    for i in range(num_transformer_loops):
        conv1_4 = transformer_encoder(conv1_4, num_heads=num_heads, ff_dim=0, dropout=dropout_rate_later)
    # conv1_4 = conv_batchnorm_relu_block_SE_and_residual_connections(conv1_4, nb_filter=nb_filter[0], dropout_rate = dropout_rate_later, kernel_norm=kernel_norm) 
    
    #For backbone == 5
    #pool4 = MaxPool2D((2, 2), strides=(2, 2),)(conv4_1)
    #conv5_1 = inception_block(pool4, nb_filter[3], nb_filter[3], dropout_rate = dropout_rate_later,depth_multiplier=False, kernel_norm =kernel_norm)
    #conv5_1 = conv_batchnorm_relu_block_SE_and_residual_connections(pool4, nb_filter=nb_filter[2], dropout_rate = dropout_rate_later, kernel_norm =kernel_norm)

    #up4_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), padding='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(conv5_1)
    #conv4_2 = concatenate([up4_2, conv4_1], axis=bn_axis)
    #conv4_2 = inception_block(conv4_2, nb_filter[2], nb_filter[2], dropout_rate = dropout_rate_later,depth_multiplier=False, kernel_norm =kernel_norm)
    #conv4_2 = conv_batchnorm_relu_block_SE_and_residual_connections(conv4_2, nb_filter=nb_filter[2], dropout_rate = dropout_rate_later, kernel_norm =kernel_norm)

    #up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), padding='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(conv4_2)
    #conv3_3 = concatenate([up3_3, conv3_1, conv3_2], axis=bn_axis)
    #conv3_3 = inception_block(conv3_3, nb_filter[2], nb_filter[2], dropout_rate = dropout_rate_later,depth_multiplier=False, kernel_norm =kernel_norm)
    #conv3_3 = conv_batchnorm_relu_block_SE_and_residual_connections(conv3_3, nb_filter=nb_filter[2], dropout_rate = dropout_rate_later, kernel_norm=kernel_norm)

    #up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2),  padding='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(conv3_3)
    #conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3],  axis=bn_axis)
    #conv2_4 = inception_block(conv2_4, nb_filter[1], nb_filter[1], dropout_rate = dropout_rate_later,depth_multiplier=False, kernel_norm =kernel_norm)
    #conv2_4 =  conv_batchnorm_relu_block_SE_and_residual_connections(conv2_4, nb_filter=nb_filter[1], dropout_rate = dropout_rate_later, kernel_norm=kernel_norm)

    #up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2),padding='same',kernel_initializer=kernel_initializer, kernel_constraint = kernel_norm)(conv2_4)
    #conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], axis=bn_axis)
    #conv1_5 = inception_block(conv1_5, nb_filter[0], nb_filter[0], dropout_rate = dropout_rate_later,depth_multiplier=False, kernel_norm=kernel_norm)
    #conv1_5 =  conv_batchnorm_relu_block_SE_and_residual_connections(conv1_5, nb_filter=nb_filter[0], dropout_rate = dropout_rate_later, kernel_norm=kernel_norm)

    nestnet_output_1 = Conv2D(output_channels, (1, 1), activation='relu', name=f'RZSM_output_1',padding='same',kernel_initializer=kernel_initializer,dtype=tf.float32)(conv1_2)
    nestnet_output_2 = Conv2D(output_channels, (1, 1), activation='relu', name=f'RZSM_output_2', padding='same',kernel_initializer=kernel_initializer ,dtype=tf.float32)(conv1_3)
    nestnet_output_3 = Conv2D(output_channels, (1, 1), activation='relu', name=f'RZSM_output_3', padding='same',kernel_initializer=kernel_initializer,dtype=tf.float32)(conv1_4)
    #nestnet_output_4 = Conv2D(output_channels, (1, 1), activation='relu', name=f'RZSM_output_4', padding='same',kernel_initializer=kernel_initializer,dtype=tf.float32)(conv1_5)

    if using_deep_supervision:
        if number_of_UNET_backbone_max_pool == 4:
            return(nestnet_output_1,nestnet_output_2,nestnet_output_3)
        
        elif number_of_UNET_backbone_max_pool == 5:
            return(nestnet_output_1,nestnet_output_2,nestnet_output_3,nestnet_output_4)
        
        
                # model = Model(inputs=inputs, outputs=[nestnet_output_1,
                #                                     nestnet_output_2,
                #                                     nestnet_output_3,
                #                                     nestnet_output_4])
        # else:
        #     model = Model(inputs=inputs, outputs=nestnet_output_4)

        # return model

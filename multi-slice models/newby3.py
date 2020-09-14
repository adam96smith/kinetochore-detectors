import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
import scipy.io as sio
import os
import glob
from PIL import Image
import random as rd
import cv2
import csv
import math
import argparse

from scipy import ndimage as ndi
from scipy.ndimage.measurements import center_of_mass, label

from simulation import movie_maker

from datetime import datetime
import time


parser = argparse.ArgumentParser()
parser.add_argument("--mode", type = str, required = True)
parser.add_argument("--test_dir", type = str)
parser.add_argument("--checkpoint",type = str,required = True)
parser.add_argument("--out_dir", type = str)
parser.add_argument("--pre_epoch", type = int)
parser.add_argument("--epochs", type = int)
parser.add_argument("--evaluation_volumes", type = int)
parser.add_argument("--save_arr",type = str, default = "Yes")
a = parser.parse_args()

tensorboard_dir = "./tf_log/"
buffer = 10000
batch_size = 1
image_size = 256
stack_size = 90 #resolution in the z direction


def nonlinearity(inputs):
    return tf.math.log(tf.math.exp(inputs)+1)


def create_model():
    
    inputs = tf.keras.Input(shape=(image_size, image_size, 3, 1))
    prev_inputs = tf.keras.Input(shape=(image_size//2, image_size//2, 3, 6))
    
    # Layer 1:
    x11 = layers.Conv3D(3,(9,9,3),strides=(2,2,1), padding='SAME', kernel_initializer=tf.random_normal_initializer(0,0.02))(inputs)
    x11 = nonlinearity(x11)
    x12 = layers.Conv3D(3,(6,6,3),strides=(2,2,1), padding='SAME', kernel_initializer=tf.random_normal_initializer(0,0.02))(inputs)
    x12 = nonlinearity(x12)
    x13 = layers.Conv3D(3,(3,3,3),strides=1, padding='SAME', kernel_initializer=tf.random_normal_initializer(0,0.02))(inputs)
    x13 = layers.MaxPool3D(pool_size=(2,2,1))(x13)
    x13 = nonlinearity(x13)
    x1 = layers.Concatenate(axis=4)([x11,x12,x13])
    # x1 = (batch, x//2, y//2, 3, 9) # layer 1 output
    
    # Layer 2:
    x21 = layers.Conv3D(6,(7,7,3),strides=1, padding='SAME', kernel_initializer=tf.random_normal_initializer(0,0.02))(x1)
    x21 = nonlinearity(x21)
    x22 = layers.Conv3D(6,(7,7,3),strides=1, dilation_rate = (2,2,2), padding='SAME', kernel_initializer=tf.random_normal_initializer(0,0.02))(x1)
    rnn_input = nonlinearity(x22)
    x23 = layers.Conv3D(6,(3,3,3),strides=1, padding='SAME', kernel_initializer=tf.random_normal_initializer(0,0.02))(x1)
    x23 = nonlinearity(x23)
    x2 = layers.Concatenate(axis=4)([x21,x23,prev_inputs])
    # x2 = (batch, x//2, y//2, z//2, 18) # layer 2 output
    # rnn_input = (batch, x//2, y//2, z//2, 6)
    
    # RNN layer
    rnn_out = layers.Conv3D(6,(7,7,3),strides=1, dilation_rate = (3,3,3), padding='SAME', kernel_initializer=tf.random_normal_initializer(0,0.02))(rnn_input)
    # rnn_out = (batch, x//2, y//2, z//2, 6)
    
    #Layer 3:
    x3 = layers.Conv3D(1, (5,5,3), strides=1, padding='SAME', kernel_initializer=tf.random_normal_initializer(0,0.02))(x2)
#     main_out = layers.UpSampling3D(size=(2,2,2))(x3) #only uses nearest neighbour interpolation rather than the bilinear interpolation used in 2d
    main_out = layers.Conv3DTranspose(1,(3,3,3),strides=(2,2,1), padding='SAME', kernel_initializer=tf.random_normal_initializer(0,0.02))(x3)
    main_out = layers.MaxPool3D(pool_size=(1,1,3))(main_out)
    main_out = tf.keras.activations.sigmoid(main_out)
    # main_out = (batch, x, y, 1)
    
    model = tf.keras.Model(inputs=[inputs,prev_inputs], outputs=[main_out, rnn_out])
    return model
  

model = create_model()
model.summary()
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001, beta_1 = 0.5)# decay = 0.95)


@tf.function
def loss_function(outputs,targets):
    w_1 = 0.5
    w_2 = 0.4
    loss = tf.reduce_sum((w_1*targets+w_2*tf.keras.backend.mean(targets))*(targets-outputs)**2)
    return loss


@tf.function
def train_step(inputs, prev_step, targets):
    
    with tf.GradientTape() as tape:
        main_out, rnn_out = model([inputs,prev_step],training=True)
        outputs = [main_out, rnn_out]
        loss = loss_function(outputs[0],targets)
        
    gradients = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    
    return loss, outputs


#Train
def train():
    global tensorboard_dir
    now = datetime.utcnow().strftime("%Y-%m-%d_%Hh-%Mm-%Ss")
    method = 'newby3'
    tensorboard_dir = "{}/run_{}_{}/".format(tensorboard_dir, method, now)
    
    # Number of movies per set, movie length
    n = 10
    mov_len = 20
    
    writer = tf.summary.create_file_writer(tensorboard_dir)
    for epoch in range(1,epochs+1):
        
        #Generate New Training Set of 10 movies
        if epoch % 50 == 1:
            data = []
            for _ in range(0,n):
                snr = 1.5+4.5*np.random.rand()
                mov, _ = movie_maker(snr,mov_len)
                data.append(mov)
            data = np.array(data).astype(np.uint8)
            
        model_loss = []
        for i in range(0,n):
          
            rnn_out = []
            
            z_sample = rd.sample(np.linspace(21,60,40).tolist(),40)
            for z in z_sample:
                z = int(z)
                for t in range(0,data.shape[-1]):
                  
                    if z == 0:
                        input_z = tf.convert_to_tensor(np.reshape(data[i,:,:,z:z+2,t],[1,image_size,2*image_size,2,1]).astype(np.float32)/255)
                        input_z_pad = tf.zeros(shape = [1,image_size,2*image_size,1,1])
                        input_vol = tf.concat([input_z_pad,input_z],axis=3)
                        
                    elif z == data.shape[3]-1:
                        input_z = tf.convert_to_tensor(np.reshape(data[i,:,:,z-1:z+1,t],[1,image_size,2*image_size,2,1]).astype(np.float32)/255)
                        input_z_pad = tf.zeros(shape = [1,image_size,2*image_size,1,1])
                        input_vol = tf.concat([input_z,input_z_pad],axis=3)
                        
                    else:
                        input_vol = tf.convert_to_tensor(np.reshape(data[i,:,:,z-1:z+2,t],[1,image_size,2*image_size,3,1]).astype(np.float32)/255)
                        
                    if t == 0:
                        prev_step = tf.zeros(shape = [batch_size,image_size//2,image_size//2,3,6])

                    else:
                        prev_step = rnn_out[-1]
                        
                    inputs = tf.slice(input_vol,[0,0,0,0,0],[batch_size,image_size,image_size,3,1])

                    targets = tf.convert_to_tensor(np.reshape(data[i,:,image_size::,z,t],[1,image_size,image_size,1,1]).astype(np.float32)/255)

                    loss, output = train_step(inputs,prev_step,targets)
                    model_loss.append(loss)
                    rnn_out.append(output[1])

        print('Epochs {}/{}, Loss = {}'.format(
            epoch, epochs, model_loss[-1]))
        
        if epoch % 1 == 0:
            print('Saving training log...')
            with writer.as_default():
                tf.summary.scalar('summary_loss', model_loss[-1], step=epoch)

        if epoch % 20 == 0:
            print('Saving training checkpoint...')
            
            if not os.path.exists(a.checkpoint + '/'):
                os.makedirs(a.checkpoint + '/')

            model.save_weights(a.checkpoint + '/')

            
            
def train_plus():
    global tensorboard_dir
    now = datetime.utcnow().strftime("%Y-%m-%d_%Hh-%Mm-%Ss")
    method = 'newby3'
    tensorboard_dir = "{}/run_{}_{}/".format(tensorboard_dir, method, now)
    
    # Number of movies per set, movie length
    n = 10
    mov_len = 20
    
    writer = tf.summary.create_file_writer(tensorboard_dir)    
    for epoch in range(a.pre_epoch+1, a.pre_epoch + epochs + 1):
        
        #Generate New Training Set of 10 movies
        if epoch % 50 == 1:
            data = []
            for _ in range(0,n):
                snr = 1.5+4.5*np.random.rand()
                mov, _ = movie_maker(snr,mov_len)
                data.append(mov)
            data = np.array(data).astype(np.uint8)
            
        model_loss = []
        for i in range(0,n):
          
            rnn_out = []
            
            z_sample = rd.sample(np.linspace(21,60,40).tolist(),40)
            for z in z_sample:
                z = int(z)
                for t in range(0,data.shape[-1]):
                  
                    if z == 0:
                        input_z = tf.convert_to_tensor(np.reshape(data[i,:,:,z:z+2,t],[1,image_size,2*image_size,2,1]).astype(np.float32)/255)
                        input_z_pad = tf.zeros(shape = [1,image_size,2*image_size,1,1])
                        input_vol = tf.concat([input_z_pad,input_z],axis=3)
                        
                    elif z == data.shape[3]-1:
                        input_z = tf.convert_to_tensor(np.reshape(data[i,:,:,z-1:z+1,t],[1,image_size,2*image_size,2,1]).astype(np.float32)/255)
                        input_z_pad = tf.zeros(shape = [1,image_size,2*image_size,1,1])
                        input_vol = tf.concat([input_z,input_z_pad],axis=3)
                        
                    else:
                        input_vol = tf.convert_to_tensor(np.reshape(data[i,:,:,z-1:z+2,t],[1,image_size,2*image_size,3,1]).astype(np.float32)/255)
                        
                    if t == 0:
                        prev_step = tf.zeros(shape = [batch_size,image_size//2,image_size//2,3,6])

                    else:
                        prev_step = rnn_out[-1]
                        
                    inputs = tf.slice(input_vol,[0,0,0,0,0],[batch_size,image_size,image_size,3,1])

                    targets = tf.convert_to_tensor(np.reshape(data[i,:,image_size::,z,t],[1,image_size,image_size,1,1]).astype(np.float32)/255)

                    loss, output = train_step(inputs,prev_step,targets)
                    model_loss.append(loss)
                    rnn_out.append(output[1])

        print('Epochs {}/{}, Loss = {}'.format(
            epoch, epochs, model_loss[-1]))
        
            
        if epoch % 1 == 0:
            print('Saving training log...')
            with writer.as_default():
                tf.summary.scalar('summary_loss', model_loss[-1], step=epoch+a.pre_epoch)
                
        if epoch % 10 == 0:
            print('Saving training checkpoint...')
            
            if not os.path.exists(a.checkpoint + '/'):
                os.makedirs(a.checkpoint + '/')
            
            model.save_weights(a.checkpoint + '/')

            
def load_test():
        
    onlyfiles = sorted([f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))])
    raw_data = np.array([sio.loadmat(test_dir + f)['instance'] for f in onlyfiles if f.endswith('.mat')]).astype(np.float32)

    return raw_data


def find_threshold(outputs):
    
    tv = np.linspace(0.01,1.0,10)
    
    #record the number of detections and how far this is from the target 92
    detected_spots = []
    diff_from_target = []
    
    for threshold in tv:
        tmp = outputs > threshold
        x_labels = label(tmp)[0]
        merged_peaks = center_of_mass(tmp,x_labels,range(1, np.max(x_labels)+1))
        y = np.array(merged_peaks)
        detected_spots.append(len(y))
        diff_from_target.append(abs(92-len(y)))
        
    # find the optimal threshold and the number detected at this value
    ind = np.argmin(diff_from_target)
    
    optimal_tv = tv[ind]
    detections = detected_spots[ind]
    
    return optimal_tv, detections


def local_maxima_3D(data, order=1):
    """Detects local maxima in a 3D array

    Parameters
    ---------
    data : 3d ndarray
    order : int
        How many points on each side to use for the comparison

    Returns
    -------
    coords : ndarray
        coordinates of the local maxima
    values : ndarray
        values of the local maxima
    """
    size = 1 + 2 * order
    footprint = np.ones((size, size, size))
    footprint[order, order, order] = 0

    filtered = ndi.maximum_filter(data, footprint=footprint)
    mask_local_maxima = data > filtered
    coords = np.asarray(np.where(mask_local_maxima)).T
    values = data[mask_local_maxima]

    return coords, values


def normalize(arr):
    # Normalize the image to be between 0 and 1
    arr = (arr - np.min(arr))/(np.max(arr) - np.min(arr)) 
    return arr


def min_dist(arr):
    
    n = len(arr[:,0])
    
    X = np.zeros([n,n])
    Y = np.zeros([n,n])
    Z = np.zeros([n,n])
    
    for i in range(0,n):
        for j in range(0,n):
            X[i,j] = (arr[i,0] - arr[j,0])**2
            Y[i,j] = (arr[i,1] - arr[j,1])**2
            Z[i,j] = (arr[i,2] - arr[j,2])**2
            
    M = np.sqrt(X+Y+Z)
    for i in range(0,n):
        M[i,i] = 10000
        
    return np.min(M)
  
  
def post_process(out):
  
  tmp = out > 0.2
  tmp = np.pad(out*tmp,((2,2),(2,2),(2,2)))
  y, vals = local_maxima_3D(tmp, order=1)
  pos = []
  tmp_pos = np.zeros([92,3])
  zipped = sorted(tuple(zip(vals,y)),key=lambda x: x[0],reverse=True)
  c1 = 0
  c2 = 0
  
  while len(pos) < 92 and c1 < len(y):
    if zipped[c1][0] < 0.3:
      c1 += 100000
    if c1 < len(y):
      tmp_pos[c2,:] = zipped[c1][1]
      if min_dist(tmp_pos[0:c2+1,:]) > 2 and c1 < len(y):
        
        #Shift due to padding
        detected_pos = [zipped[c1][1][0]-2,zipped[c1][1][1]-2,zipped[c1][1][2]-2]
        
        pos.append(detected_pos)
        c2 += 1
    c1 += 1
  y = np.array(pos)
  
  return y


def reshape_volume_data(arr,in_dims,out_dims):
    # Function reshapes the volume data to and from the compatible shape used by the network
    # in/out dims [x,y,z], arr.shape = [x,y,z]

    out = np.zeros([out_dims[0],out_dims[1],out_dims[2]]).astype(np.float32)
    tmp = np.zeros([out_dims[0],out_dims[1],in_dims[2]]).astype(np.float32)
    for i in range(0,in_dims[2]):
        tmp[:,:,i] = cv2.resize(arr[:,:,i],(out_dims[1],out_dims[0]),interpolation = cv2.INTER_LINEAR)
        
    for j in range(0,out_dims[0]):
        out[j,:,:] = cv2.resize(tmp[j,:,:],(out_dims[2],out_dims[1]),interpolation = cv2.INTER_LINEAR)
        
    return out
    


def test():
  
    method = 'newby3'
    model = create_model()

    model.load_weights(a.checkpoint + '/')
    
    output_dir = './'+a.out_dir+'/'+method+'/method2/'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
##########Retrieve all the data from the test directory##########
    filenames = sorted([f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f)) and f.endswith('.mat')])
    print(filenames)
    for i in range(0,len(filenames)):
        
        input_data = (np.array([sio.loadmat(test_dir + f)['instance'] for f in filenames[i:i+1] if f.endswith('.mat')]).astype(np.float32))/255
        
        [_,in_x,in_y,in_z,in_t] = input_data.shape
        
########Reshape Data such that it is compatible#############

        data = np.zeros([1,image_size,image_size,in_z,in_t]).astype(np.float32)
        for t in range(0,in_t):
            data[0,:,:,:,t] = reshape_volume_data(input_data[0,:,:,:,t], [in_x,in_y,in_z], [image_size,image_size,in_z])

        
        out = np.zeros([image_size,image_size,in_z,in_t]).astype(np.float32)
        
        for z in range(0,in_z):
            print('Evaluating slice {} of {}'.format(z,in_z))
            for t in range(0,in_t):
            
                if z == 0:
                    input_z = tf.convert_to_tensor(np.reshape(data[0,:,:,z:z+2,t],[1,image_size,image_size,2,1]))
                    input_z_pad = tf.convert_to_tensor(np.reshape(data[0,:,:,z,t],[1,image_size,image_size,1,1]))
                    input_vol = tf.concat([input_z_pad,input_z],axis=3)

                elif z == in_z-1:
                    input_z = tf.convert_to_tensor(np.reshape(data[0,:,:,z-1:z+1,t],[1,image_size,image_size,2,1]))
                    input_z_pad = tf.convert_to_tensor(np.reshape(data[0,:,:,z,t],[1,image_size,image_size,1,1]))
                    input_vol = tf.concat([input_z,input_z_pad],axis=3)
                    
                else:
                    input_vol = tf.convert_to_tensor(np.reshape(data[0,:,:,z-1:z+2,t],[1,image_size,image_size,3,1]))
                    
                if t == 0:
                    prev_step = tf.zeros(shape = [batch_size,image_size//2,image_size//2,3,6])
                    

                inputs = tf.slice(input_vol,[0,0,0,0,0],[batch_size,image_size,image_size,3,1])

                tf_output, prev_step = model([inputs,prev_step], training=True)

                out[:,:,z,t] = tf_output.numpy()[0,:,:,0,0]
            
        #Save Data
        print("Analysing and Saving Movie {}...")
        number_of_detections = []
        positions = []
        
        
#######Create.csv file for the spot positions#############
        with open(output_dir+filenames[i][0:-4]+'_'+method+'_positions.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Frame', 'xPixel', 'yPixel', 'zPixel', 'ID'])
        
        for t in range(0,in_t):
          
          y = post_process(out[:,:,:,t])

          positions.append(y)
          detections = len(y[:,0])

          scale = np.ones(y.shape)
          scale[:,0] = (in_x-1)/(image_size-1)
          scale[:,1] = (in_y-1)/(image_size-1)

          loc = y*scale

          id_counter = 1

          #When writing the csv file, indexing starts at 1 in matlab for KiT and 0 for python

          for spot in range(0,len(loc[:,0])):
              with open(output_dir+filenames[i][0:-4]+'_'+method+'_positions.csv', 'a', newline='') as csvfile:
                  writer = csv.writer(csvfile, delimiter=',',
                                      quotechar='|', quoting=csv.QUOTE_MINIMAL)
                  writer.writerow(['{}'.format(t+1), 
                                   '{}'.format(loc[spot,0]+1), 
                                   '{}'.format(loc[spot,1]+1), 
                                   '{}'.format(loc[spot,2]+1), 
                                   '{}'.format(id_counter)])
                  id_counter += 1


              number_of_detections.append(detections)
            
        np.save(output_dir+'/'+filenames[i][0:-4]+'_'+method+'_detections.npy',np.array(number_of_detections))
        
                
#########Save the outputs (optional)###########
        
        outmat = np.zeros([in_x,in_y,in_z,in_t]).astype(np.uint8)
        for t in range(0,in_t):
              tmp = reshape_volume_data(out[:,:,:,t], [image_size,image_size,in_z], [in_x,in_y,in_z])
              outmat[:,:,:,t] = (255*tmp).astype(np.uint8)
        
        print('Saving Outputs...')
        sio.savemat(output_dir+'/mat/'+filenames[i][0:-4]+'_'+method+'_outputs.mat', {'instance': outmat})
###############################################
        
            

def evaluate():
    method = 'newby3'
    if not os.path.exists('./arrays/'):
        os.makedirs('./arrays/')
    
    snr = [1.5,2.0,3.0,4.0,5.0,6.0]
    
    FALSEPOSITIVES = []
    FALSENEGATIVES = []
    TRUEPOSITIVES = []
    PRECISION = []
    RECALL = []
    F1SCORE = []
    PIXELERROR = []
    LOCERROR = []
    MAXLOCERROR = []
    DETECTED = []
    
    for SNR in snr:
        print('Testing SNR:', SNR)
        
        aFP1 = []
        aFN1 = []
        aTP1 = []
        P1 = []
        R1 = []
        F1_1 = []
        aMinDist1 = []
        aMaxDist1 = []
        aPWE1 = []
        aDetect1 = []
        
        for i in range(0,movies):
            print('Generating test movie {}...'.format(i))
            movie, loc = movie_maker(SNR,10)
            
            [in_x,in_y,in_z,in_t] = movie.shape
            data = np.reshape(movie[:,0:256,:,:].astype(np.float32)/255, [1,image_size,image_size,in_z,in_t])
            targets = movie[:,256::,:,:].astype(np.float32)/255
            out = np.zeros([image_size,image_size,in_z,in_t]).astype(np.float32)
        
            for z in range(0,in_z):
                print('Evaluating slice {} of {}'.format(z,in_z))
                for t in range(0,in_t):

                    if z == 0:
                        input_z = tf.convert_to_tensor(np.reshape(data[0,:,:,z:z+2,t],[1,image_size,image_size,2,1]))
                        input_z_pad = tf.convert_to_tensor(np.reshape(data[0,:,:,z,t],[1,image_size,image_size,1,1]))
                        input_vol = tf.concat([input_z_pad,input_z],axis=3)

                    elif z == in_z-1:
                        input_z = tf.convert_to_tensor(np.reshape(data[0,:,:,z-1:z+1,t],[1,image_size,image_size,2,1]))
                        input_z_pad = tf.convert_to_tensor(np.reshape(data[0,:,:,z,t],[1,image_size,image_size,1,1]))
                        input_vol = tf.concat([input_z,input_z_pad],axis=3)

                    else:
                        input_vol = tf.convert_to_tensor(np.reshape(data[0,:,:,z-1:z+2,t],[1,image_size,image_size,3,1]))

                    if t == 0:
                        prev_step = tf.zeros(shape = [batch_size,image_size//2,image_size//2,3,6])


                    inputs = tf.slice(input_vol,[0,0,0,0,0],[batch_size,image_size,image_size,3,1])

                    tf_output, prev_step = model([inputs,prev_step], training=True)

                    out[:,:,z,t] = tf_output.numpy()[0,:,:,0,0]

                print('Evaluating Output...')
                
            for t in range(0,in_t):
                  
                  y = loc[:,:,t].astype(np.uint8)
                  detected_targets = len(y[:,0])


                  FP1 = 0

                  x1 = post_process(out[:,:,:,t])

                  detected_targets = len(x1)

                  distances = []

                  for i in range(0,len(x1)):
                          distance = np.sqrt((x1[i,0]-y[:,1])**2+(x1[i,1]-y[:,0])**2+(x1[i,2]-y[:,2])**2)
                          distances.append(min(distance))
                          if min(distance) > 2:
                              FP1 += 1

                  aMinDist1.append(np.mean(distances))
                  aMaxDist1.append(np.max(distances))

                  TP1 = len(x1) - FP1
                  FN1 = len(y[:,0])-(len(x1) - FP1)

                  aFP1.append(FP1)
                  aFN1.append(FN1)
                  aTP1.append(TP1)
                  P1.append(TP1/(TP1+FP1+1e-10))
                  R1.append(TP1/(TP1+FN1+1e-10))
                  F1_1.append(2*P1[-1]*R1[-1]/(P1[-1]+R1[-1]+1e-10))

                  aDetect1.append(len(x1))
                  average_pixel_error = []
                  # Mean-Squared Error per pixel around a true spot
                  for i in range(0,len(y[:,0])):
                      true_box = targets[y[i,1]-2:y[i,1]+3,y[i,0]-2:y[i,0]+3,y[i,2]-2:y[i,2]+3,t]
                      predicted_box = out[y[i,1]-2:y[i,1]+3,y[i,0]-2:y[i,0]+3,y[i,2]-2:y[i,2]+3,t]
                      average_pixel_error.append(np.sum(np.abs(true_box-predicted_box))/(5**3))

                  aPWE1.append(np.mean(average_pixel_error))

        print('Appending Results...')
        FALSEPOSITIVES.append(np.mean(aFP1))
        FALSENEGATIVES.append(np.mean(aFN1))
        TRUEPOSITIVES.append(np.mean(aTP1))
        PRECISION.append(np.mean(P1))
        RECALL.append(np.mean(R1))
        F1SCORE.append(np.mean(F1_1))
        PIXELERROR.append(np.mean(aPWE1))
        LOCERROR.append(np.mean(aMinDist1))
        MAXLOCERROR.append(np.mean(aMaxDist1))
        DETECTED.append(np.mean(aDetect1))

        print('FALSEPOSITIVES',FALSEPOSITIVES[-1])
        print('FALSENEGATIVES',FALSENEGATIVES[-1])
        print('TRUEPOSITIVES',TRUEPOSITIVES[-1])
        print('PRECISION',PRECISION[-1])
        print('RECALL',RECALL[-1])
        print('F1SCORE',F1SCORE[-1])
        print('LOCATION ERROR',LOCERROR[-1])
        print('MAX. LOCATION ERROR',MAXLOCERROR[-1])
        print('PIXELERROR',PIXELERROR[-1])
        print('DETECTED',DETECTED[-1])
    
    
    if a.save_arr == "Yes":
        np.save('./arrays/'+method+'_false-positives.npy',np.array(FALSEPOSITIVES))
        np.save('./arrays/'+method+'_false-negatives.npy',np.array(FALSENEGATIVES))
        np.save('./arrays/'+method+'_true-positives.npy',np.array(TRUEPOSITIVES))
        np.save('./arrays/'+method+'_precision.npy',np.array(PRECISION))
        np.save('./arrays/'+method+'_recall.npy',np.array(RECALL))
        np.save('./arrays/'+method+'_f1-score.npy',np.array(F1SCORE))
        np.save('./arrays/'+method+'_pixel-error.npy',np.array(PIXELERROR))
        np.save('./arrays/'+method+'_loc-error.npy',np.array(LOCERROR))
        np.save('./arrays/'+method+'_max-loc-error.npy',np.array(MAXLOCERROR))
        np.save('./arrays/'+method+'_detected.npy',np.array(DETECTED))
    
    
        
if a.mode == "train":
    epochs = a.epochs
    train()
    
elif a.mode == "train+":
    model.load_weights(a.checkpoint+'/')
    epochs = a.epochs
    train_plus()
    
elif a.mode == "evaluate":
    model.load_weights(a.checkpoint + '/')
    movies = a.evaluation_volumes//10
    evaluate() 
    
    
elif a.mode == "test":
    test_dir = './'+a.test_dir+'/'
    output_dir = './'+a.out_dir+'/'
    test()
    
else:
    print("Enter valid mode")
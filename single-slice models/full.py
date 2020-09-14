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
import math
import argparse
import csv

from scipy import ndimage as ndi
from scipy.ndimage.measurements import center_of_mass, label

from simulation import movie_maker

from datetime import datetime
import time


parser = argparse.ArgumentParser()
parser.add_argument("--mode", type = str, required = True)
parser.add_argument("--checkpoint",type = str,required = True)
parser.add_argument("--out_dir", type = str)
parser.add_argument("--test_dir",type = str)
parser.add_argument("--pre_epoch", type = int)
parser.add_argument("--epochs", type = int)
parser.add_argument("--evaluation_volumes", type = int)
parser.add_argument("--save_arr",type = str, default = "Yes")
parser.add_argument("--postproc", type = str, default = '1')
a = parser.parse_args()


tensorboard_dir = "./tf_log/"
batch_size = 1
image_size = 256


def create_model():
    
    network_layers = []
    
    inputs = tf.keras.Input(shape=(image_size,image_size,1))
    
    x = layers.Conv2D(64,3,strides=1, padding="SAME",kernel_initializer=tf.random_normal_initializer(0,0.02))(inputs)
    network_layers.append(x)
    
    encoder_specs = [128,256,512,1024]
    
    for out_channels in encoder_specs:
        rectified = layers.LeakyReLU(alpha=0.2)(network_layers[-1])
        
        convolved = layers.Conv2D(
            out_channels, 3, strides=2, padding="SAME",
            kernel_initializer=tf.random_normal_initializer(0,0.02))(rectified)
        
        output = layers.BatchNormalization(
            epsilon=1e-5, momentum=0.1,trainable=True,
            gamma_initializer=tf.random_normal_initializer(1.0, 0.02))(convolved)
        
        network_layers.append(output)
    
    
    decoder_specs = [(512,0.5),(256,0.5),(128,0.0),(64,0.0)]
    
    num_encoder_layers = len(network_layers)
    
    for decoder_layer, (out_channels, dropout) in enumerate(decoder_specs):
        
        skip_layer = num_encoder_layers - decoder_layer -1
        
        if decoder_layer == 0:
            x = network_layers[-1]
        else:
            x = layers.Concatenate(axis=3)([network_layers[-1],network_layers[skip_layer]])
            
        rectified = layers.ReLU()(x)
        
        output = layers.Conv2DTranspose(
            out_channels,3,strides=[2,2],padding="SAME", 
            kernel_initializer=tf.random_normal_initializer(0,0.02))(rectified)
        
        output = layers.BatchNormalization(
            epsilon=1e-5, momentum=0.1,trainable=True,
            gamma_initializer=tf.random_normal_initializer(1.0, 0.02))(output)
        
        if dropout > 0.0:
            output = layers.Dropout(rate=1-dropout)(output)
            
        network_layers.append(output)
        
    x = layers.Concatenate(axis=3)([network_layers[-1], network_layers[0]])
    rectified = layers.ReLU()(x)
    output = layers.Conv2DTranspose(
            1,3,strides=[1,1],padding="SAME", 
            kernel_initializer=tf.random_normal_initializer(0,0.02))(rectified)
    output = tf.keras.activations.sigmoid(output)
    network_layers.append(output)
    
    model = tf.keras.Model(inputs=inputs, outputs=network_layers[-1])
    return model


model = create_model()
model.summary()
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1 = 0.5)


@tf.function
def loss_function(outputs,targets):
    w_1 = 0.2
    w_2 = 1.0
    loss = tf.reduce_sum((w_1*targets+w_2*tf.keras.backend.mean(targets))*(targets-outputs)**2)
    return loss


@tf.function
def train_step(inputs, targets):
    
    with tf.GradientTape() as tape:
        outputs = model(inputs,training=True)
        
        loss = loss_function(outputs,targets)
        
    gradients = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    
    return loss


def train():
    global tensorboard_dir
    now = datetime.utcnow().strftime("%Y-%m-%d_%Hh-%Mm-%Ss")
    method = 'full'
    tensorboard_dir = "{}/run_{}_{}/".format(tensorboard_dir, method, now)
    
   
    writer = tf.summary.create_file_writer(tensorboard_dir)
    for epoch in range(1,epochs+1):
      model_loss = []
        
      if epoch % 5 == 1:
          snr = 1.5+4.5*np.random.rand()
          data, _ = movie_maker(snr,20)
          z_sample = rd.sample(np.linspace(21,60,40).tolist(),40)
        
      for z in z_sample:
          
          image_batch = tf.convert_to_tensor(np.reshape(data[:,:,int(z),:],[1,image_size,2*image_size,data.shape[-1]]).astype(np.float32))
          for i in range(0,image_batch.shape[-1]):
                
              inputs = tf.slice(image_batch,[0,0,0,i],[batch_size,image_size,image_size,1])
              targets = tf.slice(image_batch,[0,0,image_size,i],[batch_size,image_size,image_size,1])
              
              loss = train_step(inputs,targets)
              model_loss.append(loss)
            
      print('Epochs {}/{}, Loss = {}'.format(
          epoch, epochs, np.mean(model_loss)))
            
      if epoch % 1 == 0:
          print('Saving training log...')
          with writer.as_default():
              tf.summary.scalar('summary_loss', np.mean(model_loss), step=epoch)
                
      if epoch % 10 == 0:
          print('Saving training checkpoint...')
            
          if not os.path.exists(a.checkpoint + '/'):
              os.makedirs(a.checkpoint + '/')
            
          model.save_weights(a.checkpoint + '/')
            
            
def train_plus():
    global tensorboard_dir
    now = datetime.utcnow().strftime("%Y-%m-%d_%Hh-%Mm-%Ss")
    method = 'full'
    tensorboard_dir = "{}/run_{}_{}/".format(tensorboard_dir, method, now)
    
    model_loss = []
    writer = tf.summary.create_file_writer(tensorboard_dir)
    for epoch in range(a.pre_epoch+1, a.pre_epoch + epochs + 1):
        
        if epoch % 5 == 1:
            snr = 1.5+4.5*np.random.rand()
            data, _ = movie_maker(snr,20)
            z_sample = rd.sample(np.linspace(21,60,40).tolist(),40)
        
        for z in z_sample:
            
            image_batch = tf.convert_to_tensor(np.reshape(data[:,:,int(z),:],[1,image_size,2*image_size,data.shape[-1]]).astype(np.float32))
            for i in range(0,image_batch.shape[-1]):
                
                inputs = tf.slice(image_batch,[0,0,0,i],[batch_size,image_size,image_size,1])
                targets = tf.slice(image_batch,[0,0,image_size,i],[batch_size,image_size,image_size,1])

                loss = train_step(inputs,targets)
                model_loss.append(loss)
            
        print('Epochs {}/{}, Loss = {}'.format(
            epoch, epochs, np.mean(model_loss)))
            
        if epoch % 1 == 0:
            print('Saving training log...')
            with writer.as_default():
                tf.summary.scalar('summary_loss', np.mean(model_loss), step=epoch)
                
        if epoch % 10 == 0:
            print('Saving training checkpoint...')
            
            if not os.path.exists(a.checkpoint + '/'):
                os.makedirs(a.checkpoint + '/')
            
            model.save_weights(a.checkpoint + '/')   
            
            
def load_test():
    
    onlyfiles = sorted([f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))])
    
    # Load hyperstack (n,256,512,z,t)
    raw_data = np.array([sio.loadmat(test_dir + f)['instance'] for f in onlyfiles if f.endswith('.mat')]).astype(np.float32)
    
    return raw_data, onlyfiles



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
  
def process_1(out):
  
  # Find optimal threshold value
  opt_tv, detections = find_threshold(out)
  
  tmp = out > opt_tv
  x_labels = label(tmp)[0]
  merged_peaks = center_of_mass(tmp,x_labels,range(1, np.max(x_labels)+1))
  y = np.array(merged_peaks)
  
  print('Detected: ',len(y[:,0]), 'Opt. Thresh. :', opt_tv)
  
  return y
  
  
  
def process_2(out):
  
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
      if min_dist(tmp_pos[0:c2+1,:]) > 3 and c1 < len(y):
        
        #Shift due to padding
        detected_pos = [zipped[c1][1][0]-2,zipped[c1][1][1]-2,zipped[c1][1][2]-2]
        
        pos.append(detected_pos)
        c2 += 1
    c1 += 1
  y = np.array(pos)
  print(len(y))
  
  return y



def test():
  
    method = 'full'
    
    if a.postproc == '1':
      print('Post-Processing Method: Mask')
    elif a.postproc == '2':
      print('Post-Processing Method: Peak Local Max')
    else:
      raise Exception('Must specify Valid Post-Processing Method')
      
    # Make the output location
    
    output_dir = './'+a.out_dir+'/'+method+'/method'+a.postproc+'/'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.load_weights(a.checkpoint + '/')
    
    filenames = sorted([f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f)) and f.endswith('.mat')])
    print(filenames)
    for i in range(0,len(filenames)):
        
        input_data = (np.array([sio.loadmat(test_dir + f)['instance'] for f in filenames[i:i+1] if f.endswith('.mat')]).astype(np.float32))/255
        
        [_,in_x,in_y,in_z,in_t] = input_data.shape
        
        thresholds = []
        number_of_detections = []
        positions = []
        
        with open(output_dir+filenames[i][0:-4]+'_'+method+'_'+a.postproc+'_positions.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Frame', 'xPixel', 'yPixel', 'zPixel', 'ID'])
            
        outmat = np.zeros([image_size,image_size,in_z,in_t]).astype(np.uint8)
        
        for t in range(0,in_t):
            # Output is a volume (256,256,z)
            out = np.zeros([image_size,image_size,in_z]).astype(np.float32)
            
            for z in range(0,in_z):
                
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                inputs = np.reshape(cv2.resize(input_data[0,:,:,z,t],(256,256), interpolation=cv2.INTER_NEAREST),[1,image_size,image_size,1])
                tf_inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)

                tf_outputs = model(tf_inputs, training=True)
                out[:,:,z] =  tf_outputs.numpy()[0,:,:,0]

            #Save Data
            out = normalize(out)
            outmat[:,:,:,t] = (out*255).astype(np.uint8)
            
            print("Analysing and Saving Volume {} of Movie {}...".format(t+1,i+1))
            
            
            if a.postproc == '1':
              y = process_1(out)
            elif a.postproc == '2':
              y = process_2(out)
              
            positions.append(y)
            detections = len(y[:,0])
                
            scale = np.ones(y.shape)
            scale[:,0] = (in_x-1)/(image_size-1)
            scale[:,1] = (in_y-1)/(image_size-1)
                      
            loc = y*scale
            
            id_counter = 1
            
            #When writing the csv file, indexing starts at 1 in matlab for KiT and 0 for python
            
            for spot in range(0,len(loc[:,0])):
                with open(output_dir+filenames[i][0:-4]+'_'+method+'_'+a.postproc+'_positions.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(['{}'.format(t+1), 
                                     '{}'.format(loc[spot,0]+1), 
                                     '{}'.format(loc[spot,1]+1), 
                                     '{}'.format(loc[spot,2]+1), 
                                     '{}'.format(id_counter)])
                id_counter += 1
            
                                    
            number_of_detections.append(detections)
            
        
            
#         sio.savemat(output_dir+'/'+filenames[i][0:-4]+'_'+method+'_'+a.postproc+'_outputs.mat', {'instance': outmat})
            
        np.save(output_dir+'/'+filenames[i][0:-4]+'_'+method+'_'+a.postproc+'_detections.npy',np.array(number_of_detections))
            
            
def evaluate():
    method = 'full'
    if not os.path.exists('./arrays/'):
        os.makedirs('./arrays/')
    
    snr = [1.5,2.0,3.0,4.0,5.0,6.0]
    
    FALSEPOSITIVES1 = []
    FALSENEGATIVES1 = []
    TRUEPOSITIVES1 = []
    PRECISION1 = []
    RECALL1 = []
    F1SCORE1 = []
    PIXELERROR1 = []
    LOCERROR1 = []
    MAXLOCERROR1 = []
    DETECTED1 = []
    
    FALSEPOSITIVES2 = []
    FALSENEGATIVES2 = []
    TRUEPOSITIVES2 = []
    PRECISION2 = []
    RECALL2 = []
    F1SCORE2 = []
    PIXELERROR2 = []
    LOCERROR2 = []
    MAXLOCERROR2 = []
    DETECTED2 = []
    
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
        
        aFP2 = []
        aFN2 = []
        aTP2 = []
        P2 = []
        R2 = []
        F1_2 = []
        aMinDist2 = []
        aMaxDist2 = []
        aPWE2 = []
        aDetect2 = []
        
        
########################################################################################################################### 
        
        for i in range(0,movies):
            print('Generating test movie {}...'.format(i))
            movie, loc = movie_maker(SNR,10)
            
            for t in range(0,movie.shape[-1]):
                print('Running Model...')
                out = np.zeros([image_size,image_size,movie.shape[2]])
                
                for z in range(0,movie.shape[2]):
                    
                    
                    input_slice = tf.convert_to_tensor(np.reshape(movie[:,0:image_size,z,t],[1,image_size,image_size,1]))
                    
                    out_slice = model(input_slice, training=False)
                    
                    out[:,:,z] = out_slice.numpy()[0,:,:,0]
                    
                targets = movie[:,image_size::,:,t]
                
#############################################################################################################################
                #Evaluating Output
  
  
                y = loc[:,:,t].astype(np.uint8)
                detected_targets = len(y[:,0])
      
                # Mask Method
                
                FP1 = 0

                x1 = process_1(out)
          
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
                    true_box = targets[y[i,1]-2:y[i,1]+3,y[i,0]-2:y[i,0]+3,y[i,2]-2:y[i,2]+3]
                    predicted_box = out[y[i,1]-2:y[i,1]+3,y[i,0]-2:y[i,0]+3,y[i,2]-2:y[i,2]+3]
                    average_pixel_error.append(np.sum(np.abs(true_box-predicted_box))/(5**3))

                aPWE1.append(np.mean(average_pixel_error))
                
                # Peak Local Max Method
                
                FP2 = 0
                
                x2 = process_2(out)
                
                detected_targets = len(x2)

                distances = []

                for i in range(0,len(x2)):
                        distance = np.sqrt((x2[i,0]-y[:,1])**2+(x2[i,1]-y[:,0])**2+(x2[i,2]-y[:,2])**2)
                        distances.append(min(distance))
                        if min(distance) > 2:
                            FP2 += 1

                aMinDist2.append(np.mean(distances))
                aMaxDist2.append(np.max(distances))

                TP2 = len(x2) - FP2
                FN2 = len(y[:,0])-(len(x2) - FP2)

                aFP2.append(FP2)
                aFN2.append(FN2)
                aTP2.append(TP2)
                P2.append(TP2/(TP2+FP2+1e-10))
                R2.append(TP2/(TP2+FN2+1e-10))
                F1_2.append(2*P2[-1]*R2[-1]/(P2[-1]+R2[-1]+1e-10))

                aDetect2.append(len(x2))
                average_pixel_error = []
                # Mean-Squared Error per pixel around a true spot
                for i in range(0,len(y[:,0])):
                    true_box = targets[y[i,1]-2:y[i,1]+3,y[i,0]-2:y[i,0]+3,y[i,2]-2:y[i,2]+3]
                    predicted_box = out[y[i,1]-2:y[i,1]+3,y[i,0]-2:y[i,0]+3,y[i,2]-2:y[i,2]+3]
                    average_pixel_error.append(np.sum(np.abs(true_box-predicted_box))/(5**3))

                aPWE2.append(np.mean(average_pixel_error))

        print('Appending Results...')
        FALSEPOSITIVES1.append(np.mean(aFP1))
        FALSENEGATIVES1.append(np.mean(aFN1))
        TRUEPOSITIVES1.append(np.mean(aTP1))
        PRECISION1.append(np.mean(P1))
        RECALL1.append(np.mean(R1))
        F1SCORE1.append(np.mean(F1_1))
        PIXELERROR1.append(np.mean(aPWE1))
        LOCERROR1.append(np.mean(aMinDist1))
        MAXLOCERROR1.append(np.mean(aMaxDist1))
        DETECTED1.append(np.mean(aDetect1))
        
        print('Mask Method')
        print('FALSEPOSITIVES',FALSEPOSITIVES1[-1])
        print('FALSENEGATIVES',FALSENEGATIVES1[-1])
        print('TRUEPOSITIVES',TRUEPOSITIVES1[-1])
        print('PRECISION',PRECISION1[-1])
        print('RECALL',RECALL1[-1])
        print('F1SCORE',F1SCORE1[-1])
        print('LOCATION ERROR',LOCERROR1[-1])
        print('MAX. LOCATION ERROR',MAXLOCERROR1[-1])
        print('PIXELERROR',PIXELERROR1[-1])
        print('DETECTED',DETECTED1[-1])
        
        FALSEPOSITIVES2.append(np.mean(aFP2))
        FALSENEGATIVES2.append(np.mean(aFN2))
        TRUEPOSITIVES2.append(np.mean(aTP2))
        PRECISION2.append(np.mean(P2))
        RECALL2.append(np.mean(R2))
        F1SCORE2.append(np.mean(F1_2))
        PIXELERROR2.append(np.mean(aPWE2))
        LOCERROR2.append(np.mean(aMinDist2))
        MAXLOCERROR2.append(np.mean(aMaxDist2))
        DETECTED2.append(np.mean(aDetect2))
        
        print('Peak Local Max. Method')
        print('FALSEPOSITIVES',FALSEPOSITIVES2[-1])
        print('FALSENEGATIVES',FALSENEGATIVES2[-1])
        print('TRUEPOSITIVES',TRUEPOSITIVES2[-1])
        print('PRECISION',PRECISION2[-1])
        print('RECALL',RECALL2[-1])
        print('F1SCORE',F1SCORE2[-1])
        print('LOCATION ERROR',LOCERROR2[-1])
        print('MAX. LOCATION ERROR',MAXLOCERROR2[-1])
        print('PIXELERROR',PIXELERROR2[-1])
        print('DETECTED',DETECTED2[-1])
    
    
    if a.save_arr == "Yes":
        np.save('./arrays/'+method+'_1-false-positives.npy',np.array(FALSEPOSITIVES1))
        np.save('./arrays/'+method+'_1-false-negatives.npy',np.array(FALSENEGATIVES1))
        np.save('./arrays/'+method+'_1-true-positives.npy',np.array(TRUEPOSITIVES1))
        np.save('./arrays/'+method+'_1-precision.npy',np.array(PRECISION1))
        np.save('./arrays/'+method+'_1-recall.npy',np.array(RECALL1))
        np.save('./arrays/'+method+'_1-f1-score.npy',np.array(F1SCORE1))
        np.save('./arrays/'+method+'_1-pixel-error.npy',np.array(PIXELERROR1))
        np.save('./arrays/'+method+'_1-loc-error.npy',np.array(LOCERROR1))
        np.save('./arrays/'+method+'_1-max-loc-error.npy',np.array(MAXLOCERROR1))
        np.save('./arrays/'+method+'_1-detected.npy',np.array(DETECTED1))
        
        np.save('./arrays/'+method+'_2-false-positives.npy',np.array(FALSEPOSITIVES2))
        np.save('./arrays/'+method+'_2-false-negatives.npy',np.array(FALSENEGATIVES2))
        np.save('./arrays/'+method+'_2-true-positives.npy',np.array(TRUEPOSITIVES2))
        np.save('./arrays/'+method+'_2-precision.npy',np.array(PRECISION2))
        np.save('./arrays/'+method+'_2-recall.npy',np.array(RECALL2))
        np.save('./arrays/'+method+'_2-f1-score.npy',np.array(F1SCORE2))
        np.save('./arrays/'+method+'_2-pixel-error.npy',np.array(PIXELERROR2))
        np.save('./arrays/'+method+'_2-loc-error.npy',np.array(LOCERROR2))
        np.save('./arrays/'+method+'_2-max-loc-error.npy',np.array(MAXLOCERROR2))
        np.save('./arrays/'+method+'_2-detected.npy',np.array(DETECTED2))
        
        
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
    test()
    
else:
    print("Enter valid mode")
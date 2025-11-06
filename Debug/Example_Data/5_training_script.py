#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training for Fold 14 with Post-Training Quantization
Modified from dreemTrainInd.py to train fold 14 and create optimized TFLite model
"""

import sys
sys.path.append('/Users/alikavoosi/Desktop/DREEM')

import tensorflow as tf
from nn_model import *
from dreemRead import *
from scipy.signal import resample
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from scipy.signal import butter, lfilter
import numpy as np
import os
import random

# Set random seed for reproducibility
tf.random.set_seed(100)
np.random.seed(100)
random.seed(100)

def KLD(mu1, sigma1, mu2, sigma2):
    return np.log(sigma1/sigma2) + (sigma1**2+(mu1-mu2)**2)/(2*sigma2**2) - 0.5

# Function to create a bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Function to apply the bandpass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def sub_sampling(x,x_ind,y,target=0):
    target_num = len([i for i in y if i == target])
    target_inds = [num for num,i in enumerate(y) if i == target]
    target_inds = np.array(target_inds)
    new_x = []
    new_x_ind = []
    new_y = []
    new_x.append(x[target_inds])
    new_x_ind.append(x_ind[target_inds])
    new_y.append(y[target_inds])
    for j in [1,2,3,4]:
        inds_cl = [num for num,i in enumerate(y) if i == j]
        inds_cl = np.array(inds_cl)
        num=len(inds_cl)
        if num>target_num:
            inds = np.random.choice(num, size=target_num, replace=False)
            inds = np.array(inds)
            new_x.append(x[inds_cl[inds]])
            new_x_ind.append(x_ind[inds_cl[inds]])
            new_y.append(y[inds_cl[inds]])
        else:
            new_x.append(x[inds_cl])
            new_x_ind.append(x_ind[inds_cl])
            new_y.append(y[inds_cl])
    new_x = np.vstack(new_x)
    new_x_ind = np.vstack(new_x_ind)
    new_y = np.vstack(new_y)
    return new_x,new_x_ind,new_y

def pos_var_v2(length):
    t = np.arange(0,length,1)
    y = t/1000
    return y

def get_fold_indices(fold_number, total_folds=25):
    if fold_number < 0 or fold_number >= total_folds:
        raise ValueError("Fold number is out of range")

    fold_size = total_folds // 25
    test_index = fold_number
    all_indices = list(range(total_folds))
    all_indices.remove(test_index)
    validation_indices = random.sample(all_indices, 7)
    training_indices = [
        i for i in range(total_folds) if i not in [test_index] + validation_indices
    ]
    return training_indices, [test_index], validation_indices

# Configuration
gap_seq_len = 10
seq_len = 12
fs_d = 100
epoch_len = int(fs_d*30)
sig_ac = False
freq_high = 30
freq_low = 0.5
do_subsamp = False
do_aug = False

# Set working directory
os.chdir('/Users/alikavoosi/Desktop/DREEM/DATA')

# Training fold 14 specifically
fold = 14
print('##################################################################')
print(f'Training Regular Model for Fold {fold} with Optimized TFLite Output')
print('##################################################################')

# Model save paths
best_model_file = f'/Users/alikavoosi/Downloads/sleeps/cnn_model_fold{fold}_optimized.keras'
best_model_file_tflite = f'/Users/alikavoosi/Downloads/sleeps/cnn_model_fold{fold}_optimized_quantized.tflite'

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=best_model_file, 
    monitor='val_loss', 
    mode='min',
    save_best_only=True,
    save_freq="epoch"
)

# Get fold indices
train_inds, test_inds, val_inds = get_fold_indices(fold)

# Try to load existing indices, otherwise save new ones
try:
    train_inds = np.load(f'train_ind_fold{fold}.npy')
except:
    np.save(f'train_ind_fold{fold}.npy', train_inds)
try:
    test_inds = np.load(f'test_ind_fold{fold}.npy')
except:
    np.save(f'test_ind_fold{fold}.npy', test_inds)
try:
    val_inds = np.load(f'val_ind_fold{fold}.npy')
except:
    np.save(f'val_ind_fold{fold}.npy', val_inds)

print(f'Training indices: {train_inds}')
print(f'Validation indices: {val_inds}')
print(f'Test indices: {test_inds}')

# Prepare training data
print('\nLoading and processing training data...')
lens = []
x_train = []
x_train_mean = []
x_train_std = []
y_train = []
train_epoch_ind = []

for i in train_inds:
    x, hyp = extract_data(i)
    x = butter_bandpass_filter(x, freq_low, freq_high, 250)
    x_r = resample(x, int(len(x)*fs_d/250))
    inds = np.arange(0, len(x_r), 30*fs_d)
    epochs = np.reshape(x_r, (len(inds), 1, epoch_len, 1))
    lens.append(len(epochs))
    
    for num, epoch in enumerate(epochs):
        if do_aug:
            if np.random.random() > 0.3:
                f = np.random.random() + 5
                amp = (np.random.random() + 1) * 4
                noise = amp * np.sin(2*np.pi*np.linspace(0, 30, 3000)*f)
                noise += amp/2 * np.sin(2*np.pi*np.linspace(0, 30, 3000)*2*f)
                noise = np.reshape(noise, (1, 3000, 1))
                epochs[num] += noise
        
        mean = np.mean(epoch)
        std = np.std(epoch)
        epochs[num] = (epoch - mean) / std
        x_train_mean.append(mean)
        x_train_std.append(std)
    
    epoch_inds = pos_var_v2(len(epochs))
    train_epoch_ind.append(epoch_inds.reshape((len(epoch_inds), 1)))
    
    x_train.append(epochs)
    y_train.append(np.array(hyp).reshape((len(hyp), 1)))

x_train = np.vstack(x_train)
y_train = np.vstack(y_train)
train_epoch_ind = np.vstack(train_epoch_ind)

print(f'Training data shape: {x_train.shape}')
print(f'Training labels shape: {y_train.shape}')

# Prepare validation data
print('\nLoading and processing validation data...')
x_val = []
x_val_mean = []
x_val_std = []
y_val = []
val_epoch_ind = []

for i in val_inds:
    x, hyp = extract_data(i)
    x = butter_bandpass_filter(x, freq_low, freq_high, 250)
    x_r = resample(x, int(len(x)*fs_d/250))
    inds = np.arange(0, len(x_r), 30*fs_d)
    epochs = np.reshape(x_r, (len(inds), 1, epoch_len, 1))
    lens.append(len(epochs))
    
    for num, epoch in enumerate(epochs):
        mean = np.mean(epoch)
        std = np.std(epoch)
        epochs[num] = (epoch - mean) / std
        x_val_mean.append(mean)
        x_val_std.append(std)
    
    epoch_inds = pos_var_v2(len(epochs))
    val_epoch_ind.append(epoch_inds.reshape((len(epoch_inds), 1)))
    
    x_val.append(epochs)
    y_val.append(np.array(hyp).reshape((len(hyp), 1)))

x_val = np.vstack(x_val)
y_val = np.vstack(y_val)
val_epoch_ind = np.vstack(val_epoch_ind)

print(f'Validation data shape: {x_val.shape}')
print(f'Validation labels shape: {y_val.shape}')

# Prepare test data
print('\nLoading and processing test data...')
x_test = []
x_test_mean = []
x_test_std = []
y_test = []
test_epoch_ind = []

for i in test_inds:
    x, hyp = extract_data(i)
    x = butter_bandpass_filter(x, freq_low, freq_high, 250)
    x_r = resample(x, int(len(x)*fs_d/250))
    inds = np.arange(0, len(x_r), 30*fs_d)
    epochs = np.reshape(x_r, (len(inds), 1, epoch_len, 1))
    lens.append(len(epochs))
    
    for num, epoch in enumerate(epochs):
        mean = np.mean(epoch)
        std = np.std(epoch)
        epochs[num] = (epoch - mean) / std
        x_test_mean.append(mean)
        x_test_std.append(std)
    
    epoch_inds = pos_var_v2(len(epochs))
    test_epoch_ind.append(epoch_inds.reshape((len(epoch_inds), 1)))
    
    x_test.append(epochs)
    y_test.append(np.array(hyp).reshape((len(hyp), 1)))

x_test = np.vstack(x_test)
y_test = np.vstack(y_test)
test_epoch_ind = np.vstack(test_epoch_ind)

print(f'Test data shape: {x_test.shape}')
print(f'Test labels shape: {y_test.shape}')

# Shuffle training data
x_train, y_train, train_epoch_ind, x_train_mean, x_train_std = shuffle(
    x_train, y_train, train_epoch_ind, x_train_mean, x_train_std
)

# Set up class weights
class_weight = {}
total = len(y_train)
for cl in range(1, 5):
    no = len([i for i in y_train if i == cl])
    weight = (1/no) * (total/5)
    class_weight[cl] = 1
class_weight[0] = 2

print(f'Class weights: {class_weight}')

# Create base model
print('\nCreating base model...')
optimizer = tf.keras.optimizers.Adam(learning_rate=10e-4)
model = separable_resnet_incind((1, epoch_len, 1), 5, y_train=y_train, bias=False)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print(f'Base model created with {model.count_params():,} parameters')

# Train the model
print('\nTraining model...')
history = model.fit(
    [x_train, train_epoch_ind], 
    to_categorical(y_train), 
    batch_size=32, 
    epochs=6,
    validation_data=([x_val, val_epoch_ind], to_categorical(y_val)), 
    callbacks=[checkpoint_callback],
    verbose=1
)

# Load best model
print('\nLoading best trained model...')
model = tf.keras.models.load_model(best_model_file, custom_objects={'ExponentialLayer': ExponentialLayer})

# Evaluate on test set
print('\n####################  Test Results  ####################')
test_loss, test_acc = model.evaluate(
    [x_test, test_epoch_ind], 
    to_categorical(y_test, num_classes=5),
    verbose=1
)

print(f'Test Accuracy: {test_acc:.4f}')
print(f'Test Loss: {test_loss:.4f}')

# Convert to TFLite with advanced quantization
print('\nConverting to optimized TFLite...')

# Create inference model that forces training=False for batch norm
class InferenceModel(tf.keras.Model):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    
    @tf.function
    def call(self, inputs):
        return self.base_model(inputs, training=False)

inference_model = InferenceModel(model)

# Build the model with correct shapes
input_shapes = model.input_shape
if isinstance(input_shapes, list):
    build_shapes = []
    for shape in input_shapes:
        build_shape = tuple([1 if dim is None else dim for dim in shape])
        build_shapes.append(build_shape)
    inference_model.build(build_shapes)
    
    # Create input signatures
    input_signature = []
    for shape in input_shapes:
        spec_shape = tuple([1 if dim is None else dim for dim in shape])
        input_signature.append(tf.TensorSpec(shape=spec_shape, dtype=tf.float32))
    
    @tf.function(input_signature=[input_signature])
    def model_predict(inputs):
        return inference_model(inputs)

# Get concrete function
concrete_func = model_predict.get_concrete_function()

# Convert with advanced quantization
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Use representative dataset for better quantization
def representative_dataset():
    for i in range(min(500, len(x_train))):  # Use up to 500 samples
        idx = np.random.randint(0, len(x_train))
        yield [
            x_train[idx:idx+1].astype(np.float32),
            train_epoch_ind[idx:idx+1].astype(np.float32)
        ]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

# Convert to TFLite
tflite_model = converter.convert()

# Save TFLite model
with open(best_model_file_tflite, 'wb') as f:
    f.write(tflite_model)

print(f'✅ Optimized TFLite model saved: {best_model_file_tflite}')

# Size comparison
keras_size = os.path.getsize(best_model_file) / 1024 / 1024
tflite_size = len(tflite_model) / 1024 / 1024
print(f'\nModel Size Comparison:')
print(f'  Keras model: {keras_size:.2f} MB')
print(f'  TFLite model: {tflite_size:.2f} MB')
print(f'  Size reduction: {(1 - tflite_size/keras_size)*100:.1f}%')

print('\n✅ Training and quantization completed successfully!')
print(f'✅ Models saved:')
print(f'   - Keras: {best_model_file}')
print(f'   - TFLite: {best_model_file_tflite}')

# Test the TFLite model on a few samples
print('\nTesting TFLite model...')
interpreter = tf.lite.Interpreter(model_path=best_model_file_tflite)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f'TFLite input details: {input_details}')
print(f'TFLite output details: {output_details}')

# Test on a few samples
n_test_samples = 5
keras_predictions = []
tflite_predictions = []

for i in range(n_test_samples):
    # Keras prediction
    keras_pred = model.predict([x_test[i:i+1], test_epoch_ind[i:i+1]], verbose=0)
    keras_predictions.append(keras_pred[0])
    
    # TFLite prediction
    interpreter.set_tensor(input_details[0]['index'], x_test[i:i+1].astype(np.float32))
    interpreter.set_tensor(input_details[1]['index'], test_epoch_ind[i:i+1].astype(np.float32))
    interpreter.invoke()
    tflite_pred = interpreter.get_tensor(output_details[0]['index'])
    tflite_predictions.append(tflite_pred[0])

# Compare predictions
print('\nKeras vs TFLite Predictions Comparison:')
for i in range(n_test_samples):
    keras_class = np.argmax(keras_predictions[i])
    tflite_class = np.argmax(tflite_predictions[i])
    mse = np.mean((keras_predictions[i] - tflite_predictions[i]) ** 2)
    print(f'Sample {i+1}: Keras={keras_class}, TFLite={tflite_class}, MSE={mse:.6f}')

print('\n🎯 Training and optimization complete!')

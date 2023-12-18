
# Import necessary modules and functions
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
from magenta import music as mm
import midi_io
import configs
from trained_model import TrainedModel
import numpy as np
import tensorflow.compat.v1 as tf
import pretty_midi
import midi_io_FIXED
import pickle

# Define flags for command-line arguments
flags = tf.app.flags
logging = tf.logging
FLAGS = flags.FLAGS

# Define configuration parameters for the model
config = "cat-mel_2bar_small"

# Initialize the trained model with the specified config
model = TrainedModel(
    configs.CONFIG_MAP[config], batch_size=2,
    checkpoint_dir_or_path="cat-mel_2bar_small/model.ckpt")

# Load pre-processed data from pickle files
meas = pickle.load(open("pickles/meas16.pcl", "rb"))
inds = pickle.load(open("pickles/inds16.pcl", "rb"))
meas2 = [meas[i] for i in inds]
graph_vecs = []

# Process each graph in the dataset
for (graph_ind, graph) in enumerate(meas2):
    print(graph_ind)
    graph_vecs.append([])
    print(len(meas[graph_ind]))
    
    # Process each bar in the graph
    for (bar_ind, bar) in enumerate(meas[graph_ind]):
        bar = [(a[0], a[1]) for a in bar]
        input_1 = midi_io.midi_file_to_note_sequence(bar)
        mm.sequence_proto_to_midi_file(input_1, "tmpmids/tmp.mid")
        input_1 = midi_io_FIXED.midi_file_to_note_sequence("tmpmids/tmp.mid")

        # Encode the input sequence and store the result
        try:
            _, mu, _ = model.encode([input_1, input_1], False)
            graph_vecs[-1].append(mu)
        except:
            print("error")
            graph_vecs[-1].append(np.random.normal(size=256))
        
    # Save the processed vectors periodically to avoid data loss
    if graph_ind % 50 == 0 or graph_ind == len(meas) - 1: 
        pickle.dump(graph_vecs, open("pickles/analyzedmagents.pcl", "wb"))

# Initialize an empty list for the second pass of processing
graph_vecs = []

# Process each graph in the dataset
for (graph_ind, graph) in enumerate(meas):
    print(graph_ind)
    
    # Process each bar in the graph
    for (bar_ind, bar) in enumerate(meas[graph_ind]):
        bar = [(a[0], a[1]) for a in bar]
        input_1 = midi_io.midi_file_to_note_sequence(bar)
        mm.sequence_proto_to_midi_file(input_1, "tmpmids/tmp.mid")
        input_1 = midi_io_FIXED.midi_file_to_note_sequence("tmpmids/tmp.mid")

        # Encode the input sequence and store the result
        try:
            _, mu, _ = model.encode([input_1, input_1], False)
            graph_vecs.append(mu)
        except:
            print("error")
            graph_vecs.append(np.random.normal(size=256))
        
    # Save the processed vectors periodically to avoid data loss
    if graph_ind % 50 == 0 or graph_ind == len(meas) - 1:
        pickle.dump(graph_vecs, open("pickles/analyzedmagents2.pcl", "wb"))


# Import necessary libraries and modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import sys
import time

from magenta import music as mm
import midi_io
import configs
import numpy as np
import tensorflow.compat.v1 as tf
import pretty_midi
import pickle
from music21 import *
import shutil
from magenta.models.music_vae import TrainedModel
import torch
import operator

# Function to renormalize a dictionary of probabilities so that their sum is 1
def renormalizeProbs(probdict):
    sumps = sum(probdict.values())
    for k in probdict.keys():
        probdict[k] = (probdict[k] + 0.0) / sumps

# Convert a dictionary of probabilities to a single choice based on those probabilities
def probDictToChoice(pdict):
    renormalizeProbs(pdict)
    ps = {}
    prev = 0
    for (k,v) in filter(lambda i: i[1] > 0.0, pdict.items()):
        ps[k] = prev  + v
        prev = ps[k]
    r = random.uniform(0.0, 1.0)
    psitems = sorted(ps.items(), key = operator.itemgetter(1))
    for k,v in psitems:
        if r <= v:
            return k
    return ps.keys()[-1]

# Define TensorFlow flags and logging
flags = tf.app.flags
logging = tf.logging
FLAGS = flags.FLAGS

# Spherical linear interpolation function
def _slerp(p0, p1, t):
  """Spherical linear interpolation."""
  omega = np.arccos(
      np.dot(np.squeeze(p0/np.linalg.norm(p0)),
             np.squeeze(p1/np.linalg.norm(p1))))
  so = np.sin(omega)
  return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1

# Configuration for the MusicVAE model
config = "cat-mel_2bar_small"
model = TrainedModel(
            configs.CONFIG_MAP[config], batch_size=2,
            checkpoint_dir_or_path="cat-mel_2bar_small/model.ckpt")

# Load preprocessed data
reals = pickle.load(open("pickles/recons.pcl", "rb"))

# Initialize lists to store measures and magenta sequences
all_meas = []
all_magents = []

# Iterate through the preprocessed data and generate music sequences
for (q_, i) in enumerate(reals):
    meas = []
    index = 0
    for (k, val) in enumerate(i):
        while True:
            index += 1
            z = np.array([val, val])
            results = model.decode(
            length=16,
            z=z,
            temperature=1.0)
            mm.sequence_proto_to_midi_file(results[0], "tmpmids/" + str(q_) + "-" + str(k) + ".mid")
            try:
                a = converter.parse("tmpmids/" + str(q_) + "-" + str(k) + ".mid")
                for part in a:
                    for val2 in list(part):
                        if type(val2) == note.Note:
                            meas.append((val2.pitch.midi, val2.quarterLength))
                        elif type(val2) == note.Rest:
                            meas.append((0, val2.quarterLength))
                all_meas.append(meas)
                all_magents.append(z[0,:])
                break
            except:
                if index > 100:
                    break
         
    # Create a music21 stream and write the generated sequence to a MIDI file
    a = stream.Score()
    onset = 0.0
    for (pit, dur) in meas:
        if pit > 0:
            a.insert(onset, note.Note(pit, quarterLength = dur))
        else:
            a.insert(onset, note.Rest(quarterLength = dur))
        onset += dur
    a.write(fmt="mid", fp="generatedMids/" + str(q_) + "-vae.mid")

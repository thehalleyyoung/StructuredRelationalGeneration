
# Import necessary libraries and modules.
import os
import random
import sys
import time
import numpy as np
import tensorflow.compat.v1 as tf
import pretty_midi
import pickle
from music21 import *
import shutil
from magenta.models.music_vae import TrainedModel
import torch
import operator

# Function to renormalize a dictionary of probabilities so that their sum equals 1.
def renormalizeProbs(probdict):
    sumps = sum(probdict.values())
    for k in probdict.keys():
        probdict[k] = (probdict[k] + 0.0) / sumps

# Convert a dictionary of probabilities into a random choice based on those probabilities.
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

# Initialize TensorFlow flags and logging.
flags = tf.app.flags
logging = tf.logging
FLAGS = flags.FLAGS

# Function for spherical linear interpolation.
def _slerp(p0, p1, t):
  """Spherical linear interpolation."""
  omega = np.arccos(
      np.dot(np.squeeze(p0/np.linalg.norm(p0)),
             np.squeeze(p1/np.linalg.norm(p1))))
  so = np.sin(omega)
  return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1

# Set the configuration for the MusicVAE model.
config = "cat-mel_2bar_small"
model = TrainedModel(
            configs.CONFIG_MAP[config], batch_size=2,
            checkpoint_dir_or_path="cat-mel_2bar_small/model.ckpt")

# Main loop to generate and process MIDI files.
for (q_, i) in enumerate(range(200)):
    # Initialize variables to keep track of various musical patterns.
    rhy_tups = {}
    index = 0
    has_weird = 0
    has_long = False
    has_short = 0
    has_half = False
    has_only_quarter_half = 0
    has_sixteenth = 0
    # Inner loop to generate individual sequences.
    for (k, val) in enumerate(range(16)):
        # Attempt to generate a sequence that satisfies certain musical criteria.
        while True:
            index += 1
            # Generate a random latent vector and decode it into a music sequence.
            z = np.array([
                _slerp(np.random.normal(size=256), np.random.normal(size=256), t) for t in np.linspace(0, 1, 2)])
            results = model.decode(
                length=16,
                z=z,
                temperature=0.75)

            # Save the generated sequence to a MIDI file.
            mm.sequence_proto_to_midi_file(results[0], "referenceMids/" + str(q_) + "-" + str(k) + ".mid")
            try:
                # Parse the MIDI file and apply various checks to ensure musicality.
                a = converter.parse("referenceMids/" + str(q_) + "-" + str(k) + ".mid")
                

                a = converter.parse("referenceMids/" + str(q_) + "-" + str(k) + ".mid")
                    if any([type(q) == note.Rest for q in list(list(a)[0])]):
                        assert False

                    pits = list([q.pitch.midi for q in list(list(a)[0]) if type(q) == note.Note])
                    for pit_ind in range(len(pits) - 2):
                        if pits[pit_ind] == pits[pit_ind + 1] and pits[pit_ind] == pits[pit_ind + 2]:
                            assert False
                    



                    if len(pits) >= 6:
                        if has_long:
                            assert False

                    if len(pits) <= 3:
                        if has_short >= 2 or len(pits) < 3:
                            assert False
                    rhy_tup = list([q.quarterLength for q in list(list(a)[0]) if type(q) == note.Note])
                    onset_tup = [sum(rhy_tup[:q]) for q in range(len(rhy_tup))]


                    if any([k == 2.0 for k in rhy_tup]) and has_half:
                        assert False
                    if all([k == 1.0 or k == 2.0 for k in rhy_tup]):
                        if has_only_quarter_half >= 2:
                            assert False
                    if any([k % 0.5 == 0.25 for k in rhy_tup]):
                        if has_sixteenth >= 2:
                            assert False

                    
                    if any([onset_tup[q] % 0.5 == 0.25 and onset_tup[q + 1] % 0.5 == 0.25 for q in range(len(onset_tup) - 1)]):
                        assert False
                    if len(rhy_tup) > 8:
                        assert False
                    if any([k >= 3.0 for k in rhy_tup]):
                        assert(False)
                    if sum(rhy_tup) != 4.0:
                        assert(False)
                    if sum([onset_tup[q] % 0.5 == 0.25 for q in range(len(onset_tup))]) > 1:
                        if has_weird >= 1:
                            assert False
                    if tuple(rhy_tup) in rhy_tups:
                        if rhy_tups[tuple(rhy_tup)] >= 2 or len(rhy_tup) <= 3:
                            assert False
                        else:
                            rhy_tups[tuple(rhy_tup)] += 1
                    else:
                        rhy_tups[tuple(rhy_tup)] = 1

                    if all([k == 1.0 or k == 2.0 for k in rhy_tup]):
                        has_only_quarter_half += 1
                    if len(pits) <= 3:
                        has_short += 1
                    if sum([onset_tup[q] % 0.5 == 0.25 for q in range(len(onset_tup))]) > 1:
                        has_weird += 1
                    if len(pits) >= 6:
                        has_long = True
                    if any([k == 2.0 for k in rhy_tup]):
                        has_half = True
                    if any([k % 0.5 == 0.25 for k in rhy_tup]):
                        has_sixteenth += 1
                    print("success")

                    break

                print("success")
                break
            except:
                # If too many attempts fail, give up on this sequence.
                if index > 300:
                    print("failed")
                    break
    
    # Do basic conversion to Music21 format
    s = stream.Score()
    onset = 0
    bars = []
    key_a = 0
    for k in range(16):
        a = [(m) for m in list(converter.parse("referencemids/" + str(q_) + "-" + str(k) + ".mid")) if type(m) == stream.Part]

        for part in a:
            key_b = part.analyze("key")
            key_probs = {key_a:0.8, (key_a + 7) % 12:0.3, (key_a + 5) % 12: 0.3}
            key_chosen = probDictToChoice(key_probs)
            i_ = interval.Interval(key_b.tonic, pitch.Pitch(key_chosen))
            part = part.transpose(i_)
            key_a = part.analyze("key").tonic.midi
            for val in list(part):
                    if type(val) == note.Note:
                        s.insert(onset, note.Note(val.pitch.midi, quarterLength = val.quarterLength))
                        onset += val.quarterLength
                        bars.append((val.pitch.midi, val.quarterLength))
                    elif type(val) == note.Rest:
                        s.insert(onset, note.Rest(val.midi, quarterLength = val.quarterLength))
                        onset += val.quarterLength
    
    # convert to key of C
    k = s.analyze('key')
    i_ = interval.Interval(k.tonic, pitch.Pitch('C'))
    s = s.transpose(i_)
    # Write the final sequence to a MIDI file.
    s.write(fmt="mid", fp = "referenceMids/" + str(q_) + ".mid")

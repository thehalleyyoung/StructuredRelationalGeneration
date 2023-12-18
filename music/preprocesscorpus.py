
# Import required libraries for music processing, file management, and data serialization
from music21 import *
import os
import pickle
import copy
import glob
import sys
from shutil import copy, copyfile

# Define the folder where the music XML files are stored
folder = "essen"
# Initialize a counter for the total number of songs processed
tot_songs = 0
# Retrieve the list of XML filenames from the specified folder
files = [i for i in list(os.walk("./" + folder))[0][2] if i.endswith("xml")]

# Initialize an empty list to store processed songs
songs = []
# Initialize an index for tracking the number of measures processed
index = 0
# Initialize an empty list to store individual bars (measures)
bars = []
# Define the number of measures we want to process per song
n_meas = 16

# Loop through each file in the list of XML files
for (file_ind, file) in enumerate(files):
    # Initialize an empty list to store notes for the current file
    notes = []
    # Parse the current file with music21 to get a music21 stream object
    a = converter.parse(folder + "/" + file)
    # Iterate through the elements in the music21 stream object
    for val in list(a):
        # Check if the element is a Part (a single instrument's line in the score)
        if type(val) == stream.Part:
            # Iterate through the elements in the Part
            for (el_ind, el) in enumerate(list(val)):
                # Check if the element is a Measure (a single bar of music)
                if type(el) == stream.Measure:
                    # Initialize an empty list to store measure information
                    meas = []
                    # Iterate through the elements in the Measure
                    for (el2_ind, el2) in enumerate(el):
                        # Check if the element is a Rest and store relevant information
                        if type(el2) == note.Rest:
                            meas.append((0, el2.quarterLength, 0, None, None))
                        # Check if the element is a Note and store relevant information
                        elif type(el2) == note.Note:
                            meas.append((el2.pitch.midi, el2.quarterLength, el2.pitch.octave, el2.tie, el2.duration.tuplets))
                    # Check if the total duration of the measure is 4 beats (a full measure in common time)
                    if sum([q[1] for q in meas]) == 4.0:
                        # Increment the index since we've processed another measure
                        index += 1
                        # Append the processed measure to the notes list
                        notes.append(meas)
                        # Append the processed measure to the bars list
                        bars.append(meas)
    
    # Check if we have processed at least the defined number of measures for the current song
    if len(notes) >= n_meas:
        # Increment the total number of songs processed
        tot_songs += 1
        # Append the first n_meas measures of the current song to the songs list
        songs.append(notes[:n_meas])

# Serialize the bars list to a pickle file for later use or analysis
pickle.dump(bars, open("pickles/allbars.pcl", "wb"))

# Serialize the songs list to a pickle file for later use or analysis
pickle.dump(songs, open("pickles/meas16.pcl", "wb"))

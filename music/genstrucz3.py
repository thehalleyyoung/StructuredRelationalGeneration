
from z3 import *
import pickle
from music21 import *
import time
import random
import sys
import symmetries
import numpy as np
import copy

# Music generation based on generated structure using Z3 SMT solver

# Load reference features and measures from serialized files
features = pickle.load(open("../reference_features.pcl", "rb"))
ref_measures = pickle.load(open("../allrefmeasures.pcl", "rb"))

# Import all functions from the symmetries module
from inspect import getmembers, isfunction
functions_list = [o for o in getmembers(symmetries) if isfunction(o[1])]
functions_name_list = [i[1] for i in functions_list]

# Function to check if two pitches are the same or a half-step apart in modulo 12 space
def Mod12One(a, b):
    return Or(And(a == 11, b == 0), And(b == 11, a == 0), b - a == 1, a - b == 1, a - b == 0)

# Preprocess features and reference measures
for i in range(len(features)):
    for j in range(len(features[i])):
        # Convert durations to integer values representing sixteenth notes
        durs_ref = [int(k[1]*4) for k in ref_measures[i][j]]
        # Check for presence of sixteenth notes
        has_sixteenth = any([k % 4 == 1 or k % 4 == 3 for k in durs_ref])
        # Calculate cumulative durations to check for syncopation
        cum_durs = [sum(durs_ref[:k]) for k in range(len(durs_ref))]
        has_sync = any([cum_durs[k + 1] % 4 in [1,3] and cum_durs[k] % 4 in [1,3] for k in range(len(cum_durs) - 1)])
        # Convert feature values to boolean
        for k in features[i][j].keys():
            if features[i][j][k] == 1:
                features[i][j][k] = True
            elif features[i][j][k] == 0:
                features[i][j][k] = False
            else:
                features[i][j][k] = bool(features[i][j][k])
                print(features[i][j][k])
        # Randomly assign rhythm similarity based on presence of sixteenth notes
        if has_sixteenth or random.uniform(0,1) < 0.5:
            features[i][j]["hasSameRhythm"] = True
        # Assign syncopation feature
        features[i][j]["hasSyncopation"] = has_sync

# Define XOR functions for feature matching
def XOR(a, b):
    return If(And(Not(a), Not(b)), 1, If(And(a, b), 6, 0))

def XOR2(a, b):
    return If(And(Not(a), Not(b)), 3, 0)

# Generate a constrained melody based on given features and a prototype melody
def genConstrained(feats, proto, prev_pit):
    # Initialize optimization problem
    s = Optimize()
    
    # Define variables for pitches and durations
    pits = Array("pits", IntSort(), IntSort())
    length = Int("pitlength")
    durs = [Int("dur " + str(i)) for i in range(9)]
    cum_durs = Array("dur ", IntSort(), IntSort())
    reward = Int("cost")
    
    # Add constraints and reward functions for various musical features
    # These include pitch intervals, rhythm, contour, symmetry, and more
    
    	s.add(And(pits[0] - prev_pit <= 12, pits[0] - prev_pit >= -12))
	s.add(length <= 9)
	interval_reward = Sum([If(And(i < length, And(pits[i] - pits[i - 1] <= 4, pits[i] - pits[i - 1] > -4, pits[i] - pits[i - 1] != 6, pits[i] - pits[i - 1] != -6)), 1, 0) for i in range(9)])

	pit_reward = If(And([Or([pits[i] % 12 == k for k in [0,2,4,5,7,9,11]]) for i in range(9)]), 4, 0) + If(And(pits[0] - prev_pit <= 5, pits[0] - prev_pit >= -5), 4, 0) + If(And(pits[0] - pits[length - 1] <= 5, pits[0] - pits[length - 1] >= -5), 4, 0)

	a1 = XOR(feats["hasSamePitches"], And(And([Mod12One(pits[i] % 12, proto_pits[i] % 12) for i in range(proto_len)]), length == proto_len))
	
	a2 = (XOR(feats["hasSamePitchesPrefix"], And([Mod12One(pits[i] % 12, proto_pits[i] % 12) for i in range(min(proto_len, 3))])))

	a3 = XOR(feats["hasSamePitchesSuffix"], Or([And(length == k, And([Mod12One(pits[k - j] % 12, proto_pits[proto_len - j] % 12) for j in range(1, min(proto_len, 4))])) for k in range(2,9)]))

	a4 = XOR(feats["hasAddOnePitchSymmetry"], And([Or(Mod12One(pits[i] % 12, proto_pits[i] % 12), Mod12One(pits[i] % 12, proto_pits[i + 1] % 12), Mod12One(pits[i] % 12, proto_pits[i - 1] % 12)) for i in range(1, proto_len - 1)]))

	a5 = XOR(feats["hasIntervalSymmetry"], Sum([If(Or(pits[i] - pits[i - 1] - proto_ints[i - 1] > 2, pits[i] - pits[i - 1] - proto_ints[i - 1] < -2), 1, 0) for i in range(1, len(proto_ints))]) > proto_len - 2)

	if proto_len > 3:
		a6 = XOR(feats["hasIntervalPrefix"],  Sum([If(Or(pits[i] - pits[i - 1] - proto_ints[i - 1] > 2, pits[i] - pits[i - 1] - proto_ints[i - 1] < -2), 1, 0) for i in range(1, min(proto_len, 4))]) == 3)

		a7 = XOR(feats["hasIntervalSuffix"], Sum([If(Or(pits[i] - pits[i - 1] - proto_ints[i - 1] > 2, pits[i] - pits[i - 1] - proto_ints[i - 1] < -2), 1, 0) for i in range(proto_len - 4, proto_len - 1)]) == 3)

	else:
		a7 = 0

		a6 = 0

	a8 = XOR(feats["hasChangeOnePitchSymmetry"], Sum([If(Not(Mod12One(pits[i] % 12,  proto_pits[i] % 12)), 1, 0) for i in range(proto_len)]) <= 1)

	if proto_len >= 4:

		a9 = XOR(feats["hasNonTrivialPrefix"], And([And(Mod12One(pits[i] % 12, proto_pits[i] % 12), durs[i] == proto_durs[i]) for i in range(3)]))
	else:
		a9 = 0

	a10 = XOR(feats["hasNonTrivialSuffix"], And(proto_len >= 3, And([And(Mod12One(pits[length - i] % 12, proto_pits[proto_len - i] % 12), durs[i] == proto_durs[i]) for i in range(1,min(proto_len, 4))])))

	a11 = XOR(feats["hasAddOneRhythmSymmetry"], And(And(proto_len - length <= 1, proto_len - length >= -1), And([Or(cum_durs[i] == proto_cum[i - 1], cum_durs[i] == proto_cum[i + 1], cum_durs[i] == proto_cum[i]) for i in range(proto_len - 1)])))

	a12 = XOR(feats["hasSubsetRhythm"], Or(And([Or(i >= length, Or([cum_durs[i] == k for k in proto_cum])) for i in range(9)]), And([Or([proto_cum[i] == cum_durs[k] for k in range(9)]) for i in range(proto_len)])))

	a13 = XOR(feats["hasSameRhythm"], And([durs[i] == proto_durs[i] for i in range(proto_len)]))

	a14 = 2*XOR(feat_syncopation, Or([And(i < length, And(Or(cum_durs[i] % 4 == 1, cum_durs[i] % 4 == 3), Or(cum_durs[i + 1] % 4 == 1, cum_durs[i + 1] % 4 == 3))) for i in range(9)]))

	a15 = 10*XOR2(has_sixteenth, Or([And(i < length, Or(durs[i] % 4 == 1, durs[i] % 4 == 3)) for i in range(9)]))

	a16 = XOR(feats["hasSameContour"], And(And([durs[i] == proto_durs[i] for i in range(proto_len)]), Sum([If(Or(pits[i] - pits[i - 1] - proto_ints[i - 1] > 3, pits[i] - pits[i - 1] - proto_ints[i - 1] < -3), 1, 0) for i in range(proto_len)]) < 1))

	if proto_len > 3:
		a17 = XOR(feats["hasSameContourPrefix"], And(And([durs[i] == proto_durs[i] for i in range(proto_len)]), Sum([If(Or(pits[i] - pits[i - 1] - proto_ints[i - 1] > 3, pits[i] - pits[i - 1] - proto_ints[i - 1] < -3), 1, 0) for i in range(3)]) < 1))
		a18 = XOR(feats["hasSameContourSuffix"], And(And([durs[i] == proto_durs[i] for i in range(proto_len)]), Sum([If(Or(pits[i] - pits[i - 1] - proto_ints[i - 1] > 3, pits[i] - pits[i - 1] - proto_ints[i - 1] < -3), 1, 0) for i in range(proto_len - 3, proto_len)]) < 1))

	else:
		a17 = 0
		a18 = 0


	threepeat = Or([And(i < length - 2, pits[i] == pits[i + 1], pits[i] == pits[i + 2]) for i in range(9)])

	last_note_reward = If(And(pits[length - 1] >= 12, pits[length - 1] <= 24), 3, 0) + If(threepeat, -3, 0)

        # establish basic formal constraints on what pitches/durs can be
    	for i in range(9):
		s.add(pits[i] > 0)
		s.add(pits[i] < 36)
		if i == 0:
			s.add(pits[i] - prev_pit <= 6)
			s.add(cum_durs[i] == 0)
		else:
			s.add(pits[i] - pits[i - 1] <= 7)
			s.add(pits[i] - pits[i - 1] >= -7)
			s.add(cum_durs[i] == cum_durs[i - 1] + durs[i - 1])
		s.add(durs[i] >= 1)
		s.add(durs[i] <= 16)
		s.add(cum_durs[length] == 16)
		s.add(Implies(pits[i] - pits[i - 1] > 7, Or([pits[i + 1] == pits[i] + k for k in [-1,-2,-3]])))
		s.add(Implies(pits[i] - pits[i - 1] < -7, Or([pits[i + 1] == pits[i] + k for k in [1,2,3]])) )
	for i in range(9 - 3):
		s.add(Not(And(pits[i] == pits[i + 1], pits[i] == pits[i + 2], pits[i] == pits[i + 3], i < length - 3)))
	s.add(length > 1)
	s.add(length <= 16)
	s.add(Sum([If(i < length, durs[i], 0) for i in range(9)]) == 16)

    # Set timeout for the optimization solver
    s.set("timeout", 120*1000)
    
    # Solve the optimization problem
    s.check()
    s.model()
    
    # Extract the solution as pitches and durations
    length = int(str(s.model().eval(length)))
    pits = [int(str(s.model().eval(pits[i]))) for i in range(length)]
    durs = [int(str(s.model().eval(durs[i]))) for i in range(length)]
    
    # Return the generated melody as a tuple of pitches and durations
    return (pits, durs)

# Main loop for generating music based on features and reference measures
false_negs = 0
false_pos = 0
mean_sym = []
mean_supposed_sym = []
start_ind = int(sys.argv[1]) if len(sys.argv) > 1 else 0
for z in range(start_ind, start_ind + 500):
    # Generate melodies for each measure based on previously generated features and reference measures
    start_time = time.time()
    prev_pits = [12]
    prev_durs = []
    prev_features = copy.copy(features[z])
    for i in range(len(features[z])):
            for j_ in range(6):
                    features[z][i][random.choice(functions_list)] = True
            measure_start = time.time()
            # get constrained definition for one measure
            pits, durs = genConstrained(features[z][i], ref_measures[z][i], prev_pits[-1]) 
            prev_pits.extend(pits)
            prev_durs.extend(durs)
            prev_pit = prev_pits[-1]

            pits_durs = [(pits[k], durs[k]/4) for k in range(len(pits))]
            tot_sym = 0
            print(ref_measures[z][i], pits_durs)
            for (name, func) in functions_list:
                    if name in features[z][i].keys():
                            has_sym_name = func(ref_measures[z][i], pits_durs)
                            ref_sym = features[z][i][name]
                            if not ref_sym and has_sym_name:
                                    false_pos += 1
                            elif ref_sym and not has_sym_name:
                                    false_negs += 1
                            tot_sym += int(has_sym_name)

            # logging mean symmetries for efficacy reviewing
            mean_sym.append(tot_sym)
            mean_supposed_sym.append(sum([1 if k else 0 for k in features[z][i].values()]))
            print((false_pos, false_negs, np.mean(mean_sym), np.mean(mean_supposed_sym)))

            print("measure time: " + str(time.time() - measure_start))
    prev_pits = prev_pits[1:]

    # Write the generated melodies to MIDI files for listening
    s = stream.Score()
    onset = 0
    for i in range(len(prev_pits)):
        n = note.Note(prev_pits[i] + 48, quarterLength=prev_durs[i]/4)
        s.insert(onset, n)
        onset += prev_durs[i]/4
    s.write(fmt="midi", fp="../results/results_our_generated/our-model-z3-strong/ref" + str(z) + ".mid")

    #log time taken
    print("tot time: " + str(time.time() - start_time)) 


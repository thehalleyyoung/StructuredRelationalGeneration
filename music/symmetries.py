
import symhelpers

# Function to check if two sequences have the same pitches, ignoring octaves
def hasSamePitches(x, y):
    return symhelpers.mod12Same([i[0] for i in x], [i[0] for i in y])

# Function to check if the first three elements of two sequences have the same pitches
def hasSamePitchesPrefix(x, y):
    if len(x) < 3 or len(y) < 3:
        return len(x) < 3 and len(y) < 3
    return symhelpers.mod12Same([i[0] for i in x[:3]], [i[0] for i in y[:3]])

# Function to check if sequences are the same when one pitch is added to one of them
def hasAddOnePitchSymmetry(x, y):
    if abs(len(x) - len(y)) != 1:
        return False
    return any([[q[0] for q in x[:i]] + [q[0] for q in x[i+1:]] == y for i in range(len(x))]) \
        or any([[q[0] for q in y[:i]] + [q[0] for q in y[i+1:]] == x for i in range(len(y))])

# Function to check if the last three elements of two sequences have the same pitches
def hasSamePitchesSuffix(x, y):
    if len(x) < 3 or len(y) < 3:
        return len(x) < 3 and len(y) < 3
    return symhelpers.mod12Same([i[0] for i in x[-3:]], [i[0] for i in y[-3:]])

# Function to check if two sequences have similar intervals within a tolerance of 2 semitones
def hasIntervalSymmetry(x, y):
    if len(x) != len(y):
        return False
    intervals_x = [x[i][0] - x[i - 1][0] for i in range(1, len(x))]
    intervals_y = [y[i][0] - y[i - 1][0] for i in range(1, len(y))]
    return len(intervals_x) < 3 or sum([abs(intervals_x[k] - intervals_y[k]) <= 2 for k in range(len(intervals_x))]) >= len(intervals_x) - 2

# Function to check if the first three intervals of two sequences are similar
def hasIntervalPrefix(x, y):
    if len(x) < 4 or len(y) < 4:
        return len(x) < 4 and len(y) < 4
    intervals_x = [x[i][0] - x[i - 1][0] for i in range(1, len(x))]
    intervals_y = [y[i][0] - y[i - 1][0] for i in range(1, len(y))]
    return all([abs(intervals_x[k] - intervals_y[k]) <= 2 for k in range(3)])

# Function to check if the last three intervals of two sequences are similar
def hasIntervalSuffix(x, y):
    if len(x) < 4 or len(y) < 4:
        return len(x) < 4 and len(y) < 4
    intervals_x = [x[i][0] - x[i - 1][0] for i in range(1, len(x))]
    intervals_y = [y[i][0] - y[i - 1][0] for i in range(1, len(y))]
    return len(intervals_x) < 3 or (len(intervals_y) >= 3 and all([abs(intervals_x[-k] - intervals_y[-k]) <= 2 for k in range(1, 4)]))

# Function to check if sequences are the same when one pitch is changed in one of them
def hasChangeOnePitchSymmetry(x, y):
    if len(x) != len(y):
        return False
    return hasSamePitches(x, y) or any([symhelpers.mod12Same([q[0] for q in x[:i]] + [q[0] for q in x[i+1:]], [q[0] for q in y[:i]] + [q[0] for q in y[i+1:]]) for i in range(len(x))])

# Function to check if the first three elements have the same non-trivial attributes (pitch and another attribute)
def hasNonTrivialPrefix(x, y):
    return [i[1] for i in x[:3]] == [i[1] for i in y[:3]] and symhelpers.mod12Same([i[0] for i in x[:3]], [i[0] for i in y[:3]])

# Function to check if the last three elements have the same non-trivial attributes (pitch and another attribute)
def hasNonTrivialSuffix(x, y):
    return [i[1] for i in x[-3:]] == [i[1] for i in y[-3:]] and symhelpers.mod12Same([i[0] for i in x[-3:]], [i[0] for i in y[-3:]])

# Function to check if two sequences have the same rhythm
def hasSameRhythm(x, y):
    return [q[1] for q in x] == [q[1] for q in y]

# Function to check if sequences are the same when one rhythm value is added to one of them
def hasAddOneRhythmSymmetry(x, y):
    if abs(len(x) - len(y)) > 1:
        return False
    onsets_x = [sum([q[1] for q in x[:i]]) for i in range(len(x))]
    onsets_y = [sum([q[1] for q in y[:i]]) for i in range(len(y))]
    return hasSameRhythm(x, y) or any([onsets_x == onsets_y[:i] + onsets_y[i + 1:] for i in range(len(y))]) or any([onsets_y == onsets_x[:i] + onsets_x[i + 1:] for i in range(len(x))])

# Function to check if the rhythm of one sequence is a subset of the other's rhythm
def hasSubsetRhythm(x, y):
    onsets_x = [sum([q[1] for q in x[:i]]) for i in range(len(x))]
    onsets_y = [sum([q[1] for q in y[:i]]) for i in range(len(y))]
    return all([x in onsets_y for x in onsets_x]) or all([y in onsets_x for y in onsets_y])

# Function to check if two sequences have the same contour for pitch and duration
def hasSameContour(x, y):
    if symhelpers.veryDifferentLengths(x, y):
        return False
    contour_x_pitch = [len(set([k for k in [q[0] for q in x] if k < x[i][0]])) for i in range(len(x))]
    contour_x_duration = [len(set([k for k in [q[1] for q in x] if k < x[i][1]])) for i in range(len(x))]
    contour_y_pitch = [len(set([k for k in [q[0] for q in y] if k < y[i][0]])) for i in range(len(y))]
    contour_y_duration = [len(set([k for k in [q[1] for q in y] if k < y[i][1]])) for i in range(len(y))]
    return contour_x_pitch == contour_y_pitch and contour_x_duration == contour_y_duration

# Function to check if the first three elements of two sequences have the same contour
def hasSameContourPrefix(x, y):
    if len(x) < 3 or len(y) < 3 or symhelpers.veryDifferentLengths(x, y):
        return len(x) < 3 and len(y) < 3
    contour_x_pitch = [len(set([k for k in [q[0] for q in x] if k < x[i][0]])) for i in range(3)]
    contour_x_duration = [len(set([k for k in [q[1] for q in x] if k < x[i][1]])) for i in range(3)]
    contour_y_pitch = [len(set([k for k in [q[0] for q in y] if k < y[i][0]])) for i in range(3)]
    contour_y_duration = [len(set([k for k in [q[1] for q in y] if k < y[i][1]])) for i in range(3)]
    return contour_x_pitch[:3] == contour_y_pitch[:3] and contour_x_duration[:3] == contour_y_duration[:3]

# Function to check if the last three elements of two sequences have the same contour
def hasSameContourSuffix(x, y):
    if len(x) < 3 or len(y) < 3 or symhelpers.veryDifferentLengths(x, y):
        return len(x) < 3 and len(y) < 3
    contour_x_pitch = [len(set([k for k in [q[0] for q in x] if k < x[i][0]])) for i in range(len(x) - 3, len(x))]
    contour_x_duration = [len(set([k for k in [q[1] for q in x] if k < x[i][1]])) for i in range(len(x) - 3, len(x))]
    contour_y_pitch = [len(set([k for k in [q[0] for q in y] if k < y[i][0]])) for i in range(len(y) - 3, len(y))]
    contour_y_duration = [len(set([k for k in [q[1] for q in y] if k < y[i][1]])) for i in range(len(y) - 3, len(y))]
    return contour_x_pitch[-3:] == contour_y_pitch[-3:] and contour_x_duration[-3:] == contour_y_duration[-3:]

# Function to check if both sequences start with a long note (duration >= 2)
def hasLongFirstNote(x, y):
    return x[0][1] >= 2 and y[0][1] >= 2

# Function to check if both sequences end with a long note (duration >= 2)
def hasLongLastNote(x, y):
    return x[-1][1] >= 2 and y[-1][1] >= 2

# Function to check if either half of one sequence's rhythm appears in the other sequence
def hasSameRhythmSequence(x, y):
    if hasSameRhythm(x, y):
        return True
    if len(x) < 4 or len(y) < 4:
        return False
    x_rhythms = [n[1] for n in x]
    y_rhythms = [n[1] for n in y]
    x_first_half_rhythms = [n[1] for n in x[:(len(x) // 2)]]
    x_second_half_rhythms = [n[1] for n in x[(len(x) // 2):]]
    y_first_half_rhythms = [n[1] for n in y[:(len(y) // 2)]]
    y_second_half_rhythms = [n[1] for n in y[(len(y) // 2):]]
    if (len(x_first_half_rhythms) > 2 and symhelpers.listSequenceIn(x_first_half_rhythms, y_rhythms)) \
        or (len(x_second_half_rhythms) > 2 and symhelpers.listSequenceIn(x_second_half_rhythms, y_rhythms)) \
        or (len(y_first_half_rhythms) > 2 and symhelpers.listSequenceIn(y_first_half_rhythms, x_rhythms)) \
        or (len(y_second_half_rhythms) > 2 and symhelpers.listSequenceIn(y_second_half_rhythms, x_rhythms)):
        return True
    return False

# Function to check if both sequences have no leaps larger than a fifth and are within an octave
def hasNoLeapsAndWithinOctave(x, y):
    intervals_x = [x[i][0] - x[i - 1][0] for i in range(1, len(x))]
    intervals_y = [y[i][0] - y[i - 1][0] for i in range(1, len(y))]
    if any([abs(i) > 5 for i in intervals_x + intervals_y]):
        return False
    all_pitches = [i[0] for i in x + y]
    return max(all_pitches) - min(all_pitches) < 12

# Function to check if all notes in both sequences are part of the same diatonic scale
def hasAllInDiatonicScale(x, y):
    all_notes = set([i[0] % 12 for i in x + y])
    major_scale = [0, 2, 4, 5, 7, 9, 11]
    for root in range(12):
        if all_notes <= set([(note + root) % 12 for note in major_scale]):
            return True
    return False

# Function to check if both sequences have syncopation or neither have syncopation
def hasSyncopationOrNoSyncopation(x, y):
    if len(x) < 2 or len(y) < 2:
        return False
    onsets_x = [sum([q[1] for q in x[:i]]) for i in range(len(x))]
    onsets_y = [sum([q[1] for q in y[:i]]) for i in range(len(y))]
    sync_x = any([onsets_x[i] % 1 != 0 for i in range(1, len(onsets_x))])
    sync_y = any([onsets_y[i] % 1 != 0 for i in range(1, len(onsets_y))])
    return sync_x == sync_y

# Function to check if both sequences have consecutive short notes (duration < 1)
def hasConsecutiveShortNotes(x, y):
    durations_x = [n[1] for n in x]
    durations_y = [n[1] for n in y]
    short_x = any([durations_x[i] < 1 and durations_x[i - 1] < 1 for i in range(1, len(durations_x))])
    short_y = any([durations_y[i] < 1 and durations_y[i - 1] < 1 for i in range(1, len(durations_y))])
    return short_x and short_y


# Import necessary libraries for data processing
import pickle
import json

# Read the lines from the 'gutenberg-poetry-v001.ndjson' file, evaluate each line as a Python expression, and store them in a list
a = [eval(i) for i in (open("gutenberg-poetry-v001.ndjson", "r+")).readlines()]

# Initialize an empty list to store the poems
poems = []

# Set the current group id to 0, which will be used to group the poems
cur_gid = 0

# Iterate over each evaluated line from the file
for i in a:
    # Check if the current line's group id ('gid') is different from the current group id
    if i["gid"] != cur_gid:
        # If it is, start a new poem by appending an empty list to the poems list
        poems.append([])
    # Append the current line's text ('s') to the last poem in the poems list
    poems[-1].append(i["s"])
    # Update the current group id to the current line's group id
    cur_gid = i["gid"]

# Serialize the poems list using pickle and save it to the 'poems.pcl' file within the 'pickles' directory
pickle.dump(poems, open("pickles/poems.pcl", "wb"))

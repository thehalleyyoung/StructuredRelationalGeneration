
import pickle
import os
import re

# Define the folder containing poem data and the file type to look for
poetry_folder = "poems-data"
file_type = ".txt"
n_lines = 9  # Expected number of lines in each poem

poems = []  # List to store processed poems

# Get a list of files in the poetry folder
file_list = os.listdir(poetry_folder)

# Function to process each line of a poem
def process_line(line):
    line = line.decode('utf-8')  # Decode the line from bytes to a string
    line = line.strip()  # Remove leading and trailing whitespace
    line = re.sub(' +', ' ', line)  # Replace multiple spaces with a single space
    return line

# Loop through files in the poetry folder
for filename in os.listdir(poetry_folder):
    if filename.endswith(file_type):  # Check if the file is a text file
        with open(os.path.join(poetry_folder, filename), 'rb') as file:  # Open the file in binary read mode
            print(filename)  # Print the name of the file being processed
            poem = []  # Initialize a list to store lines of the current poem
            for line in file:  # Iterate over each line in the file
                line = process_line(line)  # Process the line
                poem.append(line)  # Add the processed line to the poem list
            if len(poem) != n_lines:  # Check if the poem has the expected number of lines
                print("wrong number of lines: " + str(len(poem)))  # Print a warning if the number of lines is incorrect
            else:
                poems.append(poem)  # Add the correctly-sized poem to the list of poems

# Print the total number of poems found
print("found " + str(len(poems)) + " poems")

# Save the list of poems to a pickle file
pickle.dump(poems, open("pickles/poems.pcl", "wb"))

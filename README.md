# Relational Neurosymbolic Generative Models

This is code to reproduce the results in "[Neurosymbolic Deep Generative Models for Sequence Data with Relational Constraints](https://proceedings.neurips.cc/paper_files/paper/2022/hash/f13ceb1b94145aad0e54186373cc86d7-Abstract-Conference.html)" from NEURIPS 2022.  The publication describes a unique type of program synthesis to extract latent structure from music and poetry, and then render that structure within new examples via multiple possible approaches.


To run the poetry example, download the json file at https://github.com/aparrish/gutenberg-poetry-corpus, download the poetry foundation dataset at https://www.kaggle.com/johnhallman/complete-poetryfoundationorg-dataset, place both in the poetry folder, and then run

```
cd poetry
python3 runeverything.py
```

To run the music example, download the music dataset at https://kern.humdrum.org/cgi-bin/browse?l=/essen, convert the contents to musicxml (for instance, through the musescore program), and then place the results in a folder inside the music folder in a dataset named "essen".  Then download the "cat-mel_2bar_big" model at https://github.com/magenta/magenta/tree/master/magenta/models/music_vae, and place it in a folder titled "cat-mel_2bar_big".  Then run

```python3 runeverything.py vae```

for Approach 3 as described in the paper on page 7, or

```python3 runeverything.py z3``` 

for Approach 2 as described in the paper on page 6.
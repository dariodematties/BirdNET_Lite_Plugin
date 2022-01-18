[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

# BirdNET_Plugin for avian diversity monitoring on the edge
The original [BirdNET](https://github.com/kahst/BirdNET) repository with the model for identification of birds by sounds is completely developed by [Stefan Kahl](https://github.com/kahst), [Shyam Madhusudhana](https://www.birds.cornell.edu/brp/shyam-madhusudhana/), and [Holger Klinck](https://www.birds.cornell.edu/brp/holger-klinck/).

This repository is a clone of the [original one](https://github.com/kahst/BirdNET) with the necessary modifications added in order to make it work as a plugin on the nodes of the the [Sage project](https://sagecontinuum.org/).

Basically I have incorporated the [pywaggle](https://github.com/waggle-sensor/pywaggle) functionality which allows to collect sounds from microphones as inputs for the model which identifies the birds that might have produced such sounds. Afterwards I use pywaggle to publish the model results as well as performance measures.

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

## Usage

Please, reffer to the [pywaggle](https://github.com/waggle-sensor/pywaggle) and [BirdNET-Lite](https://github.com/kahst/BirdNET-Lite) repositories to instal the necessary dependences.

For usage just clone this repository

`https://github.com/dariodematties/BirdNET_Lite_Plugin`

Then

`cd BirdNET_Lite_Plugin`

and run

`python3 analyze.py --num_rec 6 --sound_int 5`

which will record 6 audio files of 10 seconds each, analyze them, publish the results and inference times of each file and finally remove the recorded input files.

```
LOADING TF LITE MODEL... DONE!
IN THIS RUN  6  FILES OF  5.0  SECONDS WILL BE PROCESSED
RECORDING NUMBER:  0
RECORDING AUDIO FROM MIC DURING:  5.0  SECONDS...  DONE!
RECORDING NUMBER:  1
RECORDING AUDIO FROM MIC DURING:  5.0  SECONDS...  DONE!
RECORDING NUMBER:  2
RECORDING AUDIO FROM MIC DURING:  5.0  SECONDS...  DONE!
RECORDING NUMBER:  3
RECORDING AUDIO FROM MIC DURING:  5.0  SECONDS...  DONE!
RECORDING NUMBER:  4
RECORDING AUDIO FROM MIC DURING:  5.0  SECONDS...  DONE!
RECORDING NUMBER:  5
RECORDING AUDIO FROM MIC DURING:  5.0  SECONDS...  DONE!
FILES IN DATASET: 6
READING AUDIO DATA... DONE! READ 2 CHUNKS.
READING AUDIO DATA... DONE! READ 2 CHUNKS.
READING AUDIO DATA... DONE! READ 2 CHUNKS.
READING AUDIO DATA... DONE! READ 2 CHUNKS.
READING AUDIO DATA... DONE! READ 2 CHUNKS.
READING AUDIO DATA... DONE! READ 2 CHUNKS.
ANALYZING AUDIO... DONE! Time 0.3 SECONDS
ANALYZING AUDIO... DONE! Time 0.2 SECONDS
ANALYZING AUDIO... DONE! Time 0.2 SECONDS
ANALYZING AUDIO... DONE! Time 0.2 SECONDS
ANALYZING AUDIO... DONE! Time 0.2 SECONDS
ANALYZING AUDIO... DONE! Time 0.2 SECONDS
PUBLISHING DETECTION 0 ... DONE!
PUBLISHING DETECTION 1 ... DONE!
PUBLISHING DETECTION 2 ... DONE!
PUBLISHING DETECTION 3 ... DONE!
PUBLISHING DETECTION 4 ... DONE!
PUBLISHING DETECTION 5 ... DONE!
REMOVING THE INPUT COLLECTED BY THE MICROPHONE ... DONE!
```

Beyond publishing, if you want to save the outputs of the model in files, first create a folder in the directory of the project; let's say

`mkdir output`

Then, run the following command

`python3 analyze.py --num_rec 6 --sound_int 5 --o output --min_conf 0.01`

This will instruct the script to save the results of the analysis of the different recordings in the directory `output`

The format of the output is 
```
tart (s);End (s);Scientific name;Common name;Confidence
0.0;3.0;Dicrurus paradiseus;Greater Racket-tailed Drongo;0.025848534
0.0;3.0;Capito wallacei;Scarlet-banded Barbet;0.019484155
0.0;3.0;Dendrocopos leucotos;White-backed Woodpecker;0.013595802
0.0;3.0;Myophonus horsfieldii;Malabar Whistling-Thrush;0.012276235
0.0;3.0;Grallaria flavotincta;Yellow-breasted Antpitta;0.01151198
0.0;3.0;Formicarius rufipectus;Rufous-breasted Antthrush;0.011332592
3.0;6.0;Capito wallacei;Scarlet-banded Barbet;0.058079723
3.0;6.0;Saltator grossus;Slate-colored Grosbeak;0.03138132
3.0;6.0;Sylvia abyssinica;African Hill Babbler;0.023456778
3.0;6.0;Copsychus luzoniensis;White-browed Shama;0.02323199
3.0;6.0;Hypsipetes everetti;Yellowish Bulbul;0.022299841
3.0;6.0;Sylvia atriceps;Rwenzori Hill Babbler;0.016690737
3.0;6.0;Copsychus malabaricus;White-rumped Shama;0.016142305
3.0;6.0;Saltator fuliginosus;Black-throated Grosbeak;0.0121659925
```



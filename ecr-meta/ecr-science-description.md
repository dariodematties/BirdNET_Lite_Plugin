# Science

By observing the trends in the diversity variations of certain species, researchers can track the current ecosystem conditions.
Birds are ideal to monitor the ecosystem's health since of the diversity of environments they can occupy is vast.
Relative to other species, birds are prominently chosen since they can be sensitive to similar factors affecting such species.
These facts make birds study one of the most effective baselines for the determination of the ecosystem health.
Furthermore, there are plenty of avian research efforts, which have also turned some avian species into model organisms, 
enabling the development of novel quantitative methods that can then be applied beyond ornithology.
As a consequence, birds could be rendered as sentinel species, umbrella species, model organisms, and flagship species.

Following this line, *Avian diversity monitoring on the edge* is an autonomous avian diversity monitoring system, which uses sounds taken from microphones located in natural areas.

This project will allow the determination of avian biodiversity autonomously through the use of machine learning on edge devices by placing microphones in specific forest locations. Consequently it will be possible to get exposure to many different organisms occupying such areas without needing to detect them during demanding and expensive human fieldwork [1].

In the figure at the right (Credits to S. Kahl et al.) we can see an illustration of the utility of this network.
In  such a figure we can see the migratory species occurrence correlation (r) between weekly cumulative BirdNET detections (in blue) and human point count observations (eBird checklist frequency, in red). As can be see in the plot, the detections of the Network closely resemble human observational performance. In [1], the authors achieved a high correlation for migratory species that vocalize frequently (i.e., multiple hundreds of detections per week). This is indicative of the importance of this kind of automated detection systems.

# AI@Edge

We are using a DNN, called **BirdNET**, which is designed to bird sound recognition of more than 6,000 species worldwide.. The model architecture is derived from the family of residual networks (ResNets), consists of 157 layers with more than 27 million parameters, and was trained using extensive data pre-processing, augmentation, and mixup [1].


# Using the code

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


# Arguments

This plugin has the following knobs

   **--num_rec**      'Number of microphone recordings. Each mic recording will be saved in a different file. Default to 1.'
    
   **--silence_int**  'Time interval [s] in which there is not sound recording. Default to 1.0.'
    
   **--sound_int**    'Time interval [s] in which there is sound recording. Default to 10.0.'

   **--i**			      'Path to input file or directory. If not specified, the plugin will record from the microphone.
   
   **--o**			      'Path to output directory. If not specified, the the plugin will not save the output on files (just publish them by means of pywaggle).'
   
   **--lat**		      'Recording location latitude. It is a float. Set -1 to ignore (which is set by default).
   
   **--lon**		      'Recording location longitude. It is a float. Set -1 to ignore (which is set by default).
   
   **--week**		      'Week of the year when the recordings were made. It is an int. Values in [1, 48]. Set -1 to ignore (which is set by default).'
   
   **--overlap**		  'Overlap in seconds between extracted spectrograms. It is a float. Values in [0.0, 2.9]. Default is 0.0.'
   
   **--sensitivity**	'Sigmoid sensitivity; Higher values result in lower sensitivity. It is a float. Values in [0.25, 2.0]. Defaults to 1.0.'
   
   **--min_conf**     'Minimum confidence threshold. Values in [0.01, 0.99]. It is a float. Defaults to 0.1.'

   **--custom_list**  'Path to text file containing a list of species. Not used if not provided.
   
   **--keep**         'Keeps all the input files collected from the mic. Default is false'

# Plugin outputs

The Output of the plugin is a text file

  * **Begin Time (s):** Time mark to which the detection begins.

  * **End Time (s):** Time mark to which the detection ends.

  * **Scientific Name:** Name given to the species by scientists.

  * **Common Name:** Colloquial name given to the species.

  * **Confidence:** Classification confidence level (from 0.0 to 1.0).


# References


[1] Stefan Kahl, Connor M. Wood, Maximilian Eibl and Holger Klinck. BirdNET: A deep learning solution for avian diversity monitoring. Ecological Informatics Volume 61, March 2021.


# Credits

- Image credit:
  * Creator: Becky Matsubara 
  * Copyright: © 2017, Becky Matsubara

- Original [BirdNET](https://github.com/kahst/BirdNET-Lite) network by
  * Stefan Kahl
  * Shyam Madhusudhana
  * Holger Klinck

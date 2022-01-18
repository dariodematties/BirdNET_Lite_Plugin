# Avian diversity monitoring on the edge

## Usage

### This plugin has the following knobs

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



### The Output of the plugin is a csv text file

  * **Begin Time (s):** Time mark to which the detection begins.

  * **End Time (s):** Time mark to which the detection ends.

  * **Scientific Name:** Name given to the species by scientists.

  * **Common Name:** Colloquial name given to the species.

  * **Confidence:** Classification confidence level (from 0.0 to 1.0).

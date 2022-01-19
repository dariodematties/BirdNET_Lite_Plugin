
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

try:
    import tflite_runtime.interpreter as tflite
except:
    from tensorflow import lite as tflite

import argparse
import operator
# import librosa
from scipy.io import wavfile
from scipy import interpolate

import numpy as np
import math
import time
import shutil

from waggle.plugin import Plugin
from waggle.data.audio import Microphone

from time import sleep

def readAudioDataset(args):
    # Parse dataset
    dataset = parseTestSet(args.i, args.filetype)

    # Read audio data
    audioData = []
    timeStamps = []
    for s in dataset:
        audioData.append(readAudioData(s, args.overlap))
        #audioData = readAudioData(args.i, args.overlap)
        timeStamps.append(os.path.getmtime(s))

    return audioData, timeStamps

def parseTestSet(path, file_type='wav'):

    # Find all soundscape files
    dataset = []
    if os.path.isfile(path):
        dataset.append(path)
    else:
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                if f.rsplit('.', 1)[-1].lower() == file_type:
                    dataset.append(os.path.abspath(os.path.join(dirpath, f)))

    # Dataset stats
    print('FILES IN DATASET:', len(dataset))
    
    return dataset

def audioRecording(path, number_of_recordings, silence_interval, sound_interval, file_type='wav'):

    print('IN THIS RUN ', number_of_recordings, ' FILES OF ', sound_interval, ' SECONDS WILL BE PROCESSED')
    microphone = Microphone(samplerate=48000)
    for i in range(number_of_recordings):
        # Recording audio
        print('RECORDING NUMBER: ', i)
        print('RECORDING AUDIO FROM MIC DURING: ', sound_interval, ' SECONDS... ', end=' ')
        sample = microphone.record(sound_interval)
        filename = "sample_" + str(i) + "." + file_type
        sample.save(os.path.abspath(os.path.join(path, filename)))
        print('DONE!')
        sleep(silence_interval)

def loadModel():

    global INPUT_LAYER_INDEX
    global OUTPUT_LAYER_INDEX
    global MDATA_INPUT_INDEX
    global CLASSES

    print('LOADING TF LITE MODEL...', end=' ')

    # Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path='model/BirdNET_6K_GLOBAL_MODEL.tflite')
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get input tensor index
    INPUT_LAYER_INDEX = input_details[0]['index']
    MDATA_INPUT_INDEX = input_details[1]['index']
    OUTPUT_LAYER_INDEX = output_details[0]['index']

    # Load labels
    CLASSES = []
    with open('model/labels.txt', 'r', encoding="utf-8") as lfile:
        for line in lfile.readlines():
            CLASSES.append(line.replace('\n', ''))

    print('DONE!')

    return interpreter

def loadCustomSpeciesList(path):

    slist = []
    if os.path.isfile(path):
        with open(path, 'r') as csfile:
            for line in csfile.readlines():
                slist.append(line.replace('\r', '').replace('\n', ''))

    return slist

def splitSignal(sig, rate, overlap, seconds=3.0, minlen=1.5):

    # Split signal with overlap
    sig_splits = []
    for i in range(0, len(sig), int((seconds - overlap) * rate)):
        split = sig[i:i + int(seconds * rate)]

        # End of signal?
        if len(split) < int(minlen * rate):
            break
        
        # Signal chunk too short? Fill with zeros.
        if len(split) < int(rate * seconds):
            temp = np.zeros((int(rate * seconds)))
            temp[:len(split)] = split
            split = temp
        
        sig_splits.append(split)

    return sig_splits

def readAudioData(path, overlap, sample_rate=48000):

    print('READING AUDIO DATA...', end=' ', flush=True)

    # Open file with librosa (uses ffmpeg or libav)
    # sig, rate = librosa.load(path, sr=sample_rate, mono=True, res_type='kaiser_fast')
    old_rate, old_sig = wavfile.read(path)
    if len(old_sig.shape) == 2:
        old_sig = old_sig[:,0]

    if old_rate != sample_rate:
        duration = old_sig.shape[0] / old_rate
        
        time_old  = np.linspace(0, duration, old_sig.shape[0])
        time_new  = np.linspace(0, duration, int(old_sig.shape[0] * sample_rate / old_rate))
        
        interpolator = interpolate.interp1d(time_old, old_sig.T)
        sig = interpolator(time_new).T
        sig = np.round(sig).astype(old_sig.dtype)
    else:
        sig = old_sig

    rate = sample_rate


    # Split audio into 3-second chunks
    chunks = splitSignal(sig, rate, overlap)

    print('DONE! READ', str(len(chunks)), 'CHUNKS.')

    return chunks

def convertMetadata(m):

    # Convert week to cosine
    if m[2] >= 1 and m[2] <= 48:
        m[2] = math.cos(math.radians(m[2] * 7.5)) + 1 
    else:
        m[2] = -1

    # Add binary mask
    mask = np.ones((3,))
    if m[0] == -1 or m[1] == -1:
        mask = np.zeros((3,))
    if m[2] == -1:
        mask[2] = 0.0

    return np.concatenate([m, mask])

def custom_sigmoid(x, sensitivity=1.0):
    return 1 / (1.0 + np.exp(-sensitivity * x))

def predict(sample, interpreter, sensitivity):

    # Make a prediction
    interpreter.set_tensor(INPUT_LAYER_INDEX, np.array(sample[0], dtype='float32'))
    interpreter.set_tensor(MDATA_INPUT_INDEX, np.array(sample[1], dtype='float32'))
    interpreter.invoke()
    prediction = interpreter.get_tensor(OUTPUT_LAYER_INDEX)[0]

    # Apply custom sigmoid
    p_sigmoid = custom_sigmoid(prediction, sensitivity)

    # Get label and scores for pooled predictions
    p_labels = dict(zip(CLASSES, p_sigmoid))

    # Sort by score
    p_sorted = sorted(p_labels.items(), key=operator.itemgetter(1), reverse=True)

    # Remove species that are on blacklist
    for i in range(min(10, len(p_sorted))):
        if p_sorted[i][0] in ['Human_Human', 'Non-bird_Non-bird', 'Noise_Noise']:
            p_sorted[i] = (p_sorted[i][0], 0.0)

    # Only return first the top ten results
    return p_sorted[:10]

def analyzeAudioData(chunks, lat, lon, week, sensitivity, overlap, interpreter):

    detections = {}
    start = time.time()
    print('ANALYZING AUDIO...', end=' ', flush=True)

    # Convert and prepare metadata
    mdata = convertMetadata(np.array([lat, lon, week]))
    mdata = np.expand_dims(mdata, 0)

    # Parse every chunk
    pred_start = 0.0
    for c in chunks:

        # Prepare as input signal
        sig = np.expand_dims(c, 0)

        # Make prediction
        p = predict([sig, mdata], interpreter, sensitivity)

        # Save result and timestamp
        pred_end = pred_start + 3.0
        detections[str(pred_start) + ';' + str(pred_end)] = p
        pred_start = pred_end - overlap

    print('DONE! Time', int((time.time() - start) * 10) / 10.0, 'SECONDS')

    return detections

def writeResultsToFile(allDetections, min_conf, path):
    if os.path.isdir(path):
        for dets_n, detections in enumerate(allDetections):
            print('WRITING RESULTS TO', path, '...', end=' ')
            rcnt = 0
            with open(path + '/result_' + str(dets_n) + '.csv', 'w') as rfile:
                rfile.write('Start (s);End (s);Scientific name;Common name;Confidence\n')
                for d in detections:
                    for entry in detections[d]:
                        if entry[1] >= min_conf and (entry[0] in WHITE_LIST or len(WHITE_LIST) == 0):
                            rfile.write(d + ';' + entry[0].replace('_', ';') + ';' + str(entry[1]) + '\n')
                            rcnt += 1
            print('DETECTIONS', dets_n, 'DONE! WROTE', rcnt, 'RESULTS.')
    else:
        print("Unexpected output path: {}, it must be an existing directory" .format(path))

def publishDatections(plugin, allDetections, timeStamps, args, min_conf, WHITE_LIST):
        for i, (detections, timestamp) in enumerate(zip(allDetections, timeStamps)):
            print('PUBLISHING DETECTION', i, '...', end=' ')
            for d in detections:
                times = d.split(';')
                start_time = times[0]
                end_time = times[1]
                for entry in detections[d]:
                    if entry[1] >= min_conf and (entry[0] in WHITE_LIST or len(WHITE_LIST) == 0):
                        class_label = entry[0].split('_')
                        scientific_name = class_label[0].lower().replace(' ', '_')
                        common_name = class_label[1].lower()
                        common_name = ''.join(e for e in common_name if e.isalnum())
                        plugin.publish(f'env.detection.avian.{start_time}', str(entry[1]), timestamp=timestamp, meta={'record_duration': args.sound_int})
                        plugin.publish(f'env.detection.avian.{end_time}', str(entry[1]), timestamp=timestamp, meta={'record_duration': args.sound_int})
                        plugin.publish(f'env.detection.avian.{scientific_name}', str(entry[1]), timestamp=timestamp, meta={'record_duration': args.sound_int})
                        plugin.publish(f'env.detection.avian.{common_name}', str(entry[1]), timestamp=timestamp, meta={'record_duration': args.sound_int})

            print('DONE!')


def main():

    global WHITE_LIST

    # Parse passed arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_rec', type=int, default=1, help='Number of microphone recordings. Each mic recording will be saved in a different file. Default to 1.')
    parser.add_argument('--silence_int', type=float, default=1.0, help='Time interval [s] in which there is not sound recording. Default to 1.0.')
    parser.add_argument('--sound_int', type=float, default=10.0, help='Time interval [s] in which there is sound recording. Default to 10.0.')

    parser.add_argument('--i', help='Path to input file. If not specified, the plugin will record from the microphone')
    parser.add_argument('--o', default='', help='Path to output file. Defaults to None.')
    parser.add_argument('--filetype', default='wav', help='Filetype of soundscape recordings. Defaults to \'wav\'.')
    parser.add_argument('--lat', type=float, default=-1, help='Recording location latitude. Set -1 to ignore.')
    parser.add_argument('--lon', type=float, default=-1, help='Recording location longitude. Set -1 to ignore.')
    parser.add_argument('--week', type=int, default=-1, help='Week of the year when the recording was made. Values in [1, 48] (4 weeks per month). Set -1 to ignore.')
    parser.add_argument('--overlap', type=float, default=0.0, help='Overlap in seconds between extracted spectrograms. Values in [0.0, 2.9]. Defaults tp 0.0.')
    parser.add_argument('--sensitivity', type=float, default=1.0, help='Detection sensitivity; Higher values result in higher sensitivity. Values in [0.5, 1.5]. Defaults to 1.0.')
    parser.add_argument('--min_conf', type=float, default=0.1, help='Minimum confidence threshold. Values in [0.01, 0.99]. Defaults to 0.1.')   
    parser.add_argument('--custom_list', default='', help='Path to text file containing a list of species. Not used if not provided.')
    parser.add_argument('--keep', action='store_true', help='Keeps all the input files collected from the mic.')

    args = parser.parse_args()

    enable_rm = False

    with Plugin() as plugin:
        with plugin.timeit("plugin.duration.loadmodel"):
            # Load model
            interpreter = loadModel()

            # Load custom species list
            if not args.custom_list == '':
                WHITE_LIST = loadCustomSpeciesList(args.custom_list)
            else:
                WHITE_LIST = []

        with plugin.timeit("plugin.duration.input"):
            if args.i == None:
                # Record audio from microphone
                dir_name = "mic_dir_" + str(time.time())
                os.mkdir(dir_name)
                args.i = dir_name
                audioRecording(args.i, args.num_rec, args.silence_int, args.sound_int, args.filetype)
                enable_rm = True

            audioData, timeStamps = readAudioDataset(args)

        with plugin.timeit("plugin.duration.inference"):
            # Process audio data and get detections
            week = max(1, min(args.week, 48))
            sensitivity = max(0.5, min(1.0 - (args.sensitivity - 1.0), 1.5))
            
            allDetections = []
            for data in audioData:
                allDetections.append(analyzeAudioData(data, args.lat, args.lon, week, sensitivity, args.overlap, interpreter))

        # Write detections to output file
        min_conf = max(0.01, min(args.min_conf, 0.99))
        if args.o:
            writeResultsToFile(allDetections, min_conf, args.o)

        # Publish detections
        publishDatections(plugin, allDetections, timeStamps, args, min_conf, WHITE_LIST)

        if not args.keep and enable_rm:
            print('REMOVING THE INPUT COLLECTED BY THE MICROPHONE ...', end=' ')
            shutil.rmtree(dir_name)
            print('DONE!')

if __name__ == '__main__':

    main()

    # Example calls
    # The following will produce 6 recording of 5 seconds each at 1 second silent intervals
    # python3 analyze.py --num_rec 6 --sound_int 5 --lat 35.4244 --lon -120.7463 --week 18
    # python3 analyze.py --num_rec 6 --sound_int 5 --lat 47.6766 --lon -122.294 --week 11 --overlap 1.5 --min_conf 0.25 --sensitivity 1.25 --custom_list 'example/custom_species_list.txt'

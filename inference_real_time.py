import model
from pathlib import Path
import torch
import librosa
import numpy as np
from Audio_proc_lib.audio_proc_functions import load_music,sound_write
import Create_Dataset
import json
import evaluate
import argparse
import pyaudio
import os
import struct
import matplotlib.pyplot as plot
from tkinter import TclError
#from getkey import getkey, keys

#TODO
#1. train the model invariant to signal scale ampitude
#2. train the model to Fs=44100hz


if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()


    parser.add_argument('--Model_dir', type=str, default="~/ML_pipeline_code_multiple_classes_v1/Spectrograms_tst/pretr_model",
                            help='Flag to add the sisec18 methods for comparison  with the current method that we examine ')   

    parser.add_argument('--input-wav', type=str, default="~/ML_pipeline_code_multiple_classes_v1//tst_dataset/test/kinhsh_koble/κίνηση-κομπλε-3.wav",
                            help='Flag to add the sisec18 methods for comparison  with the current method that we examine ') 


    args = parser.parse_args()  


    model_path = args.Model_dir


    with open(model_path+'\\model.json', 'r') as openfile:
        # Reading from json file
        train_json = json.load(openfile)    

    Dataset_params = train_json["args"]["Dataset_params"]
    nb_classes = train_json["args"]["nb_classes"]
    classes_lookup = Dataset_params["classes_lookup"]
    fc_dim = train_json["args"]["fc_dim"]               
    evaluate.nb_classes = nb_classes
    evaluate.fc_dim = fc_dim
    classes_names = list(Dataset_params["classes_lookup"].keys())

    Forward , Backward = Create_Dataset.pick_front_end(front_end_params = Dataset_params["FE_params"], seq_dur= Dataset_params["seq_dur"], Fs = Dataset_params["Fs"] )

    #Load model
    pretr_model = evaluate.load_target_models(model_path=model_path)    


    # constants
    FORMAT = pyaudio.paInt16                # audio format (bytes per sample?)
    CHANNELS = 1                            # single channel for microphone
    RATE = Dataset_params["Fs"]             # samples per second
    CHUNK = Dataset_params["seq_dur"]*RATE  # samples per frame 1024, 512, 256, 128

    # pyaudio class instance
    p = pyaudio.PyAudio()

    # stream object to get data from microphone
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        output=True,
        frames_per_buffer=CHUNK
    )    

# #if q pressed while exits for linux only
#     filedescriptors = termios.tcgetattr(sys.stdin)
#     tty.setcbreak(sys.stdin)
#     stop_hear = 0

#forever hear and classify
    while True:
                
        # binary data
        data = stream.read(CHUNK)  
        
        # convert data to integers, make np array, then offset it by 127
        # data_int = struct.unpack(str(2 * CHUNK) + 'B', data)
        data_int = struct.unpack_from ("%dh" % CHUNK * 1, data)
        
        # create np array and offset by 128
        # data_np = np.array(data_int, dtype='b')[::2] + 128
        # mono = np.array(data_int)[::2] - 128
        mono = np.array(data_int)

        # 2 ^ 16 = 65536 -> [-32768, 32767]
        # librosa simply scale by dividing each entry by 32768
        x = mono/32768
        x_sig=x


        #SEGMENTING and FORWARDING--------------------------------------------------------------------------------------------------------
        X_segs = Create_Dataset.get_Forward_segs_from_one_song(x,Dataset_params,Forward)
        X_segs = np.expand_dims(np.array(X_segs),1)
        X_amp = np.abs(X_segs)
        x = torch.from_numpy(X_amp).float()
        seg_pred_class = evaluate.inference(x,pretr_model)   

        print(classes_names[seg_pred_class])       

        #plot realtime
        #PLOTTING--------------------------------
           
        fig, axs = plot.subplots(nrows=2, ncols=2, figsize=(10, 7))
                
        axs[0,0].set_title('Plot the input sound .wav signal in time')
        # axs[0,0].set_ylim(-32000,32000)
        # axs[0,0].set_xlim = (0,CHUNK*5)
        axs[0,0].plot(x_sig)
        axs[0,0].set_xlabel('Sound for classification')
        axs[0,0].set_ylabel('Amplitude')
                
        axs[1,0].set_title('Plot the input sound .wav signal in Spectrum')
        axs[1,0].phase_spectrum(x_sig,Fs=Dataset_params["Fs"])
        axs[1,0].set_xlabel('Sound for classification')
        axs[1,0].set_ylabel('Magnitude')
            
                
        axs[0,1].set_title('Spectrogram of input .wav sound ')
        axs[0,1].specgram(x_sig,Fs=Dataset_params["Fs"],NFFT=2048,scale="dB")
        #plot.specgram(X_spec_full,Fs=Dataset_params["Fs"])
        axs[0,1].set_xlabel('Time')
        axs[0,1].set_ylabel('Frequency')
        
        # axs[1,1].text(2, 0.25,classes_names[seg_pred_class],fontsize=8)
        # axs[1,1].set_xlabel('Classification')
        # axs[1,1].set_ylabel('Class')
        fig.tight_layout()
        fig.show()

        fig.canvas.draw()
        fig.canvas.flush_events()

        # #stop_hear=input('enter to continue or q + enter to stop:')
        # for linux 
        # stop_hear=sys.stdin.read(1)[0]
        # if stop_hear=="q":
        #     exit()
        # else:
        #     continue

        # print("press q to exit or any key to continue hearing: " )
        # key = getkey()
        # if key == 'q':
        #     exit()
        # else:
        #     continue

  
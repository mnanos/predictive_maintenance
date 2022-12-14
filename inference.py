import model
from pathlib import Path
import torch
import librosa
import numpy as np

from IPython.display import Audio
import matplotlib.pyplot as plot




if __name__ == "__main__":

    from Audio_proc_lib.audio_proc_functions import load_music,sound_write
    import Create_Dataset
    import json
    import numpy as np
    import evaluate
    import argparse



    parser = argparse.ArgumentParser()


    parser.add_argument('--Model_dir', type=str, default="/home/mnanos/ML_pipeline_code_multiple_classes_v0/Spectrograms_tst",
                            help='Flag to add the sisec18 methods for comparison  with the current method that we examine ')   

    parser.add_argument('--input-wav', type=str, default="/home/mnanos/ML_pipeline_code_multiple_classes_v0/Spectrograms_tst",
                            help='Flag to add the sisec18 methods for comparison  with the current method that we examine ') 


    args = parser.parse_args()  


    model_path = args.Model_dir


    with open(model_path+'/model.json', 'r') as openfile:
        # Reading from json file
        train_json = json.load(openfile)    

    Dataset_params = train_json["args"]["Dataset_params"]
    nb_classes = train_json["args"]["nb_classes"]
    classes_lookup = Dataset_params["classes_lookup"]
    fc_dim = train_json["args"]["fc_dim"]               
    evaluate.nb_classes = nb_classes
    evaluate.fc_dim = fc_dim


    Forward , Backward = Create_Dataset.pick_front_end(front_end_params = Dataset_params["FE_params"], seq_dur= Dataset_params["seq_dur"], Fs = Dataset_params["Fs"] )

  
    #TODO:play input .wav sound
    x,_ = librosa.load(args.input_wav, sr=Dataset_params["Fs"])
    x_sig = x
    

    pretr_model = evaluate.load_target_models(model_path=model_path)

    #SEGMENTING and FORWARDING--------------------------------------------------------------------------------------------------------
    X_segs = Create_Dataset.get_Forward_segs_from_one_song(x,Dataset_params,Forward)
    X_segs = np.expand_dims(np.array(X_segs),1)
    X_amp = np.abs(X_segs)
    x = torch.from_numpy(X_amp).float()
    #print(x)
    seg_pred_class = evaluate.inference(x,pretr_model)    
    
    
    
    #Compute histogram------------------------------------------------------------------------
    print(seg_pred_class)
    print("Most frequent value in the above array:")
    pred_class = np.bincount(seg_pred_class).argmax()
    print(pred_class) 
       
    classes_names = list(Dataset_params["classes_lookup"].keys())

    Answer = classes_names[pred_class]
    #Print answer and plot  
    print(Answer)
    
    
    #PLOTTING--------------------------------
    fig, axs = plot.subplots(nrows=2, ncols=2, figsize=(10, 7))

    axs[0,0].set_title('Plot the input sound .wav signal in time')
    axs[0,0].plot(x_sig)
    axs[0,0].set_xlabel('Sound for classification')
    axs[0,0].set_ylabel('Amplitude')
    
    
    axs[1,0].set_title('Plot the input sound .wav signal in Spectrum')
    axs[1,0].phase_spectrum(x_sig,Fs=Dataset_params["Fs"])
    axs[1,0].set_xlabel('Sound for classification')
    axs[1,0].set_ylabel('Magnitude')
    
    axs[0,1].set_title('Spectrogram of input .wav sound')
    axs[0,1].specgram(x_sig,Fs=Dataset_params["Fs"],NFFT=2048,scale="dB")
    #plot.specgram(X_spec_full,Fs=Dataset_params["Fs"])
    axs[0,1].set_xlabel('Time')
    axs[0,1].set_ylabel('Frequency')
    
    
    axs[1,1].set_title('Histogram of classified segments')
    axs[1,1].hist(seg_pred_class)
    
    plot.tight_layout()
    
    plot.show()






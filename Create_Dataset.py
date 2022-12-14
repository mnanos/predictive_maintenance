
import numpy as np
import argparse
import librosa
import random
import torch
import os
import yaml    
import json
import tqdm
from pathlib import Path



'''
    Module to create the training folder for the source-sep task containing the (X,Y) spectrograms

    Params:
    
        Wav_folder
        seq_dur
        FE_params
        target_source
        Fs
        other preproc

'''


def padding_or_not(x,M,flag):
    #FUNCTION to PADD or TRUNCATE a signal x in order 
    #for it's length to be perfectly divisible by M ( L=k*M i.e. L is an integer multiple of M) 

    L = len(x)
    reminder = int(L%M)

    if reminder:
        #L%M!=0 (not perfect division)

        if flag:
            #ZERO PADDING CASE:

            #Find the number of zeros that we will pad
            nb_zeros = int( M*np.ceil(L/M) - L )
            x_new = np.concatenate(( x , np.zeros(nb_zeros) ))

        else:
            #TRUNCATING SAMPLES CASE:
            trunc_until = int(L - reminder)
            x_new = x[:trunc_until] 

    else:
        x_new = x        

    return x_new


def consec_segments_tensor(x,valid_seq_dur,sample_rate):
    
    x = padding_or_not(x,valid_seq_dur*sample_rate,1) + np.finfo(np.float64).eps   #Adding epsilon in order to use STFT_mine :)

    nb_samples = int(len(x)/(valid_seq_dur*sample_rate)) 
    nb_frames = int(valid_seq_dur*sample_rate) 

    #reshaping to nb_segments,nb_frames,nb_channels and then permuting to get the correct tensor for the 
    x_segs = x.reshape(nb_samples , nb_frames)


    return x_segs 



def cputime():
    utime, stime, cutime, cstime, elapsed_time = os.times()
    return utime






#DETRMINING FE FORWARD-BACKWARD-----------------------------------------------------------------------------------------------

# //NSGT_SCALE_FRAMES
# "{ front_end_name : NSGT_SCALE_FRAMES , onset_det : custom , ksi_s : 44100 , min_scl : 512 , ovrlp_fact : 0.7 , middle_window : sg.hanning , matrix_form : 1 , multiproc : 1 }",


# //STFT_custom
# //"{ front_end_name : STFT_custom , a : 1024 , M : 4096 , support : 4096 }"

# //STFT_scipy
# //"{ front_end_name : scipy , a : 1024 , M : 4096 , support : 4096 }"

# //NSGT_CQT
# //"{ front_end_name : NSGT_CQT , ksi_s : 44100 , ksi_min : 32.07 , ksi_max : 10000 , B : 12 , matrix_form : 1 }"




def pick_front_end(front_end_params,seq_dur,Fs):  

    '''
        A FUNCTION to get front_end forward and backward methods---

            front_end_params : A dict containing the front end params
            seq_dur : SEQ-DUR IN SECONDS
            Fs : sampling rate
    '''
    


    #FRONT_ENDs available (its scalable)
    from Time_Frequency_Analysis.NSGT_CQT import NSGT_cqt
    from Time_Frequency_Analysis.SCALE_FRAMES import scale_frame
    from Time_Frequency_Analysis.STFT_custom import STFT_CUSTOM
    import nsgt as nsg      
    import scipy   
    import librosa  
    import numpy as np 
    
    
    front_end_lookup ={
        "STFT_custom":STFT_CUSTOM,
        "librosa":librosa,
        "scipy":scipy.signal,
        "nsgt_grr":nsg,
        "NSGT_CQT":NSGT_cqt,
        "NSGT_SCALE_FRAMES":scale_frame
    }


    #The STFTs and CQTs are L (signal len) dependend ie the only thing needed to construct the transform windows is L
    L = seq_dur*Fs
    #scale_frame is SIGNAL DEPENDEND i.e. in order to determine the windows positions (and consecuently construct them) you need the onsets
    #of the particular signal (its more complicated for the stereo chanel case so we test the mono)
    # mono_mix = librosa.to_mono(musdb_track.audio.T)


    front_end = front_end_lookup[front_end_params["front_end_name"]]

    if front_end==scipy.signal:
        g = np.hanning(front_end_params["support"])
        forward = lambda y : scipy.signal.stft( y , window=g, nfft=front_end_params["M"] , noverlap=front_end_params["a"] ,nperseg=front_end_params["support"])[-1]
        backward = lambda Y : scipy.signal.istft( Y, window=g, nfft=front_end_params["M"] , noverlap=front_end_params["a"]  ,nperseg=front_end_params["support"])[1]

    elif front_end==librosa:
        g = np.hanning(front_end_params["support"])
        #X = np.array( list( map( lambda x :  librosa.stft(x,n_fft=front_end_params["M"],hop_length=front_end_params["a"],win_length=front_end_params["support"],window=g) , track.audio.T ) ) )
        forward = lambda y : librosa.stft( y=y , n_fft=front_end_params["M"],hop_length=front_end_params["a"],win_length=front_end_params["support"],window=g ) 
        backward = lambda Y : librosa.istft( stft_matrix=Y ,hop_length=front_end_params["a"]  ,win_length=front_end_params["support"],window=g )
    
    elif front_end==STFT_CUSTOM:
        g = np.hanning(front_end_params["support"])
        stft = front_end(g,front_end_params["a"],front_end_params["M"],front_end_params["support"],L)
        forward = stft.forward  
        backward = stft.backward
    
    elif front_end==nsg:
        scale = nsg.LogScale
        scl = scale(front_end_params["ksi_min"], front_end_params["ksi_max"], front_end_params["B"] )
        nsgt = nsg.NSGT(scl, Fs, Ls=L, real=1, matrixform=1, reducedform=0 ,multithreading=0)
        forward = nsgt.forward
        backward = nsgt.backward
        # get_Transform(front_end_params,L)
        


    elif front_end==NSGT_cqt:
        nsgt = front_end(ksi_s=Fs,ksi_min=front_end_params["ksi_min"], ksi_max=front_end_params["ksi_max"], B=front_end_params["B"],L=L,matrix_form=1)
        forward = nsgt.forward
        backward = nsgt.backward


    


    return forward , backward





#IMPORTANT FUNC
def get_Forward_segs_from_one_song(x,dataset_params,Forward):
    #Segment-Forward-Preproc---------------------------------------

    #Segment
    x_segs  = consec_segments_tensor(x,dataset_params["seq_dur"],dataset_params["Fs"])   

    #Forward
    X_segs = list( map( Forward  , x_segs ) )

    #Preproc (i.e. dB , standardization)


    return X_segs


def  create_spec_pair_list_for_one_wav_fold(root_class_wav_dir,label):

    path_dir_list = list( map( lambda dir : dir.path ,  os.scandir(root_class_wav_dir) ) )


    pbar = tqdm.tqdm( path_dir_list )

    print("Processing the "+root_class_wav_dir+"\n")
    for count,path in enumerate(pbar):  

        #LOADING - segmenting - Forward------------------------------------------
        x,_ = librosa.load( path  , sr = Fs , mono= True ) 


        X_segs =  get_Forward_segs_from_one_song(x,dataset_params,Forward)

        #Convert to spectrograms (amplitude) and maybe Preproc----------------------------------------------
        X_segs_amps = list( map( lambda seg : np.abs(seg) , X_segs ) ) 

        # XY_seg_amps =  list( map( lambda seg : (np.abs(seg[0]),np.abs(seg[1])) , zip(X_segs,Y_segs) ) ) 



        #MERGING THE TWO LISTS [(X_seg,Y_seg),..] and converting to torch tensors
        if count:
            Spec_seg_pair_list = Spec_seg_pair_list + random.sample( list( map( lambda seg : ( torch.from_numpy( np.array([seg]) ).float() , label ) , X_segs_amps  )  )  , len(X_segs) )    
        else:
            Spec_seg_pair_list = random.sample( list( map( lambda seg : ( torch.from_numpy( np.array([seg]) ).float() , label ), X_segs_amps  )  )  , len(X_segs) )

    return Spec_seg_pair_list

    

def create_spec_pair_list(root_dir):

    #root_dir referes to the root directory which contains the directories of each class

    #Creating the list containing the samples (Spec_seg,label)


    classes_dir_names = list(classes_lookup.keys())
    classes_dir_labels = list(classes_lookup.values())
    path_dir_list = list( map( lambda classes_dir_name : root_dir+"/"+classes_dir_name ,  classes_dir_names ) )

    pbar = tqdm.tqdm( zip(classes_dir_labels,path_dir_list) )

     
    for class_dir_label,path in pbar:  
   
        pbar.set_description("Processing the "+path+" class dir of wavs\n")   

        #FOR THE WAVs OF CLASS i-----------------------------------------------------------------------------------------------------
        Spec_pair_list_tmp = create_spec_pair_list_for_one_wav_fold(path,label=class_dir_label)


        #CREATING Spec_pair_list .
        #It is a list constaing the samples (Spec_seg,label) .
        #The shufling will be done in training time 
        if class_dir_label:
            Spec_pair_list = Spec_pair_list + Spec_pair_list_tmp
        else:
            Spec_pair_list = Spec_pair_list_tmp


    return Spec_pair_list


if __name__ == "__main__":




    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset-params', '--Dataset-params', type=yaml.safe_load,
                            help='provide Transform parameters as a quoted json string')

             


    args = parser.parse_args()    


    dataset_params = args.Dataset_params 
    Wav_folder = args.Dataset_params["Wav_folder"]
    Target_folder = args.Dataset_params["Target_folder"]
    Fs = args.Dataset_params["Fs"]
    seq_dur = args.Dataset_params["seq_dur"]
    classes_lookup = args.Dataset_params["classes_lookup"]
    FE_params = args.Dataset_params["FE_params"]
    Forward , _ = pick_front_end(front_end_params = FE_params, seq_dur=seq_dur, Fs=Fs)

    root_train_dir = Wav_folder+'/train'
    root_valid_dir = Wav_folder+'/valid'    

    #CREATE TARGET FOLDER
    target_path = Path(Target_folder)
    target_path.mkdir(parents=True, exist_ok=True)    

    #CREATE spec pair list-------------------------------------------------------------------------------
    t1 = cputime()

    Spec_seg_pair_list_train = create_spec_pair_list(root_train_dir)         

    Spec_seg_pair_list_valid = create_spec_pair_list(root_valid_dir)

    #Saving--------------------------------------------------------------------------------------------
    torch.save(Spec_seg_pair_list_train, Target_folder+'/Spec_seg_pair_list_train.pt')

    torch.save(Spec_seg_pair_list_valid, Target_folder+'/Spec_seg_pair_list_valid.pt')

    t2 = cputime()

    Calc_saving_Time = t2 - t1

    #Creating a json containing all the neccesary DATASET params and then Saving---------------------------
    dataset_params["Calc_Saving_Time"] = str(Calc_saving_Time/60)+" mins"

    json_object = json.dumps(dataset_params, indent=4)
    
    # Writing 
    with open(Target_folder+"/Dataset_Params_log.json", "w") as outfile:
        outfile.write(json_object)    


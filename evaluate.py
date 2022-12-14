import model
from pathlib import Path
import torch
import librosa
import numpy as np



#Load model------
def load_target_models(target="model", model_path="umxhq", device="cpu"):
    """
        INPUTS:
            target : the target source that your model is trained on
            model_path : the dir where the target.pth is located
            device : the device on which the processing (model computations) will be done

    """

    #Load the state of the model---------------------
    target_model_path = model_path+"/"+target+".pth"
    state = torch.load(target_model_path, map_location=device)    

    #INITIALIZE THE MODEL------------------
    
    #UNET arcitecture
    unmix = model.Net(fc_dim=fc_dim,nb_classes=nb_classes).to(device)           

    unmix.load_state_dict(state, strict=False)

    
    return unmix




def inference(x,model):

    model.eval()
    with torch.no_grad():

        #INFERENCE FRO ONE BATCH----------------------------------------------
        out_seg_response = model( x )
        # out_seg_response = model( x ).unsqueeze(0)

        # Normalized_x_out = np.array( list( map( lambda seg_resp :  torch.nn.Softmax(seg_resp).cpu().detach().numpy() , out_seg_response )) )
        Softmax = torch.nn.Softmax(dim = 1 if len(out_seg_response.shape) == 2 else 0)
        Normalized_out = Softmax(out_seg_response).cpu().detach().numpy() 

        #For each segment : Finding the highest propability (the index corresponding to the max) 
        seg_pred_class = np.argmax(Normalized_out,axis=  1 if len(out_seg_response.shape) == 2 else 0 )     

    return seg_pred_class


def evaluate(y_true,y_pred):

    #y_true,y_pred referes to all the test samples given to the NN 

    from sklearn.metrics import confusion_matrix

    classes_names = list(Dataset_params["classes_lookup"].keys())
    confusion = confusion_matrix(y_true, y_pred)
    print('Confusion Matrix\n')
    print(confusion)

    #importing accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_true, y_pred)))

    print('Micro Precision: {:.2f}'.format(precision_score(y_true, y_pred, average='micro')))
    print('Micro Recall: {:.2f}'.format(recall_score(y_true, y_pred, average='micro')))
    print('Micro F1-score: {:.2f}\n'.format(f1_score(y_true, y_pred, average='micro')))

    print('Macro Precision: {:.2f}'.format(precision_score(y_true, y_pred, average='macro')))
    print('Macro Recall: {:.2f}'.format(recall_score(y_true, y_pred, average='macro')))
    print('Macro F1-score: {:.2f}\n'.format(f1_score(y_true, y_pred, average='macro')))

    print('Weighted Precision: {:.2f}'.format(precision_score(y_true, y_pred, average='weighted')))
    print('Weighted Recall: {:.2f}'.format(recall_score(y_true, y_pred, average='weighted')))
    print('Weighted F1-score: {:.2f}'.format(f1_score(y_true, y_pred, average='weighted')))

    from sklearn.metrics import classification_report
    print('\nClassification Report\n')
    print(classification_report(y_true, y_pred, target_names=classes_names ))

    

def Inference_and_evaluate(val_sampler,model):
    
    #Inference_AND_EVALUATE-----------------------------------------------------------------------------------------

    device = "cpu"

   
    #INFERENCE--------------------------------------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        for count,(x , y) in enumerate(val_sampler):
            x = x.to(device)

            if count:

                #INFERENCE FOR ONE BATCH----------------------------------------------
                seg_pred_class = inference(x,model)     

                y_pred = y_pred + list(seg_pred_class)       
                #---------------------------------------------------------------------------

                y = y.cpu().detach().numpy() 

                y_true = y_true + list(y)


            else:
                #INFERENCE FOR ONE BATCH----------------------------------------------
                seg_pred_class = inference(x,model)    

                y_pred = list(seg_pred_class)                
                #---------------------------------------------------------------------------

                y = y.cpu().detach().numpy() 

                y_true = list(y)                  
    #-------------------------------------------------------------------------------------------------------------


    #EVALUATE------------------------------------------------------------------------------------------
    evaluate(y_true,y_pred)
            
       




def cputime():
    utime, stime, cutime, cstime, elapsed_time = os.times()
    return utime

if __name__ == "__main__":

    from Audio_proc_lib.audio_proc_functions import load_music,sound_write
    import Create_Dataset

    import json
    import numpy as np
    import argparse
    import yaml
    import tqdm
    import os


    parser = argparse.ArgumentParser()

    parser.add_argument('--method-name', type=str, default="TST_method",
                            help='Flag to add the sisec18 methods for comparison  with the current method that we examine ')           

    parser.add_argument('--Model_dir', type=str, default="/home/mnanos/Desktop/Spectrograms_tst",
                            help='Flag to add the sisec18 methods for comparison  with the current method that we examine ')                            


    parser.add_argument('--root_TEST_dir', type=str, default="/home/mnanos/Desktop/Spectrograms_tst",
                            help='Flag to add the sisec18 methods for comparison  with the current method that we examine ')                            


    parser.add_argument('--evaldir', type=str, default="/home/mnanos/Desktop/Spectrograms_tst",
                            help='Flag to add the sisec18 methods for comparison  with the current method that we examine ')                            



    args = parser.parse_args()   



    #LOADING MODEL AND FRONT_END--------------------------------------------------------------------------------
    model_path = args.Model_dir

    with open(model_path+'/model.json', 'r') as openfile:
        # Reading from json file
        train_json = json.load(openfile)    

    Dataset_params = train_json["args"]["Dataset_params"]
    nb_classes = train_json["args"]["nb_classes"]
    classes_lookup = Dataset_params["classes_lookup"]
    fc_dim = train_json["args"]["fc_dim"]
    
      

    #Load FE
    Forward , Backward = Create_Dataset.pick_front_end(front_end_params = Dataset_params["FE_params"], seq_dur= Dataset_params["seq_dur"], Fs = Dataset_params["Fs"] )

    #Load model
    Model = load_target_models(model_path=model_path)



    #PERFORMING INFERENCE LIKE TRAINING-----------------------------------------------

    t1 = cputime()


    root_test_dir = args.root_TEST_dir+"/test"

    Fs = Dataset_params["Fs"]
    Create_Dataset.Fs = Fs
    Create_Dataset.dataset_params = Dataset_params
    Create_Dataset.Forward = Forward
    Create_Dataset.classes_lookup = classes_lookup
    Spec_seg_pair_list_test = Create_Dataset.create_spec_pair_list(root_test_dir) 

    valid_sampler = torch.utils.data.DataLoader(
        Spec_seg_pair_list_test, batch_size=16, shuffle=False,  num_workers=4
    )    

    Inference_and_evaluate(valid_sampler,Model)

    # metrics = {
    #     'Recall' : Recall,
    #     'Precision' : Precision,
    #     'F1_score' : F1
    # }


    t2 = cputime()

    Calc_Time = t2 - t1




    #SAVING EVAL RESULTS AND LOGS to evaldir-------------------------------------------------
    # create evaldir dir if not exist
    eval_dir_path = Path(args.evaldir)
    eval_dir_path.mkdir(parents=True, exist_ok=True)


    #Saving logs-----------------------------------------------------
    Eval_Log  = vars(args)
    Eval_Log["method_name"]=args.method_name
    # Eval_Log["metrics"]=metrics
    Eval_Log["Calculation_Sep_Eval_time"] = str(Calc_Time/60) + " mins"  
    json_object = json.dumps(Eval_Log, indent=4)
    
    # Writing 
    with open(args.evaldir+'/Eval_Log.json', "w") as outfile:
        outfile.write(json_object)    


    #Saving metrics---------------------------------------------
    # import pickle 
    # with open(args.evaldir+'/scores.pickle', 'wb') as handle:
    #     pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)    



a = 0
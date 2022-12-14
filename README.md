# predictive_maintenance
Predictive Maintenance AI framework
# 1. greate DataSet from .wav files
python3 Create_Dataset.py -dataset-params "{ Wav_folder : c:\ML1\ML_pipeline_code_multiple_classes_v1\tst_dataset , Target_folder : c:\ML1\ML_pipeline_code_multiple_classes_v1\Spectrograms_tst , Fs : 44100 , seq_dur : 5 , classes_lookup : { Moter_single : 0 , Motor_gran_no_chain : 1 , seatrak_all_elements : 2 , kinhsh_koble : 3 } , FE_params : { front_end_name : STFT_custom , a : 768 , M : 1024 , support : 1024 } , preproc : None } "

# 2. train model
python3 train.py --root c:\ML1\ML_pipeline_code_multiple_classes_v1\Spectrograms_tst --nb_classes 4 --output c:\ML1\ML_pipeline_code_multiple_classes_v1\Spectrograms_tst\pretr_model --epochs 10

# 3. evaluate
python3 evaluate.py --method-name TST --Model_dir c:\ML1\ML_pipeline_code_multiple_classes_v1\Spectrograms_tst/pretr_model --root_TEST_dir c:\ML1\ML_pipeline_code_multiple_classes_v1\tst_dataset --evaldir c:\ML1\ML_pipeline_code_multiple_classes_v1\Spectrograms_tst\evaldir

# 4. inference
python3 inference.py --Model_dir c:\ML1\ML_pipeline_code_multiple_classes_v1\Spectrograms_tst\pretr_model --input-wav c:\ML1\ML_pipeline_code_multiple_classes_v1\tst_dataset\test\kinhsh_koble\κίνηση-κομπλε-3.wav  

# 5. inference real time
python3 inference_real_time.py --Model_dir c:\ML1\ML_pipeline_code_multiple_classes_v1\Spectrograms_tst\pretr_model

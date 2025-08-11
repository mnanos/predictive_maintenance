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


#0) in order to install all the libraries we want:
pip install -r requirements.txt


##To run an experiment, follow these commands in sequence:
--------------------------------------------------------

/home/user/ML_pipeline_code_multiple_classes_v0 instead of  /home/mnanos/ML_pipeline_code_multiple_classes_v0/

#1)DATA
-----------------------------------------------------------------------------------------

TODO:
	-

COMMAND:
python3 Create_Dataset.py -dataset-params "{ Wav_folder : /home/mnanos/ML_pipeline_code_multiple_classes_v0/tst_dataset , Target_folder : /home/mnanos/ML_pipeline_code_multiple_classes_v0/Spectrograms_tst , Fs : 14700 , seq_dur : 5 , classes_lookup : { Moter_single : 0 , Motor_gran_no_chain : 1 , seatrak_all_elements : 2 , kinhsh_koble : 3 } , FE_params : { front_end_name : STFT_custom , a : 768 , M : 1024 , support : 1024 } , preproc : None } "

Parameter explaination:
    Wav_folder-> (STR) 	This is the PATH of the directory, which must have the following structure; the files are from Audacity:

            Wav_folder   --+--   train --+-- ith_class_wav  --+-- όλα τα .wav αρχεία τα οποία θεωρείς ότι είναι στην ι-οστη κλάση
                           |                          
                           |                          
                           |     
                           +--   valid --+-- ith_class_wav  --+-- όλα τα .wav αρχεία τα οποία θεωρείς ότι είναι στην ι-οστη κλάση                           
                           |                          
                           |                          
                           |     
                           +--   test  --+-- ith_class_wav  --+-- όλα τα .wav αρχεία τα οποία θεωρείς ότι είναι στην ι-οστη κλάση                  
                           
                           
                                                                                                  
    
    Target_folder-> (STR)	 Είναι το PATH του dir (το οποίο δημιουργείται αν δεν υπάρχει) και στο οποίο θα αποθηκευτούν τα ακόλουθα: 
					    Spec_seg_pair_list_train.pt: η μεταβλητή (iterable) η οποία θα περιέχει τα παραδείγματα training ζεύγη (In_spectr,label)
					    Spec_seg_pair_list_valid.pt: η μεταβλητή (iterable) η οποία θα περιέχει τα παραδείγματα validation ζεύγη (In_spectr,label)
					    Dataset_Params_log.json: Log το οποίο περιέχει τις παραμέτρους του Dataset
                    

    Fs-> (ΙΝΤ) Συχνότητα δειγματοληψίας στην οποία γίνονται resample τα wavs     
					(ΧΡΗΣΗ: σε περίπτωση που έχεις δει ότι οι κυματομορφές δεν 
                    περιέχουν ενέργεια πάνω από μια συχνότητα μπορείς να κάνεις resampling σε αυτή για γρηγορότερο processing)

    seq_dur-> (ΙΝΤ) 		Η διάρκεια της ακολουθίας (σε sec) των παραδειγμάτων εκπαίδευσης τα οποία τροφοδοτούμε στο δίκτυο. 
					(ΧΡΗΣΗ: επειδή τα wavs που έχουμε είναι μεγάλης διάρκειας (π.χ. 30mins) 
                    σε blocks διάρκειας seq-dur=5sec ώστε 1)να γίνεται γρηγορότερα το processing ,
                                                          2)Να έχουμε περισσότερα παραδείγματα εκπαίδευσης ) 

    FE_params-> (dictionary) 		Είναι οι παράμετροι του FE (front end ή αναπαράσταση εισόδου) με το οποίο θα τροφοδοτούμε το δίκτυο .     
					Επιλέξαμε το FE να είναι spectrogram για να εκμαιταλευτούμε τα CNN νευρωνικά δίκτυα καθώς αυτά 
					τα πάνε πολύ καλά με εικόνες και τα spectrograms είναι εικόνες.
	   
   
    classes_lookup-> (dictionary) 	 Είναι ένα lookup table όπου : 
    						key -> είναι το όνομα του dir μιας κλάσης (είναι string)
    						value -> είναι το αντίστοιχο label (είναι int)
    						
    		    *Σημαντική Σημείωση: Θα πρέπει το key value pair που αντιστοιχεί 
    		     στην κλάση με label=0 να γράφεται πρώτο στο classes_lookup dictionary!!!! 
    		     (για λόγους υλοποίησης)     
    						          
                                          

#2)TRAIN
------------------------------------------------------------------------------------------------------------------------------------------------------------------------

TODO:
	
ΕΝΤΟΛΗ:
python3 train.py --root /home/mnanos/ML_pipeline_code_multiple_classes_v0/Spectrograms_tst --nb_classes 4 --output /home/mnanos/ML_pipeline_code_multiple_classes_v0/Spectrograms_tst/pretr_model --epochs 10

ΕΠΕΞΗΓΗΣΗ ΠΑΡΑΜΕΤΡΩΝ: 
    root-> (STR)  Είναι το PATH του dir το οποίο περιέχει τα αρχεία : 
		   Spec_seg_pair_list_train.pt,Spec_seg_pair_list_valid.pt,Dataset_Params_log.json
		   και το οποίο δημιουργήθηκε με το προηγούμενο script (Create_Dataset.py). 
		   Ως εκ τούτου το path αυτό θα πρέπει να είναι ίδιο με το Target_folder (παράμετρος του  Create_Dataset.py)


    output-> (STR) Είναι το PATH του dir (το οποίο δημιουργείται αν δεν υπάρχει) και στο οποίο θα αποθηκευτούν τα ακόλουθα αρχεία:
		     model.pth-> Αρχείο απαραίτητο αν θες να χρησιμοποιήσεις το μοντέλο για inference ή για evalution
		     model.json-> Αρχείο Log το οποίο περιέχει στοιχεία για την εκπαίδευση (π.χ. trainig-validation losses, execution time, 
			 Dataset parameters, arguments του train.py script )
		     model.chkpnt-> Αρχείο απαραίτητο αν θές να κάνεις συνεχίσεις την εκπαίδευση ενός ήδη εκπαιδευμένου μοντέλου από εκεί που είχε σταματήσει
             
    nb_classes-> Είναι το πλήθος των κλάσεων για τις οποίες θα εκπαιδευτεί το NN.

Βασικές Υπερπαράμετροι (hyperparameters) εκπαίδευσης:

    epochs-> (INT) Εποχές τις οποίες θα εκπαιδευτεί το Νευρωνικό δίκτυο 


    batch-size-> (INT)	Το μέγεθος του batch με το οποίο τροφοδοτούμε το δίκτυο 
		        (το πλήθος των παραδειγμάτων που του δίνουμε ώστε μετά να κάνει backprop).
		        Όσο μεγαλύτερο είναι +Με τη διαδικασία εκπαίδευσης θα βρεθεί σίγουρα ένα τοπικό ελάχιστο
		                             +Γρηγορότερη επεξεργασία καθώς εκμαιταλευόμαστε περισσότερο την GPU
		                             -Περισσότερες απαιτήσεις σε μνήμη

    Υπάρχουν και άλλες αλλά καλό θα ήταν να αφεθούν στις Default τιμές:)
                
    

#3)EVALUATION
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

TODO:
	
ΕΝΤΟΛΗ:
python3 evaluate.py --method-name TST --Model_dir /home/mnanos/ML_pipeline_code_multiple_classes_v0/Spectrograms_tst/pretr_model --root_TEST_dir /home/mnanos/ML_pipeline_code_multiple_classes_v0/tst_dataset --evaldir /home/mnanos/ML_pipeline_code_multiple_classes_v0/Spectrograms_tst/evaldir


ΕΠΕΞΗΓΗΣΗ ΠΑΡΑΜΕΤΡΩΝ:
    method-name->(STR)   Είναι το όνομα της μεθόδου που κάνουμε evaluate π.χ. μπορεί να θές να 
		          συγκρίνεις διάφορα FEs εισόδου ή διάφορες αρχιτεκτονικές δικτύων .
		          Αυτό το όνομα θα φαίνεται στα Logs (μπορείς και να μην το θέσεις).
		          
    Model_dir->(STR)	 Είναι το path για το dir που περιέχει τα απαραίτητα αρχείο για το pretrained μοντέλο.
               	 (θα πρέπει να είναι το ίδιο με το argument output του train.py script)

    root_TEST_dir->(STR)     Είναι το path για το dir που περιέχει τα testing wavs και πρέπει να έχει την δομή που φαίτεται παραπάνω
                                                        


    evaldir->(STR)	 Είναι το PATH του dir (το οποίο δημιουργείται αν δεν υπάρχει) και στο οποίο θα αποθηκευτούν τα ακόλουθα αρχεία:
			      Eval_Log.json-> Περιέχει τις παραμέτρους του script και τις μετρικές αξιολόγησης
			      scores.pickle-> Περιέχει τις μετρικές αξιολόγησης σε μια python3 μεταβλητή   


#4)INFERENCE
--------------------------------------------------------------------------------------------------------------------------------------------------------

TODO:
	-


ΕΝΤΟΛΗ:
python3 inference.py --Model_dir /home/mnanos/ML_pipeline_code_multiple_classes_v0/Spectrograms_tst/pretr_model --input-wav /home/mnanos/ML_pipeline_code_multiple_classes_v0/tst_dataset/test/kinhsh_koble/κίνηση-κομπλε-3.wav  


ΕΠΕΞΗΓΗΣΗ ΠΑΡΑΜΕΤΡΩΝ:
    Model_dir->(STR)	 Είναι το path για το dir που περιέχει τα απαραίτητα αρχεία για το pretrained μοντέλο.
                	(θα πρέπει να είναι το ίδιο με το argument output του train.py script)

    input-wav->(STR) 	Είναι το path για έναν ήχο για τον οποίο θες να πάρεις απάντηση 


Για να συνεχίσουμε την εκπαίδευση εκτελούμε την παρακάτω εντολή
--------------------------------------------------------------- 
python3 train.py --model /home/mnanos/ML_pipeline_code_multiple_classes_v0/Spectrograms_tst/pretr_model --checkpoint /home/mnanos/ML_pipeline_code_multiple_classes_v0/Spectrograms_tst/pretr_model /home/mnanos/ML_pipeline_code_multiple_classes_v0/Spectrograms_tst --nb_classes 4 --output /home/mnanos/ML_pipeline_code_multiple_classes_v0/Spectrograms_tst/pretr_model --epochs 10 


*ΣΗΜΑΝΤΙΚΕΣ ΣΗΜΕΙΩΣΕΙΣ-ΣΧΟΛΙΑ: 

	DATASET:
		Το Dataset (WAVs) δημιουργήθηκκε με τη βοήθεια του λογισμικού Audacity. 
		Ακολουθήθηκε η διαδικασία: 
				Αρχικά να σημειωθεί ότι μια κλάση που το αντίστοιχο wav ήταν 22secs δεν την έλαβα υπόψιν για αυτό μείναμε με 4 κλάσεις. 
				Για τις υπόλοιπες 4 κλάσεις κόψαμε τα WAVs που είχαμε για την κάθε μια ώστε να τα βάλουμε κατάλληλα στα 3 Folders (test,train,valid).
				Διάρκειες των αντίστοιχων WAVs -> Train>Test>valid. 
				
				H παραπάνω διαδικασία μπορεί εύκολα:
					1)Να επεκταθεί για περισσότερα του ενός WAV ανά κλάση.
					2)Να επεκταθεί για περισσότερες κλάσεις.

		    
	ΝΕΥΡΩΝΙΚΟ ΔΙΚΤΥΟ:
		Το δίκτυο προβλέπει για ένα block διάρκειας seq_dur=5sec σε ποια κλάση (από τις 4) ανήκει. Η έξοδος του δικτύου είναι ένα μητρώο διάστασης (nb_blocks,nb_classes) 		    
		και η κάθε του γραμμή είναι οι unormalized αποκρίσεις για τις 4 κλάσεις και για ένα block. To objective function που χρησιμοποιούμε για ελαχιστοποίηση είναι 		    
		το CrossEntropyLoss https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html όπου χρησιμοποιούμε και weights για την κάθε κλάση γιατί το 			    
		dataset είναι unbalanced. 
		
nb_blocks = number of blocks  --> signal duration/seq_dur
nb_classes = 4 είδος ήχου που θέλουμε να αναγνωρίσουμε
		     
	EVALUATION και INFERENCE:
		Αρχικά να τονίσουμε ότι το inference και το evaluation γίνονται κατά blocks διάρκειας seq_dur όπως στην εκπαίδευση.
		    
		    
		EVALUATION:
		    Στο evaluation αξιολογούμε την απόδοση του δικτύου στο πόσο καλά κατηγοριοποιεί τα blocks που του δίνουμε στις διάφορες κλάσεις. Για να αποφανθούμε σε πια 			    
			κλάση ανήκει ένα block:
		    Εφαρμόζουμε στην unormalized αποκρίση του δικτύου (διάνυσμα 4 στοιχείων) τη softmax() μη-γραμμικότητα. 
			Αυτό το βήμα το κάνουμε έτσι ώστε να μοντελοποιήσουμε (μετατρέψουμε) αυτές τις unormalized αποκρίσεις σε 4 πιθανότητες. 
			Ο δείκτης της μεγαλύτερης πιθανότητας (του διανύσματος) αποτελεί και την κλάση που απαντάει το δίκτυο για το συγκεκριμένο block.   
		    
		INFERENCE:
			Στο inference για να αποφανθούμε για το σε πια κλάση ανήκει το input-wav εκτελούμε το ίδιο βήμα με αυτό στο evaluation στην έξοδο του δικτύου. 
			Εδώ όμως θέλουμε να αποφασίσουμε σε πια κλάση ανήκει όλο το input-wav και όχι απλά τα blocks του. Επομένως αποφασίζουμε υπέρ της κλάσης με τα περισσότερα 			    
			occurences, δηλαδή σαν να συμβουλευόμαστε ένα ιστόγραμμα (Θα μπορούσαμε να πούμε ότι η κλάση που ανήκει ένα block είναι τυχαία μεταβλητή και μέσω αυτού του 		    
			ιστογράμματος υπολογίζουμε τη κατανομή αυτης της τυχαίας μεταβλητής).     
		    

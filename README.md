# OMG_Challenge

This repository contains: train and evaluation scripts (train_script.py and eval_script.py), their models (AffWildNet.py and AffWildNet_valid.py) and a script used by the train and evaluation scripts for processing the data (data_process.py) .
To be more exact each script has flags:
- train_script: 
               initial_learning_rate 
               concordance_loss : if 1 use concordance as loss function, else if 0: use MSE as loss function
               batch_size
               seq_length
               size : size of input images, e.g. if is set to 96 then input_images_size = 96x96
               h_units
               network : which network to use, pick between: "CNN_GRU_1RNN", "CNN_GRU_3RNN"  
               input_file
               train_dir
               pretrained_model_checkpoint_path 


- eval_script: 
               batch_size
               seq_length
               size : size of input images, e.g. if is set to 96 then input_images_size = 96x96
               h_units
               network : which network to use, pick between: "CNN_GRU_1RNN", "CNN_GRU_3RNN"  
               input_file
               pretrained_model_checkpoint_path 




Dependencies:

numpy : we are using version 1.13.1
tensorflow: we are using version 1.1.0 
(we also use tensorflow.contrib.slim : slim is part of tensorflow after version 1.0)


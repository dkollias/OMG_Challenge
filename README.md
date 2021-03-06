# OMG_Challenge

This repository contains our solution to the OMG-Emotion Challenge 2018 -using only visual data- that ranked 2nd for vision-only valence estimation and 2nd for overall valence estimation.

If you use our scripts please cite the papers: 

1) A Multi-component CNN-RNN Approach for Dimensional Emotion Recognition in-the-wild

>@article{kollias2018multi,
  title={A Multi-component CNN-RNN Approach for Dimensional Emotion Recognition in-the-wild},
  author={Kollias, Dimitrios and Zafeiriou, Stefanos},
  journal={arXiv preprint arXiv:1805.01452},
  year={2018}
}

2) Exploiting multi-CNN features in CNN-RNN based Dimensional Emotion Recognition on the OMG in-the-wild Dataset

>@article{kollias2019exploiting,
  title={Exploiting multi-cnn features in cnn-rnn based dimensional emotion recognition on the omg in-the-wild dataset},
  author={Kollias, Dimitrios and Zafeiriou, Stefanos},
  journal={arXiv preprint arXiv:1910.01417},
  year={2019}
}

This repository contains: train and evaluation scripts (train_script.py and eval_script.py), their models (AffWildNet.py and AffWildNet_valid.py) and a script used by the train and evaluation scripts for processing the data (data_process.py) .
To be more exact each script has flags (one can find more detailed explanation withing each script) :
- train_script flags: 

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


- eval_script flags: 

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


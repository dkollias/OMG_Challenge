from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf   ## we are using tensorflow version 1.1.0
import AffWildNet_valid as  AffWildNet
import data_process
import numpy as np

slim = tf.contrib.slim


# Create FLAGS
FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_integer('batch_size', 1, '''The batch size to use.''')

tf.app.flags.DEFINE_integer('seq_length', 80, 'the sequence length: how many consecutive frames to use for the RNN')

tf.app.flags.DEFINE_integer('size', 96, 'dimensions of input images, e.g. 96x96')

tf.app.flags.DEFINE_integer('h_units', 128, 'the hidden units of each of the rnn layers, use 128 for CNN_GRU_1RNN network or 256 for CNN_GRU_3RNN network ')

tf.app.flags.DEFINE_string('network',  'CNN_GRU_1RNN' , ' which network architecture we want to use,  pick between : CNN_GRU_1RNN, CNN_GRU_3RNN '     )                           

tf.app.flags.DEFINE_string('input_file',  '/homes/input.csv' , 'the input file : it should be in the format: image_file_location,valence_value,arousal_value  and images should be jpgs'     )                           


tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '/homes//homes/model.ckpt-16115',
                           '''the pretrained model checkpoint path to restore,if there exists one  '''
                           '''''')



###############################################################################################################################################################
####  The sample code and the model weights are for RESEARCH PURPOSES only and cannot be used for commercial use.      ########################################
####                                 Do not redistribute this elsewhere                                                ########################################
################################################################################################################################################################



def evaluate():
  g = tf.Graph()
  with g.as_default():


    image_list, label_list = data_process.read_labeled_image_list(FLAGS.input_file)
    # split into sequences
    image_list, label_list = data_process.make_rnn_input_per_seq_length_size(image_list,label_list,FLAGS.seq_length)

    images = tf.convert_to_tensor(image_list)
    labels = tf.convert_to_tensor(label_list)

    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, labels,images],num_epochs=None, shuffle=False, seed=None,capacity=1000, shared_name=None, name=None)
    images_batch, labels_batch, image_locations_batch = data_process.decodeRGB(input_queue,FLAGS.seq_length,FLAGS.size)
    images_batch = tf.to_float(images_batch)
    images_batch -= 128.0
    images_batch /= 128.0  # scale all pixel values in range: [-1,1]

    images_batch = tf.reshape(images_batch,[-1,96,96,3])
    labels_batch = tf.reshape(labels_batch,[-1,2])
    
    if FLAGS.network == 'CNN_GRU_1RNN':
     network = AffWildNet.CNN_GRU_1RNN(FLAGS.seq_length,FLAGS.batch_size,FLAGS.h_units)
    elif FLAGS.network == 'CNN_GRU_3RNN':
     network = AffWildNet.CNN_GRU_3RNN(FLAGS.seq_length,FLAGS.batch_size,FLAGS.h_units)

    network.setup(images_batch)
    prediction = network.get_output()


    num_batches = int(len(image_list)/FLAGS.batch_size)


    variables_to_restore =  tf.global_variables()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5

    with tf.Session() as sess:

         init_fn = slim.assign_from_checkpoint_fn(
                        FLAGS.pretrained_model_checkpoint_path, variables_to_restore,
                        ignore_missing_vars=False)

         init_fn(sess)
         print('Loading model {}'.format(FLAGS.pretrained_model_checkpoint_path))


         tf.train.start_queue_runners(sess=sess)

         coord = tf.train.Coordinator()
 
 
         evaluated_predictions = []
         evaluated_labels = []
         images = []
 
         try:
             for _ in range(num_batches):

                 pr, l,imm = sess.run([prediction,labels_batch, image_locations_batch])
                 evaluated_predictions.append(pr)
                 evaluated_labels.append(l)
                 images.append(imm)
 
                 if coord.should_stop():
                     break
             coord.request_stop()
         except Exception as e:
             coord.request_stop(e)

         predictions = np.reshape(evaluated_predictions, (-1, 2))
         labels = np.reshape(evaluated_labels, (-1, 2))
         images = np.reshape(images, (-1))

         conc_arousal = concordance_cc2(predictions[:,1], labels[:,1])
         conc_valence = concordance_cc2(predictions[:,0], labels[:,0])
 
         print('Concordance on valence : {}'.format(conc_valence))
         print('Concordance on arousal : {}'.format(conc_arousal))
         print('Concordance on total : {}'.format((conc_arousal+conc_valence)/2))

         mse_arousal = sum((predictions[:,1] - labels[:,1])**2)/len(labels[:,1])
         print('MSE Arousal : {}'.format(mse_arousal))
         mse_valence = sum((predictions[:,0] - labels[:,0])**2)/len(labels[:,0])
         print('MSE Valence : {}'.format(mse_valence))
        

  
    

    return conc_valence, conc_arousal, (conc_arousal+conc_valence)/2, mse_arousal, mse_valence
 
def concordance_cc2(r1, r2):
     mean_cent_prod = ((r1 - r1.mean()) * (r2 - r2.mean())).mean()
     return (2 * mean_cent_prod) / (r1.var() + r2.var() + (r1.mean() - r2.mean()) ** 2)
 


if __name__ == '__main__':
    evaluate()


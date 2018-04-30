from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf   ## we are using tensorflow version 1.1.0
import AffWildNet 
import data_process
from tensorflow.python.platform import tf_logging as logging

slim = tf.contrib.slim


# Create FLAGS
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('initial_learning_rate', 0.001, 'Initial learning rate.')

tf.app.flags.DEFINE_float('concordance_loss', 1, ' defines which loss function to use: if set to 0, then the mean squarred error will be the cost function, else the concordance correlation coefficient ')


tf.app.flags.DEFINE_integer('batch_size', 10, '''The batch size to use.''')

tf.app.flags.DEFINE_integer('seq_length', 80, 'the sequence length: how many consecutive frames to use for the RNN')

tf.app.flags.DEFINE_integer('size', 96, 'dimensions of input images, e.g. 96x96')

tf.app.flags.DEFINE_integer('h_units', 128, 'the hidden units of each of the rnn layers, use 128 for CNN_GRU_1RNN network or 256 for CNN_GRU_3RNN network ')

tf.app.flags.DEFINE_string('network',  'CNN_GRU_1RNN' , ' which network architecture we want to use,  pick between : CNN_GRU_1RNN, CNN_GRU_3RNN '     )                           

tf.app.flags.DEFINE_string('input_file',  '/homes/input.csv' , 'the input file : it should be in the format: image_file_location,valence_value,arousal_value  and images should be jpgs'     )                           


tf.app.flags.DEFINE_string('train_dir', '/homes/train_dir',
                           '''the directory to save the model checkpoints, weights and event files  '''
                           '''''')


tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '/homes/model.ckpt-16115',
                           '''the pretrained model checkpoint path to restore,if there exists one  '''
                           '''''')


###############################################################################################################################################################
####  The sample code is for RESEARCH PURPOSES only and cannot be used for commercial use.      ########################################
####                                 Do not redistribute this elsewhere                                                ########################################
################################################################################################################################################################



def train():
  g = tf.Graph()
  with g.as_default():

    image_list, label_list = data_process.read_labeled_image_list(FLAGS.input_file)
    # split into sequences
    image_list, label_list = data_process.make_rnn_input_per_seq_length_size(image_list,label_list,FLAGS.seq_length)

    images = tf.convert_to_tensor(image_list)
    labels = tf.convert_to_tensor(label_list)

    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, labels,images],num_epochs=None, shuffle=True, seed=None,capacity=1000, shared_name=None, name=None)
    images_sequence, labels_sequence, image_locations_sequence = data_process.decodeRGB(input_queue,FLAGS.seq_length,FLAG.size)
    images_sequence = tf.to_float(images_sequence)
    images_sequence -= 128.0
    images_sequence /= 128.0  # scale all pixel values in range: [-1,1]

    images_batch, labels_batch, image_locations_batch = tf.train.shuffle_batch(
                          [images_sequence, labels_sequence, image_locations_sequence],
                          batch_size=FLAGS.batch_size,
                          min_after_dequeue=100,  
                          num_threads=1,
                          capacity=1000)

    images_batch = tf.reshape(images_batch,[-1,96,96,3])
    
    labels_batch = tf.reshape(labels_batch,[FLAGS.batch_size,FLAGS.seq_length,2])
    
    if FLAGS.network == 'CNN_GRU_1RNN':
     network = AffWildNet.CNN_GRU_1RNN(FLAGS.seq_length,FLAGS.batch_size,FLAGS.h_units)
    elif FLAGS.network == 'CNN_GRU_3RNN':
     network = AffWildNet.CNN_GRU_3RNN(FLAGS.seq_length,FLAGS.batch_size,FLAGS.h_units)

    network.setup(images_batch)
    prediction = network.get_output()
   

    prediction = tf.reshape(prediction,[FLAGS.batch_size,FLAGS.seq_length,2])
    for i, name in enumerate(['valence','arousal']):
      preds = []
      labs = []
      for j in range(FLAGS.batch_size):
            pred_single = tf.reshape(prediction[j, :, i], (-1,))
            gt_single = tf.reshape(labels_batch[j, :, i], (-1,))
            preds.append(tf.reduce_mean(pred_single))
            labs.append(tf.reduce_mean(gt_single))
      preds = tf.convert_to_tensor(preds)
      labs = tf.convert_to_tensor(labs)
      if FLAGS.concordance_loss:
              loss = concordance_cc2(preds, labs)
      else:
              loss = tf.reduce_mean(tf.square(preds - labs))
      slim.losses.add_loss(loss / 2.)       
    

    total_loss = slim.losses.get_total_loss()
    optimizer = tf.train.AdamOptimizer(FLAGS.initial_learning_rate)

    
    ## if you want to restore only a subset of the weights/biases, replace tf.global_variables() with another subset 
    variables_to_restore =  tf.global_variables()


  with tf.Session(graph=g) as sess:
        if FLAGS.pretrained_model_checkpoint_path:
            init_fn = slim.assign_from_checkpoint_fn(
                        FLAGS.pretrained_model_checkpoint_path, variables_to_restore,
                        ignore_missing_vars=True)
        else:
             init_fn = None  

        ## here in variables_to_train I have declared all weights and biases, if you want to train only a subset then change accordingly
        train_op = slim.learning.create_train_op(total_loss,
                                                 optimizer,
                                                 variables_to_train = tf.global_variables(),                                                 
                                                 summarize_gradients=True)
        logging.set_verbosity(1)


        slim.learning.train(train_op,
                            FLAGS.train_dir,
                            init_fn=init_fn,
                            save_summaries_secs=600*360,
                            log_every_n_steps=500,
                            save_interval_secs=60*15)
        


def concordance_cc2(predictions, labels):
    pred_mean, pred_var = tf.nn.moments(predictions, (0,))
    gt_mean, gt_var = tf.nn.moments(labels, (0,))
    mean_cent_prod = tf.reduce_mean((predictions - pred_mean) * (labels - gt_mean))

    return 1 - (2 * mean_cent_prod) / (pred_var + gt_var + tf.square(pred_mean - gt_mean))

   


if __name__ == '__main__':
    train()

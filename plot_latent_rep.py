import tensorflow as tf
from tensorflow import keras
import numpy as np
import io
import sklearn.metrics
from tensorboard.plugins import projector
import cv2
import os
import shutil
import argparse
from tensorflow.contrib.tensorboard.plugins import projector


def save_embeddings(images_features_labels, save_dir):
    '''
    Function to save embeddings (with corresponding labels and images) to a
        specified directory. Point tensorboard to that directory with
        tensorboard --logdir=<save_dir> and your embeddings will be viewable.
    Arguments:
    images_features_labels : dict
        each key in the dict should be the desired name for that embedding, and 
        each element should be a list of [images, embeddings, labels] where 
        images are a numpy array of images between 0. and 1. of shape [N*W*H*D] 
        or [N*H*W] if grayscale (or None if no images), embeddings is a numpy 
        array of shape [N*D], and labels is a numpy array of something that can
        be converted to string of shape D (or None if no labels available)
    save_dir : str
        path to save tensorboard checkpoints
    '''
    assert len(list(images_features_labels.keys())), 'Nothing in dictionary!'
    
    # Make directory if necessary
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Reset graph and initialise file writer and session
    tf.reset_default_graph()
    writer = tf.summary.FileWriter(os.path.join(save_dir), graph=None)
    sess = tf.Session()
    config = projector.ProjectorConfig()

    # For each embedding name in the provided dictionary of embeddings
    for name in list(images_features_labels.keys()):
    
        [ims, fts, labs] = images_features_labels[name]
        
        # Save sprites and metadata
        if labs is not None:
            metadata_path = os.path.join(save_dir, name + '-metadata.tsv')
            save_metadata(labs, metadata_path)
        if ims is not None:
            sprites_path = os.path.join(save_dir, name + '.png')
            save_sprite_image(ims, path=sprites_path, invert=len(ims.shape)<4)
        
        # Make a variable with the embeddings we want to visualise
        embedding_var = tf.Variable(fts, name=name, trainable=False)
        
        # Add this to our config with the image and metadata properties
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        if labs is not None:
            embedding.metadata_path = metadata_path
        if ims is not None:
            embedding.sprite.image_path = sprites_path
            embedding.sprite.single_image_dim.extend(ims[0].shape)
    
        # Save the embeddings
        projector.visualize_embeddings(writer, config)
    saver = tf.train.Saver(max_to_keep=1)
    sess.run(tf.global_variables_initializer())
    saver.save(sess, os.path.join(save_dir, 'ckpt'))


''' Functions below here inspired by / taken from:
http://www.pinchofintelligence.com/simple-introduction-to-tensorboard-embedding-visualisation/'''

def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. 
       Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))    
    if len(images.shape) > 3:
        spriteimage = np.ones(
            (img_h * n_plots, img_w * n_plots, images.shape[3]))
    else:
        spriteimage = np.ones((img_h * n_plots, img_w * n_plots))
    four_dims = len(spriteimage.shape) == 4
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                if four_dims:
                    spriteimage[i * img_h:(i + 1) * img_h,
                      j * img_w:(j + 1) * img_w, :] = this_img
                else:
                    spriteimage[i * img_h:(i + 1) * img_h,
                      j * img_w:(j + 1) * img_w] = this_img
    return spriteimage
    

def save_sprite_image(to_visualise, path, invert=True):
    if invert:
        to_visualise = invert_grayscale(to_visualise)
    sprite_image = create_sprite_image(to_visualise)


def invert_grayscale(mnist_digits):
    """ Makes black white, and white black """
    return 1-mnist_digits


def save_metadata(batch_ys, metadata_path):
    with open(metadata_path,'w') as f:
        f.write("Index\tLabel\n")
        for index,label in enumerate(batch_ys):
            if type(label) is int:
                f.write("%d\t%d\n" % (index, label))
            else:
                f.write('\t'.join((str(index), str(label))) + '\n')



class FileWriter():
    def __init__(self, path):
        self.path = path

    def get_logdir(self):
        return self.path


def plot_to_projector(
    feature_vector,
    y,
    class_names,
    log_dir="default_log_dir",
    meta_file="metadata.tsv",
):
    # Generate label names
    labels = [class_names[int(y[i])] for i in range(int(y.shape[0]))]

    with open(os.path.join(log_dir, meta_file), "w") as f:
        for label in labels:
            f.write("{}\n".format(label))

    if feature_vector.ndim != 2:
        print(
            "NOTE: Feature vector is not of form (BATCH, FEATURES)"
            " reshaping to try and get it to this form!"
        )
        feature_vector = tf.reshape(feature_vector, [feature_vector.shape[0], -1])

    feature_vector = tf.convert_to_tensor(feature_vector)
    print(type(feature_vector))

    feature_vector = tf.Variable(feature_vector)
    checkpoint = tf.train.Checkpoint(embedding=feature_vector)
    checkpoint.save(os.path.join(log_dir, "embeddings.ckpt"))

    # Set up config
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = meta_file
    print(log_dir)
    projector.visualize_embeddings(FileWriter(log_dir), config)


def proj_z():
    """
    Ideas:
        - Plot space of Z with different fairness coeffs, and show clusters of protected variable being more dispersed
    """
    # Sample args: experiments/full_sweep_adult/data--adult--model_class-WeightedDemParWassGan--model_fair_coeff-0_1/checkpoints
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint_path', type=str)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--folder_name', type=str)
    args = parser.parse_args()

    npz_path = f"{args.model_checkpoint_path}/Epoch_{args.epoch}_Valid/npz"
    Z_path = f"{npz_path}/Z.npz"
    A_path = f"{npz_path}/A.npz"

    Z = np.load(Z_path, mmap_mode='r')
    A = np.load(A_path, mmap_mode='r')

    Z = Z['X']
    A = A['X']

    features_labels = {}
    features_labels['x'] = [None, Z, A]

    save_embeddings(features_labels, args.folder_name)


def proj_x():
    """
    Ideas:
        - Plot PCA space of X
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_path', type=str)
    parser.add_argument('--folder_name', type=str)
    args = parser.parse_args()

    X_path = args.x_path
    X = np.load(X_path, mmap_mode='r')
    Y = X['y_train']
    X = X['x_train']
    
    features_labels = {}
    features_labels['x'] = [None, X, Y]
    save_embeddings(features_labels, args.folder_name)


def proj_z_debiasing():
    """
    Ideas:
        - Plot space of Z with different fairness coeffs, and show clusters of protected variable being more dispersed
    """
    # Sample args: experiments/full_sweep_adult/data--adult--model_class-WeightedDemParWassGan--model_fair_coeff-0_1/checkpoints
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str)
    parser.add_argument('--coef', type=str)
    parser.add_argument('--folder_name', type=str)
    args = parser.parse_args()

    Z_path = f"{args.file_path}/Z_{args.coef}.npz"
    A_path = f"{args.file_path}/A_{args.coef}.npz"

    Z = np.load(Z_path, mmap_mode='r')
    A = np.load(A_path, mmap_mode='r')

    Z = Z['data']
    A = A['data']

    features_labels = {}
    features_labels['x'] = [None, Z, A]

    save_embeddings(features_labels, args.folder_name)


proj_z_debiasing()
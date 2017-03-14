import data_utils as utils
 
import os
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
import random
import pickle
 
import tree_gru
 
DIR = '.../treelstm/data/sst'
GLOVE_DIR ='.../treelstm/data/glove'
 
import pdb
import time

class Config(object):
 
    num_emb=None
 
    emb_dim = 300
    hidden_dim = 150
    output_dim=None
    degree = 2
 
    num_epochs = 25
    early_stopping = 2
    dropout = 0.5
    lr = 0.05
    emb_lr = 0.1
    reg=0.0001
 
    batch_size = 25
    #num_steps = 10
    maxseqlen = None
    maxnodesize = None
    fine_grained=False
    trainable_embeddings=True
    nonroot_labels=True
 
def train(restore=False):
 
    config=Config()
 
 
    data,vocab = utils.load_sentiment_treebank(DIR,config.fine_grained)
 
    train_set, dev_set, test_set = data['train'], data['dev'], data['test']
    print 'train', len(train_set)
    print 'dev', len(dev_set)
    print 'test', len(test_set)
 
    num_emb = len(vocab)
    num_labels = 5 if config.fine_grained else 3
    for _, dataset in data.items():
        labels = [label for _, label in dataset]
        assert set(labels) <= set(xrange(num_labels)), set(labels)
    print 'num emb', num_emb
    print 'num labels', num_labels
 
    config.num_emb=num_emb
    config.output_dim = num_labels
 
    config.maxseqlen=utils.get_max_len_data(data)
    config.maxnodesize=utils.get_max_node_size(data)
 
    print config.maxnodesize,config.maxseqlen ," maxsize"
    #return 
    random.seed()
    np.random.seed()
 
 
    with tf.Graph().as_default():
        model = tree_gru.tf_NarytreeGRU(config)
        init=tf.initialize_all_variables()
        saver = tf.train.Saver()
        best_valid_score=0.0
        best_valid_epoch=0
        dev_score=0.0
        test_score=0.0
        with tf.Session() as sess:
 
            sess.run(init)
            start_time=time.time()
 
            if restore:saver.restore(sess,'.../weights')
            results_b_gru = pd.DataFrame(columns=['time','train_loss','train','dev','test'], index = [np.arange(0,config.num_epochs,1)])
             
            for epoch in range(config.num_epochs):
                print 'epoch', epoch
                avg_loss=0.0
                avg_loss = train_epoch(model, train_set,sess)
                results_b_gru.iloc[epoch,1] = avg_loss
                print 'avg loss', avg_loss
 
                train_score=evaluate(model,train_set,sess)
                results_b_gru.iloc[epoch,2] = train_score
                print 'train_set-scoer', train_score
 
                dev_score=evaluate(model,dev_set,sess)
                results_b_gru.iloc[epoch,3] = dev_score
                print 'dev-scoer', dev_score
 
                if dev_score > best_valid_score:
                    best_valid_score=dev_score
                    best_valid_epoch=epoch
                    saver.save(sess,'.../weights')
 
                if epoch -best_valid_epoch > config.early_stopping:
                    break
 
                print "time per epochis {0}".format(
                    time.time()-start_time)
                results_b_gru.iloc[epoch,0] = time.time()-start_time
            test_score = evaluate(model,test_set,sess)
            results_b_gru.iloc[1,4] = test_score
            print test_score,'test_score'
 
            output_filename = 'sentiment.csv'
            #1rst alteration
            output_path = os.path.join('.../results', output_filename)
            #2nd alteration
            #output_path = os.path.join('.../results', output_filename)
            results_b_gru.to_csv(output_path)
 
def train_epoch(model,data,sess):
 
    loss=model.train(data,sess)
    return loss
 
def evaluate(model,data,sess):
    acc=model.evaluate(data,sess)
    return acc
 
if __name__ == '__main__':
    if len(sys.argv) > 1:
        restore=True
    else:restore=False
    train(restore)
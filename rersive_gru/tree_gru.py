import numpy as np
import tensorflow as tf
import os
import sys

from data_utils import extract_tree_data,extract_batch_tree_data

class tf_NarytreeGRU(object):
# Tune the parameters and the hyperparameters
    def __init__(self,config):
        self.emb_dim = config.emb_dim
        self.hidden_dim = config.hidden_dim
        self.num_emb = config.num_emb
        self.output_dim = config.output_dim
        self.config=config
        self.batch_size=config.batch_size
        self.reg=self.config.reg
        self.degree=config.degree
        assert self.emb_dim > 1 and self.hidden_dim > 1

        self.add_placeholders()

        emb_leaves = self.add_embedding()

        self.add_model_variables()

        batch_loss = self.compute_loss(emb_leaves)

        self.loss,self.total_loss=self.calc_batch_loss(batch_loss)

        self.train_op1,self.train_op2 = self.add_training_op()
        #self.train_op=tf.no_op()

    def add_embedding(self):

        #embed=np.load('glove{0}_uniform.npy'.format(self.emb_dim))
        with tf.variable_scope("Embed",regularizer=None):
            embedding=tf.get_variable('embedding',[self.num_emb,
                                                   self.emb_dim],initializer = tf.random_uniform_initializer(-0.05,0.05),
                                                   trainable=True,regularizer=None)
            ix = tf.to_int32(tf.not_equal(self.input,-1))*self.input
            emb_tree = tf.nn.embedding_lookup(embedding,ix)
            emb_tree = emb_tree*(tf.expand_dims(tf.to_float(tf.not_equal(self.input,-1)),2))

            return emb_tree


    def add_placeholders(self):
        dim2=self.config.maxnodesize
        dim1=self.config.batch_size
        self.input = tf.placeholder(tf.int32,[dim1,dim2],name='input')
        self.treestr = tf.placeholder(tf.int32,[dim1,dim2,2],name='tree')
        self.labels = tf.placeholder(tf.int32,[dim1,dim2],name='labels')
        self.dropout = tf.placeholder(tf.float32,name='dropout')

        self.n_inodes = tf.reduce_sum(tf.to_int32(tf.not_equal(self.treestr,-1)),[1,2])
        self.n_inodes = self.n_inodes/2

        self.num_leaves = tf.reduce_sum(tf.to_int32(tf.not_equal(self.input,-1)),[1])
        self.batch_len = tf.placeholder(tf.int32,name="batch_len")

    def calc_wt_init(self,fan_in=300):
        eps=1.0/np.sqrt(fan_in)
        return eps

    def add_model_variables(self):
# Initialize the weights and biases
        with tf.variable_scope("Composition",
                                initializer = tf.contrib.layers.xavier_initializer(),
                                regularizer = tf.contrib.layers.l2_regularizer(self.config.reg)):
#cU : 300X300
#initial CW : 300X750
#GRU : CW : 300X600
#initial cb : 4*150=600 -> 450 
            cU = tf.get_variable("cU",[self.emb_dim,2*self.hidden_dim],initializer=tf.random_uniform_initializer(-self.calc_wt_init(),self.calc_wt_init()))
            cW = tf.get_variable("cW",[self.degree*self.hidden_dim,(self.degree+2)*self.hidden_dim],initializer=tf.random_uniform_initializer(-self.calc_wt_init(self.hidden_dim),self.calc_wt_init(self.hidden_dim)))
            cb = tf.get_variable("cb",[3*self.hidden_dim],initializer=tf.constant_initializer(0.0),regularizer=tf.contrib.layers.l2_regularizer(0.0))
        
        with tf.variable_scope("Projection",regularizer=tf.contrib.layers.l2_regularizer(self.config.reg)):

            U = tf.get_variable("U",[self.output_dim,self.hidden_dim],
                                initializer=tf.random_uniform_initializer(self.calc_wt_init(self.hidden_dim),self.calc_wt_init(self.hidden_dim))
                                    )
            bu = tf.get_variable("bu",[self.output_dim],initializer=
                                 tf.constant_initializer(0.0),regularizer=tf.contrib.layers.l2_regularizer(0.0))

    def process_leafs(self,emb):
        #cU : 300X300
        #cb : 600 -> 450
        #b : 300

        with tf.variable_scope("Composition",reuse=True):
            cU = tf.get_variable("cU",[self.emb_dim,2*self.hidden_dim])
            cb = tf.get_variable("cb",[3*self.hidden_dim])
            b = tf.slice(cb,[0],[2*self.hidden_dim])
            def _recurseleaf(x):

                concat_zh = tf.matmul(tf.expand_dims(x,0),cU) + b
                z,h_ = tf.split(1,2,concat_zh)

                z=tf.nn.sigmoid(z)
                h_=tf.nn.tanh(h_)


                h = (1-z)* h_ + z * x[0]
                h=tf.squeeze(h) 
                return h
        hc = tf.map_fn(_recurseleaf,emb)
        return hc


    def compute_loss(self,emb_batch,curr_batch_size=None):
        outloss=[]
        prediction=[]
        for idx_batch in range(self.config.batch_size):

            tree_states=self.compute_states(emb_batch,idx_batch)
            logits = self.create_output(tree_states)

            labels1=tf.gather(self.labels,idx_batch)
            labels2=tf.reduce_sum(tf.to_int32(tf.not_equal(labels1,-1)))
            labels=tf.gather(labels1,tf.range(labels2))
            loss = self.calc_loss(logits,labels)


            pred = tf.nn.softmax(logits)

            pred_root=tf.gather(pred,labels2-1)


            prediction.append(pred_root)
            outloss.append(loss)

        batch_loss=tf.pack(outloss)
        self.pred = tf.pack(prediction)

        return batch_loss


    def compute_states(self,emb,idx_batch=0):


        num_leaves = tf.squeeze(tf.gather(self.num_leaves,idx_batch))
        #num_leaves=tf.Print(num_leaves,[num_leaves])
        n_inodes = tf.gather(self.n_inodes,idx_batch)
        #embx=tf.gather(emb,tf.range(num_leaves))
        embx=tf.gather(tf.gather(emb,idx_batch),tf.range(num_leaves))
        #treestr=self.treestr#tf.gather(self.treestr,tf.range(self.n_inodes))
        treestr=tf.gather(tf.gather(self.treestr,idx_batch),tf.range(n_inodes))
        leaf_h = self.process_leafs(embx)
       # leaf_h,_=tf.split(1,2,leaf_hc)


        node_h=tf.identity(leaf_h)

        idx_var=tf.constant(0) #tf.Variable(0,trainable=False)

        with tf.variable_scope("Composition",reuse=True):

#cW : 300,5*150' -> 300,4*150
#cb : 600

            cW = tf.get_variable("cW",[self.degree*self.hidden_dim,(self.degree+2)*self.hidden_dim])
            cb = tf.get_variable("cb",[3*self.hidden_dim])
            bh,bz,br=tf.split(0,3,cb)

            #1rst Alteration
            def _recurrence(node_h,idx_var):
                node_info=tf.gather(treestr,idx_var)

                child_h=tf.gather(node_h,node_info)

                flat_ = tf.reshape(child_h,[-1])
                tmp=tf.matmul(tf.expand_dims(flat_,0),cW)
                h,rl,rr,z=tf.split(1,4,tmp)


                z = tf.nn.sigmoid(z+bz)
                rl = tf.nn.sigmoid(rl+br)
                rr = tf.nn.sigmoid(rr+br)
                h_ = tf.nn.tanh(h+bh)
                r=tf.concat(0,[rl,rr])

                h = (1-z) * h_ + tf.reduce_sum(z*child_h,[0])

                node_h = tf.concat(0,[node_h,h])

                idx_var=tf.add(idx_var,1)


                return node_h,idx_var#,node_c
            loop_cond = lambda a1,idx_var: tf.less(idx_var,n_inodes)

            loop_vars=[node_h,idx_var]
            node_h,idx_var=tf.while_loop(loop_cond, _recurrence,
                                                loop_vars,parallel_iterations=10)

            return node_h

            #2nd Alteration
            # def _recurrence(node_h,idx_var):
            #     node_info=tf.gather(treestr,idx_var)

            #     child_h=tf.gather(node_h,node_info)

            #     flat_ = tf.reshape(child_h,[-1])
            #     tmp=tf.matmul(tf.expand_dims(flat_,0),cW)
            #     h,rl,rr,zl,zr=tf.split(1,5,tmp)


            #     zl = tf.nn.sigmoid(zl+bz)
            #     zr = tf.nn.sigmoid(zr+bz)
            #     rl = tf.nn.sigmoid(rl+br)
            #     rr = tf.nn.sigmoid(rr+br)
            #     h_ = tf.nn.tanh(h+bh)
            #     r=tf.concat(0,[rl,rr])
            #     z=tf.concat(0,[zl,z])

            #     h = (1-z) * h_ + tf.reduce_sum(z*child_h,[0])

            #     node_h = tf.concat(0,[node_h,h])

            #     idx_var=tf.add(idx_var,1)

            #     return node_h,idx_var#,node_c
            # loop_cond = lambda a1,idx_var: tf.less(idx_var,n_inodes)

            # loop_vars=[node_h,idx_var]
            # node_h,idx_var=tf.while_loop(loop_cond, _recurrence,
            #                                     loop_vars,parallel_iterations=10)

            return node_h



    def create_output(self,tree_states):

        with tf.variable_scope("Projection",reuse=True):

            U = tf.get_variable("U",[self.output_dim,self.hidden_dim],
                                )
            bu = tf.get_variable("bu",[self.output_dim])

            h=tf.matmul(tree_states,U,transpose_b=True)+bu
            return h



    def calc_loss(self,logits,labels):

        l1=tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits,labels)
        loss=tf.reduce_sum(l1,[0])
        return loss

    def calc_batch_loss(self,batch_loss):
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        regpart=tf.add_n(reg_losses)
        loss=tf.reduce_mean(batch_loss)
        total_loss=loss+0.5*regpart
        return loss,total_loss

    def add_training_op_old(self):

        opt = tf.train.AdagradOptimizer(self.config.lr)
        train_op = opt.minimize(self.total_loss)
        return train_op

    def add_training_op(self):
        loss=self.total_loss
        opt1=tf.train.AdagradOptimizer(self.config.lr)
        opt2=tf.train.AdagradOptimizer(self.config.emb_lr)

        ts=tf.trainable_variables()
        gs=tf.gradients(loss,ts)
        gs_ts=zip(gs,ts)

        gt_emb,gt_nn=[],[]
        for g,t in gs_ts:
            #print t.name,g.name
            if "Embed/embedding:0" in t.name:
                #g=tf.Print(g,[g.get_shape(),t.get_shape()])
                gt_emb.append((g,t))
                #print t.name
            else:
                gt_nn.append((g,t))
                #print t.name

        train_op1=opt1.apply_gradients(gt_nn)
        train_op2=opt2.apply_gradients(gt_emb)
        train_op=[train_op1,train_op2]

        return train_op



    def train(self,data,sess):
        from random import shuffle
        data_idxs=range(len(data))
        shuffle(data_idxs)
        losses=[]
        for i in range(0,len(data),self.batch_size):
            batch_size = min(i+self.batch_size,len(data))-i
            if batch_size < self.batch_size:break

            batch_idxs=data_idxs[i:i+batch_size]
            batch_data=[data[ix] for ix in batch_idxs]#[i:i+batch_size]

            input_b,treestr_b,labels_b=extract_batch_tree_data(batch_data,self.config.maxnodesize)

            feed={self.input:input_b,self.treestr:treestr_b,self.labels:labels_b,self.dropout:self.config.dropout,self.batch_len:len(input_b)}

            loss,_,_=sess.run([self.loss,self.train_op1,self.train_op2],feed_dict=feed)
            #sess.run(self.train_op,feed_dict=feed)

            losses.append(loss)
            avg_loss=np.mean(losses)
            sstr='avg loss %.2f at example %d of %d\r' % (avg_loss, i, len(data))
            sys.stdout.write(sstr)
            sys.stdout.flush()

            #if i>1000: break
        return np.mean(losses)


    def evaluate(self,data,sess):
        num_correct=0
        total_data=0
        data_idxs=range(len(data))
        test_batch_size=self.config.batch_size
        losses=[]
        for i in range(0,len(data),test_batch_size):
            batch_size = min(i+test_batch_size,len(data))-i
            if batch_size < test_batch_size:break
            batch_idxs=data_idxs[i:i+batch_size]
            batch_data=[data[ix] for ix in batch_idxs]#[i:i+batch_size]
            labels_root=[l for _,l in batch_data]
            input_b,treestr_b,labels_b=extract_batch_tree_data(batch_data,self.config.maxnodesize)

            feed={self.input:input_b,self.treestr:treestr_b,self.labels:labels_b,self.dropout:1.0,self.batch_len:len(input_b)}

            pred_y=sess.run(self.pred,feed_dict=feed)
            #print pred_y,labels_root
            y=np.argmax(pred_y,axis=1)
            #num_correct+=np.sum(y==np.array(labels_root))
            for i,v in enumerate(labels_root):
                if y[i]==v:num_correct+=1
                total_data+=1
            #break

        acc=float(num_correct)/float(total_data)
        return acc

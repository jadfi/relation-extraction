import tensorflow as tf
import numpy as np
import mycommon as mc

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)   

class BAGRNN_Model:
    def __init__(self,
                 bag_num = 50,
                 enc_dim = 256,
                 embed_dim = 200,
                 rel_dim = None,
                 cat_n = 5,
                 sent_len = 120,
                 word_n = 80000,
                 extra_n = 3,
                 word_embed = None,
                 dropout = None,
                 cell_type = 'gru',
                 adv_eps = None,
                 adv_type = 'sent',
                 tune_embed = False,
                 use_softmax_loss = None,
                 sampled_sigmoid_loss = False,
                 max_dist_embed = None,
                 excl_na_loss = True,
                 att_loss_weight = 0,
                 only_perturb_pos_rel = False):
        self.bag_num = bag_num # total number of bags
        self.enc_dim = enc_dim
        self.att_loss_weight = att_loss_weight
        if rel_dim is None:
            self.rel_dim = 3 * enc_dim if cell_type == 'pcnn' else 2 * enc_dim
        else:
            self.rel_dim = rel_dim
        self.embed_dim = embed_dim
        self.cat_n = cat_n
        self.sent_len = sent_len
        self.pretrain_word_embed = word_embed
        self.word_n = word_n
        self.extra_n = extra_n
        self.dropout = dropout
        self.cell_type = cell_type
        self.adv_eps = adv_eps  # eps for adversarial training, if None, classical feedfwd net
        self.adv_type = adv_type  # type of adversarial perturbation: batch, bag, sent
        self.tune_embed = tune_embed
        self.use_softmax_loss = (use_softmax_loss is not None)  # whether to use softmax loss or sigmoid loss
        self.use_full_softmax = self.use_softmax_loss and (use_softmax_loss > 0)
        self.sampled_sigmoid_loss = sampled_sigmoid_loss and not use_softmax_loss
        self.max_dist_embed = max_dist_embed
        self.use_pcnn = (cell_type == 'pcnn')  # None for RNN; other wise use PCNN with feature size <use_pcnn>
        self.excl_na_loss = excl_na_loss  # exclude NA in the loss function, only effective for sigmoid loss
        self.only_perturb_pos_rel = only_perturb_pos_rel

    def build(self, is_training,
              ent_dim = 3,
              dropout_embed = True):
        self.is_training = is_training

        bag_num = self.bag_num
        cat_n = self.cat_n
        L = self.sent_len
        rel_dim = self.rel_dim
        enc_dim = self.enc_dim
        cell_type = self.cell_type
        dropout = self.dropout
   
        

        ###################################
        # create placeholders
        #####
        # data shape info
        self.shapes = shapes = tf.placeholder(tf.int32, [self.bag_num + 1])
        # input data
        self.X = tf.placeholder(tf.int32, [None, L])
        self.ent = tf.placeholder(tf.int32, [None, L])
        if self.max_dist_embed is not None:
            self.ent2 = tf.placeholder(tf.int32, [None, L])
        # labels
        self.Y = ph_Y = tf.placeholder(tf.float32, [bag_num, cat_n])
        # sentence length
        self.length = length = tf.placeholder(tf.int32, [None])
        self.pool_length = tf.placeholder(tf.int32, [None])
        # sentence mask
        self.mask = mask = tf.placeholder(tf.float32, [None, L])
        # adversarial eps
        if self.adv_eps is not None:
            self.adv_eps = tf.placeholder(tf.float32, shape=())
        # loss mask
        if self.sampled_sigmoid_loss:
            self.loss_mask = loss_mask = tf.placeholder(tf.float32, [bag_num, cat_n])
        else:
            loss_mask = None
        if self.use_pcnn:
            self.pcnn_mask = tf.placeholder(tf.float32, [None, 3, L])
            pcnn_pos_mask = tf.expand_dims(tf.transpose(self.pcnn_mask, [0, 2, 1]), axis=1)  # [batch, 1, L, 3]
        if self.use_full_softmax:
            self.diag = tf.expand_dims(tf.eye(cat_n, dtype=tf.float32), axis=0)

        #################################
        # create embedding variables
        ####
        self.exclude_clip_vars = set()
        # pre-process entity embedding
        if self.max_dist_embed is None:#one-hot  ent_embed表示向量映射表
            self.ent_embed = tf.constant(np.array([[0] * ent_dim * 2, [1] * ent_dim + [0] * ent_dim, [0] * ent_dim + [1] * ent_dim],
                                                  dtype=np.float32),
                                         dtype=tf.float32)  #位置向量词典，6维，包括00,01和10的编码，一个词不可能是两个实体，所以没有11
        else:
            #位置向量随机初始化 一个距离向量是3维，距离表示范围从+dis到-dis
            self.ent_embed = tf.get_variable('dist_embed', [2 * self.max_dist_embed + 1, ent_dim],
                                             initializer=tf.random_normal_initializer(0, 0.01))
            self.exclude_clip_vars.add(self.ent_embed)
        #词向量预训练
        if self.pretrain_word_embed is not None:
            #微调词向量
            if self.tune_embed:
                pretrain_embed = tf.get_variable('pretrain_embed',
                                                 initializer=self.pretrain_word_embed)
                self.exclude_clip_vars.add(pretrain_embed)
            #固定词向量
            else:
                pretrain_embed = tf.constant(self.pretrain_word_embed,dtype=tf.float32)
            extra_embed = tf.get_variable('extra_embed', [self.extra_n,
                                                          self.embed_dim],
                                          initializer=tf.random_normal_initializer(0,0.01))
            self.exclude_clip_vars.add(extra_embed)
            self.word_embed = tf.concat([pretrain_embed, extra_embed], axis=0)
        else:
            self.word_embed = tf.get_variable('word_embed', [self.word_n+self.extra_n,
                                                             self.embed_dim],
                                              initializer=tf.random_normal_initializer(0,0.01))
            self.exclude_clip_vars.add(self.word_embed)

        ################################
        # discriminative model
        #####
        self.orig_inputs = orig_inputs = mc.get_embedding(self.X, self.word_embed,
                                                          self.dropout if dropout_embed else None, self.is_training)
        if self.max_dist_embed is not None:
            dist1_embed = mc.get_embedding(self.ent, self.ent_embed,
                                           self.dropout if dropout_embed else None, self.is_training)
            dist2_embed = mc.get_embedding(self.ent2, self.ent_embed,
                                           self.dropout if dropout_embed else None, self.is_training)
            ent_inputs = tf.concat([dist1_embed, dist2_embed], axis=2)
        else:
            ent_inputs = mc.get_embedding(self.ent, self.ent_embed)  # [batch,L,dim]
            #一句话所有词的位置向量

        use_softmax_loss = self.use_softmax_loss
        use_full_softmax = self.use_full_softmax
        use_pcnn = self.use_pcnn
        pcnn_feat_size = self.enc_dim

        def rm_noise_by_guass(curr_V,curr_alpha):
            mean_v = tf.reduce_mean(curr_V,axis=0) #[dim]
            distance_v = tf.reduce_sum(tf.square(curr_V - mean_v),axis=1) #[n]
            mean_distance = tf.reduce_mean(distance_v)
            sigma = tf.sqrt(tf.reduce_sum(tf.square(distance_v-mean_distance)))
            mean_up_3sigma = mean_distance+3*sigma
            mean_below_3sigma=mean_distance-3*sigma
            remain_index = tf.cast(distance_v>=mean_below_3sigma,tf.int16)* tf.cast(distance_v<=mean_up_3sigma,tf.int16)
            remain_index = tf.cast(remain_index,tf.bool) #[n]
            remain_x =tf.reshape(tf.where(remain_index),[-1]) #[remain_n] 存放满足高斯过滤条件的位置
            remain_v = tf.nn.embedding_lookup(curr_V,remain_x)
            remain_alpha = tf.nn.embedding_lookup(curr_alpha,remain_x)
            return remain_v,remain_alpha,remain_x

        def activate(vector):
            return tf.nn.tanh(vector)

        def cal_dense_loss(V_att,bag_all_v,ind,n):
            true_att = tf.tile(tf.expand_dims(V_att[ind,:],axis=0),[n,1]) #[n,dim]
            true_all_v = bag_all_v[ind,:,:] #[n,dim]
            true_dense_loss = tf.reduce_sum(tf.square(true_att-true_all_v),axis=1) #经过映射后的向量，和包向量应尽可能语义相似(都是基于某个某个主题的表达)
            #dense_loss = tf.reduce_mean(tf.sqrt(true_dense_loss))
            dense_loss = tf.reduce_mean(true_dense_loss)
            return dense_loss,true_dense_loss

        def dense_att_all(curr_V,ind,Q,n,cat_n):
            my_curr_v=tf.tile(tf.expand_dims(curr_V,axis=0),[cat_n,1,1]) #[cat_n,n,rel_dim]
            my_q=tf.transpose(Q,[1,0]) #[cat_n,rel_dim]
            my_full_q=tf.tile(tf.expand_dims(my_q,axis=1),[1,n,1])  #[cat_n,n,rel_dim]
            bag_all_v = my_full_q * my_curr_v #[cat_n,n,rel_dim]
            bag_all_v = activate(bag_all_v)
            V_att = tf.reduce_mean(bag_all_v,axis=1)  # [cat_n, dim]
            dense_loss_list=[]
            for i in range(cat_n):
                dense_loss,true_dense_loss = cal_dense_loss(V_att,bag_all_v,i,n)
                dense_loss_list.append(dense_loss)
            dense_loss = tf.stack(dense_loss_list) #[cat_n]
            real_dense_loss = dense_loss[ind]
            other_dense_loss = (tf.reduce_sum(dense_loss)-real_dense_loss)/(cat_n-1)
            dense_loss = real_dense_loss - 0.01*other_dense_loss
            return V_att,dense_loss,true_dense_loss

        def dense_att(curr_V,ind,Q,n,cat_n):
            my_curr_v=tf.tile(tf.expand_dims(curr_V,axis=0),[cat_n,1,1]) #[cat_n,n,rel_dim]
            my_q=tf.transpose(Q,[1,0]) #[cat_n,rel_dim]
            my_full_q=tf.tile(tf.expand_dims(my_q,axis=1),[1,n,1])  #[cat_n,n,rel_dim]
            bag_all_v = my_full_q * my_curr_v #[cat_n,n,rel_dim]
            bag_all_v = activate(bag_all_v)
            V_att = tf.reduce_mean(bag_all_v,axis=1)  # [cat_n, dim]
            dense_loss,true_dense_loss = cal_dense_loss(V_att,bag_all_v,ind,n)
            return V_att,dense_loss,true_dense_loss

        def multidense_att(curr_V,ind,Q_list,n,cat_n):
            my_curr_v=tf.tile(tf.expand_dims(curr_V,axis=0),[cat_n,1,1]) #[cat_n,n,rel_dim]
            full_q_list=[]
            for Q in Q_list:
                my_q=tf.transpose(Q,[1,0]) #[cat_n,rel_dim]
                my_full_q=tf.tile(tf.expand_dims(my_q,axis=1),[1,n,1])  #[cat_n,n,rel_dim]
                full_q_list.append(my_full_q)
            bag_all_v=my_curr_v
            for full_q in full_q_list:
                bag_all_v = bag_all_v*full_q #[cat_n,n,rel_dim]
                bag_all_v = activate(bag_all_v)
            V_att = tf.reduce_mean(bag_all_v,axis=1)  # [cat_n, dim]
            dense_loss,true_dense_loss = cal_dense_loss(V_att,bag_all_v,ind,n)
            return V_att,dense_loss,true_dense_loss

        def full_dense_att(curr_V,ind,dense_matrix,n):
            dense_v=[]
            for i in range(cat_n):
                trans_v=tf.matmul(curr_V,dense_matrix[i,:,:]) #[n,dim]
                dense_v.append(trans_v)
            dense_v=tf.stack(dense_v) #[cat_n,n,dim]
            dense_v=activate(dense_v)
            V_att=tf.reduce_mean(dense_v,axis=1) #[cat_n,dim]
            dense_loss,true_dense_loss = cal_dense_loss(V_att,dense_v,ind,n)
            return V_att,dense_loss,true_dense_loss


        def discriminative_net(word_inputs, name = 'discriminative-net', reuse = False,
                                 only_pos_rel_loss = False):
            with tf.variable_scope(name, reuse=reuse):
                if only_pos_rel_loss:
                    pos_rel_mask = ph_Y
                    # when y = [0, 0, ..., 0]: pos_rel_mask = [1, 1, ..., 1]
                    # o.w. pos_rel_mask = y
                    #na_flag = 1 - tf.reduce_max(ph_Y, axis=1, keep_dims=True)
                    #pos_rel_mask = ph_Y + na_flag

                inputs = tf.concat([word_inputs, ent_inputs], axis = 2)  # [batch, L, dim]

                if not use_pcnn:  # use RNN
                    outputs, states = mc.mybidrnn(inputs, length, enc_dim,
                                                  cell_name = cell_type,
                                                  scope = 'bidirect-rnn')
                    # sentence information
                    V = tf.concat(states, axis=1) # [batch, rel_dim]
                    V_dim = enc_dim * 2
                else:
                    # use pcnn
                    feat_size = pcnn_feat_size
                    window_size = 3
                    inputs = tf.expand_dims(inputs, axis=1)  # [batch, 1, L, dim]   1行L个词，词向量是通道
                    
                    conv_out = tf.squeeze(tf.nn.relu(
                        tf.layers.conv2d(inputs, feat_size, [1, window_size], 1, padding='same',
                                         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
                    ))  # [batch, L, feat_size]

                    conv_out = tf.expand_dims(tf.transpose(conv_out, [0, 2, 1]), axis=-1)  # [batch, feat, L, 1]
                    pcnn_pool = tf.reduce_max(conv_out * pcnn_pos_mask, axis=2)  # [batch, feat, 3]
                    
                    # V = tf.reshape(pcnn_pool, [-1, feat_size * 3])#可以看成句子向量了，维度是卷积核乘以3
                    # V_dim = feat_size * 3

                    rnn_pool = tf.transpose(pcnn_pool,[0,2,1])
                    rnn_V = tf.reshape(rnn_pool,[-1,3,feat_size])
                    rnn_out,rnn_states = mc.mybidrnn(rnn_V, self.pool_length, feat_size,
                                                  cell_name = 'lstm',
                                                  scope = 'bidirect-rnn')
                    V = tf.concat(rnn_states, axis=1) # [batch, rel_dim]
                    V_dim = feat_size * 2
                #rel_dim=V_dim
                # rel_dim=32
                if V_dim != rel_dim:
                    V = mc.linear(V, rel_dim, scope='embed_proj')
                if dropout:
                    V = tf.layers.dropout(V, rate=dropout,
                                          training=is_training)

                #################################
                # Multi Label Multi Instance Learning
                #####
                #rel_dim就是句子向量维度
                #Q=tf.random_normal([rel_dim, cat_n],0,0.01)
                Q = tf.get_variable('relation_embed', [rel_dim, cat_n],
                                   initializer=tf.random_normal_initializer(0, 0.01))
                Q2 = tf.get_variable('relation_embed2', [rel_dim, cat_n],
                                   initializer=tf.random_normal_initializer(0, 0.01))
                Q3 = tf.get_variable('relation_embed3', [rel_dim, cat_n],
                                   initializer=tf.random_normal_initializer(0, 0.01))

                #Q = tf.constant(np.random.randn(rel_dim,cat_n)*0.01,dtype=tf.float32)
                dense_matrix = tf.get_variable('dense_parameters', [cat_n,rel_dim,rel_dim],
                                   initializer=tf.random_normal_initializer(0, 0.01))
                Noise_matrix=tf.get_variable('noise_matrix',
                                    initializer=tf.eye(cat_n))
                if use_full_softmax:
                    A = tf.get_variable('classify-proj', [rel_dim, cat_n],
                                        initializer=tf.random_normal_initializer(0, 0.01))
                else:
                    #A = tf.get_variable('classify-proj', [1, rel_dim],
                    #    initializer=tf.random_normal_initializer(0, 0.01))
                    A = tf.get_variable('classify-proj', [cat_n, rel_dim],
                                       initializer=tf.random_normal_initializer(0, 0.01))
                #句子向量乘以关系向量，也就是注意力，cat_n表示句子和每个关系的注意力
                alpha = tf.matmul(tf.nn.tanh(V), Q) # [batch, cat_n]
                # process bags
                logits_list = []
                alpha_list=[]
                weight_list=[]
                distance_list=[]
                ins_alpha_list = []
                weight_loss_list = []
                dense_loss_list=[]
                remain_x_list=[]
                true_dense_loss=dense_loss=tf.constant(0)
                remain_x=tf.constant([0])
                for i in range(bag_num):
                    n = shapes[i+1] - shapes[i]
                    curr_V = V[shapes[i]:shapes[i+1], :]  # [n, rel_dim] 一个包的所有句子
                    curr_alpha = alpha[shapes[i]:shapes[i+1], :]  # [n, cat_n]  一个包在每个关系上的权重
                    #curr_V,curr_alpha,self.remain_x = rm_noise_by_guass(curr_V,curr_alpha)
                    weight = tf.nn.softmax(tf.transpose(curr_alpha, [1, 0]))  # [cat_n, n]
                    #weight = tf.transpose(tf.nn.softmax(curr_alpha),[1,0])
                    #weight = tf.nn.sigmoid(tf.transpose(curr_alpha, [1, 0]))
                    weight_list.append(tf.transpose(weight,[1,0]))  #[n,cat_n]
                    full_weight = tf.tile(tf.expand_dims(weight, axis=-1), [1, 1, rel_dim])  # [cat_n, n, dim]
                    full_V = tf.tile(tf.expand_dims(curr_V, axis=0), [cat_n, 1, 1]) # [cat_n, n, dim]
                    #注意力加权结果，在每个关系上都有一个包向量
                    #V_att = tf.reduce_sum(full_weight * full_V, axis=1)  # [cat_n, dim]
                    #V_att = tf.reduce_mean(full_weight * full_V, axis=1)
                    #V_att = tf.reduce_max(full_V,axis=1) # [cat_n,dim]
                    #V_att = tf.reduce_mean(full_V,axis=1)  
                    #V_att = tf.tile(tf.expand_dims(tf.reduce_mean(curr_V,axis=0),axis=0),[cat_n,1])
                    #V_att = tf.reduce_sum(full_V, axis=1)
                    #V_att = tf.random_normal([cat_n,rel_dim],0,0.01)
                    #V_att = tf.squeeze(full_V)

                    ind=tf.cast(tf.argmax(ph_Y[i]),tf.int32)
                    #V_att,dense_loss,true_dense_loss=dense_att(curr_V,ind,Q,n,cat_n)
                    #V_att,dense_loss,true_dense_loss=full_dense_att(curr_V,ind,dense_matrix,n)
                    #V_att,dense_loss,true_dense_loss=dense_att_all(curr_V,ind,Q,n,cat_n)
                    V_att,dense_loss,true_dense_loss=multidense_att(curr_V,ind,[Q,Q2,Q3],n,cat_n)

                    #alpha_list.append(tf.reduce_sum(tf.nn.tanh(V_att[ind])*Q[:,ind]))
                    #alpha_list.append(tf.nn.softmax(tf.matmul(tf.nn.tanh(V_att),Q))*tf.eye(cat_n))
                    # def false_func():
                    
                    
                    ind_alpha = tf.nn.softmax(tf.matmul(tf.nn.tanh(V_att),Q))[ind,ind]
                    alpha_list.append(ind_alpha)
                    distance_list.append(tf.sqrt(tf.reduce_sum(tf.square(V_att[ind,:]-Q[:,ind]))))
                    ins_alpha = tf.nn.softmax(tf.matmul(tf.nn.tanh(curr_V), Q))[:,ind]
                    ins_alpha_list.append(tf.reduce_mean(ins_alpha))
                    #weight_loss_list.append(tf.reduce_max(weight[ind,:]))
                    weight_loss_list.append(tf.reduce_min(weight))
                    dense_loss_list.append(dense_loss)
                    remain_x_list.append(remain_x)


                        # return tf.Variable(0)
                    # def true_func():
                    #     return tf.Variable(1)
                    #tf.cond(tf.equal(ind,tf.constant(0)),true_func,false_func)

                    # trans_V = tf.expand_dims(tf.expand_dims(tf.transpose(curr_V,[1,0]),axis=0),axis=0) #[1,1,rel_dim,n]
                    # new_V = tf.reshape(trans_V,[1,1,rel_dim,n])
                    # print(new_V.shape)
                    # conv = tf.squeeze(tf.nn.relu(
                    #     tf.layers.conv2d(new_V, pcnn_feat_size, [1, 3], 1, padding='same',
                    #                      kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
                    # )) #[1,rel_dim,pcnn_feat_size]
                    # pool = tf.squeeze(tf.reduce_max(conv,axis=1)) 

                    if use_full_softmax:
                        cat_logits = tf.matmul(V_att, A)  # [cat_n, cat_n]
                    else:
                        #share_A = tf.tile(A,[cat_n,1])
                        cat_logits = tf.reduce_sum(V_att * A, axis=1)  # [cat_n]  A相当于sigmoid层的权重，点积求和得到最后一层的输入,不是全连接，而是每个关系上得到一个值，然后softmax
                        #cat_logits=tf.squeeze(tf.matmul(tf.expand_dims(cat_logits,axis=0),Noise_matrix))
                    logits_list.append(cat_logits)
                logits = tf.stack(logits_list)  # [bag_num, cat_n] or [bag_num, cat_n, cat_n]
                att_loss=tf.reduce_mean(tf.stack(alpha_list))  #包到自己对应关系的权重尽可能大
                weight_list=tf.concat(weight_list,axis=0)
                distance_loss=tf.reduce_mean(distance_list)
                ins_alpha_loss = tf.reduce_mean(ins_alpha_list)
                weight_loss = tf.reduce_mean(weight_loss_list)
                denseloss=tf.reduce_mean(tf.stack(dense_loss_list))
                self.remain_x=tf.concat(remain_x_list,axis=0)
                if use_softmax_loss:
                    probs = tf.nn.softmax(logits)
                    if use_full_softmax:
                        # probs: [bag_num, cat_n, cat_n] last dimension normalized
                        probs = tf.reduce_sum(probs * self.diag, axis=-1)  # [bag_num, cat_n]
                        # optimize the sum of softmax-loss for each positive rel
                        loss = -tf.reduce_mean(tf.reduce_sum(tf.log(probs + 1e-20) * ph_Y, axis=1))
                    else:
                        # add all the probs, output the joint log probability
                        loss = -tf.reduce_mean(tf.log(tf.reduce_sum(probs * ph_Y, axis=1) + 1e-20)) #怎么感觉算错了
                        #loss = -tf.reduce_mean(tf.reduce_sum(tf.log(probs + 1e-20) * ph_Y, axis=1))

                else:
                    probs = tf.nn.sigmoid(logits)
                    if loss_mask is not None:
                        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=ph_Y, logits=logits)
                        if only_pos_rel_loss:
                            loss = loss * pos_rel_mask
                        loss = tf.reduce_sum(loss * loss_mask, axis=1)
                        weight = tf.reduce_sum(loss_mask, axis=1)
                        # weighted average of the individual sigmoid loss, rescale to full_sigmoid_loss
                        #coef = cat_n - 1 if self.excl_na_loss else cat_n
                        #loss = tf.reduce_mean(loss / weight) * coef
                        loss = tf.reduce_mean(loss) # * coef
                    else:
                        if self.excl_na_loss:
                            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=ph_Y, logits=logits)  # [bag, cat_n]
                            if only_pos_rel_loss:
                                loss = loss * pos_rel_mask
                            loss = loss[:, 1:]  # exclude NA
                            loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
                        else:
                            if only_pos_rel_loss:
                                loss = tf.losses.sigmoid_cross_entropy(ph_Y, logits, weights=pos_rel_mask)
                            else:
                                loss = tf.losses.sigmoid_cross_entropy(ph_Y, logits)
                #reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.1), tf.trainable_variables())
                #loss+=reg
                #loss+=0.1*distance_loss
                #loss -=  self.att_loss_weight * att_loss
                #loss -= 0.2 * ins_alpha_loss
                #loss -= 0.2 * weight_loss
                #loss += 0.001*denseloss
                
            return probs, loss,tf.nn.softmax(alpha),att_loss,weight_list,ins_alpha_loss,weight_loss,true_dense_loss,denseloss

        self.probs, self.raw_loss, self.alpha,self.att_loss,self.weight_list,self.ins_alpha_loss,self.weight_loss,self.true_dense_loss,self.dense_loss = discriminative_net(orig_inputs, reuse=False,
                                                       only_pos_rel_loss=(self.adv_eps is not None) and (self.only_perturb_pos_rel) and (not use_softmax_loss))
        if self.adv_eps is None:
            self.loss = self.raw_loss
        else:  # adversarial training
            print(self.adv_eps)
            print(self.adv_type)
            raw_perturb = tf.gradients(self.raw_loss, orig_inputs)[0]  # [batch, L, dim]
            if self.adv_type == 'sent':
                # normalize per sentence
                self.perturb = perturb = self.adv_eps * tf.stop_gradient(
                    tf.nn.l2_normalize(raw_perturb * tf.expand_dims(mask, axis=-1), dim=[1, 2]))
            elif self.adv_type == 'batch':
                # normalize the whole batch
                self.perturb = perturb = self.adv_eps * tf.stop_gradient(
                    tf.nn.l2_normalize(raw_perturb * tf.expand_dims(mask, axis=-1), dim=[0,1,2]))
            else:  # bag-level normalization
                raw_perturb = tf.stop_gradient(raw_perturb * tf.expand_dims(mask, axis=-1))  # [batch, L, dim]
                perturb_list = []
                for i in range(bag_num):
                    curr_pt = raw_perturb[shapes[i]:shapes[i+1], :, :]  # [bag_size, L, dim]
                    perturb_list.append(self.adv_eps * tf.nn.l2_normalize(curr_pt, dim=[0,1,2]))
                self.perturb = perturb = tf.concat(perturb_list, axis=0)  # [batch, L, dim]
            self.perturb_inputs = perturb_inputs = orig_inputs + perturb
            self.perturb_probs, self.loss,self.alpha,self.att_loss,self.weight_list = discriminative_net(perturb_inputs, reuse=True)  # optimize the loss with perturbed loss


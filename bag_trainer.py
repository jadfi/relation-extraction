import tensorflow as tf
import numpy as np
import time
import pickle
import mycommon as mc
import heapq
from sklearn.metrics import roc_auc_score

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)   

class MyLogger:
    def __init__(self, logdir, clear_file = False):
        self.fname = logdir + '/progress.txt'
        if clear_file:
            mc.remove_file(self.fname)

    def print_model_spec(self, model,comment=''):
        with open(self.fname, 'w') as f:
            print('+++++++++++++ Model Specs +++++++++++', file=f)
            if model.use_full_softmax:
                s = 'full-softmax-loss with positive rels'
            elif model.use_softmax_loss:
                s = 'combined-single-softmax-loss'
            elif model.sampled_sigmoid_loss is not None:
                if model.sampled_sigmoid_loss > 0:
                    s = 'sampled-sigmoid-loss <K={}>'.format(model.sampled_sigmoid_loss)
                else:
                    s = 'weighted-sigmoid-loss <C={}>'.format(-model.sampled_sigmoid_loss)
            else:
                s = 'full-sigmoid-loss'
            print('Loss: {}'.format(s), file=f)
            print('Model-Type: {}'.format(model.cell_type), file=f)
            print('Kernel-Size: {}'.format(model.enc_dim), file=f)
            print('Feat-Size: {}'.format(model.rel_dim), file=f)
            if model.max_dist_embed is not None:
                s = 'relative-pos-embed <max_dis={}>'.format(model.max_dist_embed)
            else:
                s = 'one-hot-pos'
            print('Entity-Pos-Embed: {}'.format(s), file=f)
            print('备注:{}'.format(comment),file=f)
            print('+++++++++++++++++++++++++++++++++++++', file=f)

    def print(self, str, to_screen = True):
        if to_screen:
            print(str)
        with open(self.fname, 'a') as f:
            print(str, file=f)


class BagTrainer:
    def __init__(self, model, loader,
                 lrate = 0.001,
                 clip_grad = None,
                 lrate_decay_step = 0,
                 sampled_loss = None,
                 adv_eps = None):
        self.model = model
        self.loader = loader
        self.lrate = lrate
        self.lrate_decay = lrate_decay_step
        self.clip_grad = clip_grad
        self.max_len = self.model.sent_len
        self.sampled_loss = sampled_loss
        self.adv_eps = adv_eps
        self.excl_na_loss = model.excl_na_loss
        tf.reset_default_graph()
        self.is_training = tf.placeholder(tf.bool)
        self.continue_n=3
        self.auc_list=[]
        self.e=0.1
        self.max_batch_size=0

    def update_learning_param(self):
        self.lrate = self.lrate * 0.9998

    def compute_relative_dist(self, batch_size, L, length, pos, max_dist):
        def calc_dist(x, a, b, cap):
            dis = 0
            if x <= a:
                dis = x-a
            elif x >= b:
                dis = x-b+1
            if dis < -cap:
                dis = -cap
            if dis > cap:
                dis = cap
            return dis + cap
        D1 = np.zeros((batch_size, L), dtype=np.int32)
        D2 = np.zeros((batch_size, L), dtype=np.int32)
        for i in range(batch_size):
            cur_len = int(length[i])
            for j in range(cur_len):
                D1[i, j] = calc_dist(j, pos[i][0], pos[i][1], max_dist)
                D2[i, j] = calc_dist(j, pos[i][2], pos[i][3], max_dist)
        return D1, D2

    def compute_pcnn_pool_mask(self, batch_size, L, length, pos):
        mask = np.zeros((batch_size, 3, L), dtype=np.float32)
        for i in range(batch_size):
            a,b,c,d = pos[i]
            if d <= a:
                c,d,a,b = pos[i]  # ensure ~ a<b<=c<d
            # piecewise cnn: 0...b-1; b ... d-1; d ... L
            if b>0:
                mask[i, 0, :b] = 1
            if b < d:
                mask[i, 1, b:d] = 1
            if d < length[i]:
                mask[i, 2, d:length[i]] = 1
        return mask

    def feed_dict(self, source='train'):
        loader = self.loader
        fd = dict(zip([self.is_training, self.learning_rate],
                      [(source == 'train'), self.lrate]))
        self.effective,self.effective_size, X, Y, E, length, shapes, mask, batch_s = loader.next_batch(self._bat,source)
        self.max_batch_size=max(self.max_batch_size,batch_s)
        self.batch_size = X.shape[0]
        if source=='train':
            self.count_size+=self.batch_size
        M = self.model
        # feed size
        fd[M.shapes] = shapes
        fd[M.X] = X
        cat_n = Y.shape[1]
        fd[M.Y] = Y #* (1-self.e) + self.e / cat_n
        fd[M.length] = length
        fd[M.mask] = mask
        fd[M.pool_length] = np.array([3]*batch_s)

        if self.adv_eps is not None:
            fd[M.adv_eps] = self.adv_eps

        if M.max_dist_embed is None:
            fd[M.ent] = E
        else:
            D1, D2 = self.compute_relative_dist(self.batch_size, self.max_len,
                                                length, loader.cached_pos, M.max_dist_embed)
            fd[M.ent] = D1
            fd[M.ent2] = D2

        if M.use_pcnn:
            pcnn_mask = self.compute_pcnn_pool_mask(self.batch_size, self.max_len,
                                                    length, loader.cached_pos)
            fd[M.pcnn_mask] = pcnn_mask

        if self.sampled_loss is not None:
            bag_size = Y.shape[0]
            cat_n = Y.shape[1]
            loss_mask = Y.copy()
            if self.excl_na_loss:
                loss_mask[:, 0] = 0  # exclude NA loss
            if self.sampled_loss > 0:  # sample
                for i in range(bag_size):
                    idx = []
                    c = self.sampled_loss
                    for j in range(cat_n):
                        if self.excl_na_loss and j == 0:
                            continue  # ignore NA
                        if Y[i,j] < 0.5:
                            idx.append(j)  # negative rel
                        else:
                            c -= 1  # positive rel
                    np.random.shuffle(idx)
                    for j in range(c):
                        loss_mask[i, idx[j]] = 1
            else:  # normalize weights for zero labels
                scale = -self.sampled_loss
                for i in range(bag_size):
                    idx = []
                    pos = 0
                    neg = 0
                    for j in range(cat_n):
                        if self.excl_na_loss and j == 0:
                            continue
                        if Y[i,j] < 0.5:
                            idx.append(j)
                            neg += 1
                        else:
                            pos += 1
                    #weight = 1 / neg * scale
                    if pos == 0:
                        weight = 1  # NA relation
                    else:
                        weight = pos / neg * scale
                    for j in idx:
                        loss_mask[i, j] = weight
            fd[M.loss_mask] = loss_mask
        return fd

    def evaluate(self, sess, stats_file = './stats/eval_stats.pkl', max_relations = None, incl_conf = False,feed_type='eval'):
        logger = self.logger
        logger.print('Evaluating ...')
        ts = time.time()
        loader = self.loader
        M = self.model
        n_bag = loader.get_bag_n(feed_type)
        n_rel = len(loader.rel_name) # assume NA is 0
        rel_conf = []
        nan_ind = []
        pos_rel_n = 0
        k = 0
        result_dict={}
        right_dict={}
        while k < n_bag:
            all_conf = sess.run(M.probs, feed_dict=self.feed_dict(feed_type))  #训练一个batch的数据
            m = self.effective
            assert(m>0)
            for i in range(m):
                conf = all_conf[i, :]  #得到batch中第i个包在最后一层的概率
                info = loader.get_bag_info(k + i,feed_type) #获取第i个包的正确关系
                # if len(info)>1:   #只考虑单标签
                #     continue
                topn=heapq.nlargest(len(info),range(len(conf)),conf.__getitem__)
                for top in topn:
                    result_dict[top]=result_dict.get(top,0)+1
                for top in info:
                    right_dict[top]=right_dict.get(top,0)+1
                for j in range(1, n_rel):
                    flag = 1 if j in info else 0
                    pos_rel_n += flag    #感觉positive更像active，统计出现的关系的总数
                    #rel_conf.append((j, conf[j], flag))  #(关系，该包在该关系上的概率，该关系是否是真实标签)
                    rel_conf.append((conf[j], flag))
                    if conf[j]!=conf[j]:
                        nan_ind.append(k+i)
            k += m
        print('测试集中真实标签分布:',right_dict,sum(right_dict.values()))
        print('测试集中测试标签分布:',result_dict,sum(result_dict.values()))
        rel_conf.sort(key=lambda x: x[0], reverse=True)
        y_pred,y_true=zip(*rel_conf)
        precis = []
        recall = []
        f1_score = []
        tar_conf = []
        correct = 0

        def get_f1(p, r):
            if p + r <= 1e-20:
                return 0
            return 2 * p * r / (p + r)

        #if max_relations is not None:
        #    rel_conf = rel_conf[:max_relations]
        for i, dat in enumerate(rel_conf):
            #r, p, f = dat
            p,f = dat
            correct += f  #当前正确的总数
            if f > 0: 
                precis.append(correct / (i + 1))  #AUC计算，按照概率排序，从大到小遍历阈值,correct表示以当前遍历到的概率为阈值，正确分类的数量
                recall.append(correct / pos_rel_n)
                f1_score.append(get_f1(precis[-1], recall[-1]))
                tar_conf.append(p)
            if (max_relations is not None) and i+1 == max_relations:
                m = len(precis)


        auc = np.mean(precis)
        if max_relations is not None:
            precis = precis[:m]
            recall = recall[:m]
            f1_score = f1_score[:m]
            tar_conf = tar_conf[:m]

        data_to_dump = [precis, recall, f1_score]
        if incl_conf:
            data_to_dump.append(tar_conf)
        with open(stats_file, 'wb') as f:
            pickle.dump(data_to_dump, f)
        best_f1 = max(f1_score)
        best_p = precis[f1_score.index(best_f1)]
        best_r = recall[f1_score.index(best_f1)]
        logger.print('  -> Done! Time Elapsed = {}s'.format(time.time()-ts))
        logger.print('>>>> Best F1 Score = %.5f, precision = %.5f, recall = %.5f' % (best_f1,best_p,best_r))
        logger.print('>>>> AUC (full) = %.5f' % auc )
        #logger.print('>>>> AUC (full) = %.5f , my AUC = %.5f' % (auc , roc_auc_score(y_true,y_pred)))
        logger.print('nan predict num:'+str(len(nan_ind)))
        return auc

    def change_label(self,ep_alpha,loader,logger):
        '''
        将句子标签修改为句子距离最近的关系向量下
        '''
        new_data=[]
        ind=0
        bag_dict={}
        for rel,mention,e in loader.train_data:#访问每个包
            for sentence in mention:
                alp=ep_alpha[ind]
                #predict_label=heapq.nlargest(len(rel),range(len(alp)),alp.__getitem__)
                #if len(predict_label)>1 and 0 in predict_label:
                #    predict_label.remove(0)
                #predict_label=tuple(predict_label)
                predict_label=tuple(rel)
                if np.max(alp)>0.6:
                    predict_label=(np.argmax(alp),)
                rel=tuple(rel)
                if (predict_label,e,rel) not in bag_dict:
                    bag_dict[(predict_label,e,rel)]=[]
                bag_dict[(predict_label,e,rel)].append(sentence)
                ind+=1
        for key in bag_dict:
            new_data.append((key[0],bag_dict[key],key[1]))
        logger.print('Total bag num before changing:{}'.format(len(loader.train_data)))
        loader.train_data=new_data
        logger.print('Total bag num after changing:{}'.format(len(loader.train_data)))
        print('total_size:',ind,sum([len(d[1]) for d in new_data]))

    def change_label_continue(self,ep_alpha,loader,logger):
        '''
        句子连续n轮都和这个向量接近才转移
        '''
        new_data=[]
        ind=0
        bag_dict={}
        
        for rel,mention,e in loader.train_data:#访问每个包
            for sentence in mention:
                alp=ep_alpha[ind]
                predict=np.argmax(alp)  #可以卡阈值
                if np.max(alp) < 0.8:
                    sentence[2]=[-1,0]
                else:
                    if sentence[2][0]==-1:
                        sentence[2]=[predict,1]
                    else:
                        if predict==sentence[2][0] :
                            sentence[2][1]+=1
                        else:
                            sentence[2]=[predict,1]
                predict_label=tuple(rel)
                if sentence[2][1]>=self.continue_n: #and sentence[2][0]!=0:
                    predict_label=(sentence[2][0],)
               
                rel=tuple(rel)
                if (predict_label,e,rel) not in bag_dict:
                    bag_dict[(predict_label,e,rel)]=[]
                bag_dict[(predict_label,e,rel)].append(sentence)
                ind+=1
        for key in bag_dict:
            new_data.append((key[0],bag_dict[key],key[1]))
        
        logger.print('Total bag num before changing:{}'.format(len(loader.train_data)))
        loader.train_data=new_data
        logger.print('Total bag num after changing:{}'.format(len(loader.train_data)))
        print('total_size:',ind,sum([len(d[1]) for d in new_data]))


    def evaluate_train(self,ep_probs,rel_dis):
        loader=self.loader
        test_dis={}
        logger=self.logger
        for i,dat in enumerate(loader.train_data):
            topn=heapq.nlargest(len(dat[0]),range(len(ep_probs[i])),ep_probs[i].__getitem__)
            for top in topn:
                test_dis[top]=test_dis.get(top,0)+1
        logger.print('训练集真实标签分布:{r},{s}'.format(r=rel_dis,s=sum(rel_dis.values())))
        logger.print('训练集预测标签分布:{r},{s}'.format(r=test_dis,s=sum(test_dis.values())))

    def train(self,
              name = 'bagrnn',
              epochs = 10,
              log_dir = './log',
              model_dir = './model',
              stats_dir = './stats',
              restore_dir = None,
              test_gap = 5,
              report_rate = 0.5,
              gpu_usage = 0.9,
              max_eval_rel = None):
        # [NOTE] assume model is already built
        loader = self.loader
        self.logger = logger = MyLogger(stats_dir)
        logger.print("\n\n\n\n", False)
        logger.print(">>>>>>>>>>>>>>>>> New Start <<<<<<<<<<<<<<<<<<<<<", False)
        comment='加入注意力损失,权重为:'+str(self.model.att_loss_weight)
        comment='使用softmaxloss,池化层lstm,三层dense,无额外损失,200轮'
        print(comment)
        logger.print_model_spec(self.model,comment)

        # build tensorboard monitoring variables
        tf.summary.scalar('miml-loss', self.model.raw_loss)
        if self.model.adv_eps is not None:
            tf.summary.scalar('adversarial-loss', self.model.loss)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(log_dir + '/train')
        self.test_writer = tf.summary.FileWriter(log_dir + '/test')

        # training related
        self.loss = self.model.loss
        self.learning_rate = tf.placeholder(tf.float32)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        if self.clip_grad is None:
            self.train_op = optimizer.minimize(self.loss)
        else:
            self.train_op = mc.minimize_and_clip(optimizer, self.loss, clip_val=self.clip_grad,
                                                 exclude=self.model.exclude_clip_vars)

        # Training
        total_batches = loader.train_batches  # total batches 1个数据集有多少个batch
        rel_dis={}
        for dat in loader.train_data:
            for r in dat[0]:
                rel_dis[r]=rel_dis.get(r,0)+1   
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_usage)
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(allow_soft_placement=True,
                                gpu_options=gpu_options)
        saver = tf.train.Saver(max_to_keep=epochs//2+1)  # keep the last half of the models
        global_batch_counter = 0
        global_ep_counter = 0
        early_stop_rounds=5
        best_auc=0
        best_count=0
        
        with tf.Session(config=config) as sess:
            try:
                if restore_dir is None:
                    sess.run(tf.global_variables_initializer())
                else:
                    logger.print('Warmstart from {} ...'.format(restore_dir))
                    saver.restore(sess, restore_dir)
                    with open(restore_dir+'-counter.pkl','rb') as f:
                        global_batch_counter, global_ep_counter = pickle.load(f)

                # Run Training

                accu_batch = 0
                accu_loss = 0
                accu_att_loss=0
                accu_ins_loss=0
                accu_weight_loss=0
                if epochs <= 0:
                    # only perform evaluation
                    loader.new_epoch()
                    self.evaluate(sess, stats_dir + '/{m}_eval_stats_full.pkl'.format(m=name), incl_conf=True)
                
                for ep in range(epochs):
                    global_ep_counter += 1
                    iter_n = 0
                    ts = time.time()
                    cur_rate = report_rate
                    logger.print ('>>> Current Starting Epoch#{}'.format(global_ep_counter))
                    loader.new_epoch()
                    # FOR DEBUG
                    #if ep == 0:
                    #    self.evaluate(sess, stats_dir + '/random_eval_stats.pkl')
                    #    exit(0)
                    ep_alpha=[]
                    ep_probs=[]
                    ep_weight=[]
                    ep_y=[]
                    ep_remain_x=[]
                    self.count_size=0
                    self._bat=0
                    total_batches=loader.train_batches
                    print('total_batches:',total_batches)
                    for i in range(total_batches):
                        iter_n += 1
                        global_batch_counter += 1
                        self._bat+=1
                        fd=self.feed_dict('train')
                        #print(fd[self.model.X][:5])
                        #alpha1,loss1=sess.run([self.model.alpha,self.model.loss],feed_dict=fd)
                        summary,  _ = \
                            sess.run([self.merged, self.train_op],
                                     feed_dict=fd)
                        
                        #print(fd[self.model.X][:5])
                        #alpha2,loss2=sess.run([self.model.alpha,self.model.loss],feed_dict=fd)
                        #print(len(alpha1),type(alpha1))
                        #print(alpha1[:5])
                        #print(len(batch_alpha),type(batch_alpha))
                        #print(batch_alpha[:5])
                        #print(len(alpha2),type(alpha2))
                        #print(alpha2[:5])
                        #iprint(loss1,loss2)
                        #print(c_loss,att_loss)
                        #exit(0)
                        batch_alpha,c_loss,att_loss,ins_alpha_loss,weight_loss,m_probs,true_dense_loss,dense_loss,batch_weight_list,remain_x=sess.run([self.model.alpha,self.model.loss,\
                            self.model.att_loss,self.model.ins_alpha_loss,self.model.weight_loss,self.model.probs,self.model.true_dense_loss,self.model.dense_loss,\
                            self.model.weight_list,self.model.remain_x],feed_dict=fd)
                        ep_alpha+=batch_alpha[:self.effective_size].tolist()  #effective_size是有效的句子数，effective是包
                        ep_probs+=m_probs[:self.effective].tolist()
                        ep_y+=fd[self.model.Y][:self.effective,:].tolist()
                        ep_weight+=batch_weight_list[:self.effective_size].tolist()
                        ep_remain_x+=remain_x[:self.effective_size].tolist()
                        #ep_weight+=batch_weight_list.tolist()
                        accu_batch += 1
                        accu_loss += c_loss
                        accu_att_loss += att_loss
                        accu_ins_loss += ins_alpha_loss
                        accu_weight_loss += weight_loss
                        self.train_writer.add_summary(summary, global_batch_counter)
                        # if iter_n % int(total_batches/5) == 0: #验证集，用于展示的
                        #     summary = sess.run(self.merged,
                        #                        feed_dict=self.feed_dict('test'))
                        #     self.test_writer.add_summary(summary, global_batch_counter)

                        if iter_n >= total_batches * cur_rate:
                            logger.print(' --> {x} / {y} finished! ratio = {r},   elapsed = {t}'.format(
                                x = iter_n,
                                y = total_batches,
                                r = (1.0 * iter_n) / total_batches,
                                t = time.time() - ts))
                            logger.print('   > loss = %.6f, attention loss = %.6f, instance attention loss %.6f, weight_loss %.6f' % (accu_loss / accu_batch,accu_att_loss / accu_batch,accu_ins_loss / accu_batch,accu_weight_loss/accu_batch))
                            accu_batch = 0
                            accu_loss = 0
                            accu_att_loss = 0
                            accu_ins_loss = 0
                            accu_weight_loss=0
                            cur_rate += report_rate
                            logger.print('max batch size: %d' %self.max_batch_size)
                            logger.print(true_dense_loss)
                            logger.print(dense_loss)

                        if self.lrate_decay > 0 and \
                           global_batch_counter % self.lrate_decay == 0:
                            self.update_learning_param()  # learning rate decay

                    logger.print('-------------------->>len of ep_alpha:{}'.format(len(ep_alpha)))  #总句子数
                    logger.print('-------------------->>len of ep_weight:{}'.format(np.array(ep_weight).shape))
                    logger.print('-------------------->>len of ep_remain_x:{}'.format(np.array(ep_remain_x).shape))
                    logger.print('-------------------->>one epoch size:{}'.format(self.count_size)) #一个ep的句子数，因为会不够的话会补上头部数据，所以略大
                    logger.print('-------------------->>len of ep_probs:{}'.format(len(ep_probs)))#总包数
                    logger.print('-------------------->>len of ep_y:{}'.format(len(ep_y)))
                    #logger.print(ep_weight[:5])
                    #logger.print(ep_weight[-5:])
                    # with tf.variable_scope('discriminative-net', reuse=True):
                    #         print(sess.run(tf.get_variable('noise_matrix')))

                    save_path = saver.save(sess,
                                           model_dir+'/{m}_ep{i}'.format(m=name,i=global_ep_counter))
                    with open(model_dir+'/{m}_ep{i}-counter.pkl'.format(m=name,i=global_ep_counter), 'wb') as f:
                        pickle.dump([global_batch_counter, global_ep_counter], f)
                    logger.print("Model saved in file: %s" % save_path)

                    loader.train_ptr=0
                    loader.test_ptr=0
                    loader.eval_ptr=0
                    test_auc=eval_auc=0
                    #test_auc=self.evaluate(sess, stats_dir + '/{m}_eval_stats_ep{i}.pkl'.format(m=name, i=global_ep_counter),
                    #              max_relations=max_eval_rel,feed_type='train')
                    eval_auc=self.evaluate(sess, stats_dir + '/{m}_eval_stats_ep{i}.pkl'.format(m=name, i=global_ep_counter),
                                  max_relations=max_eval_rel)
                    ts=time.time()
                    

                    self.auc_list.append((test_auc,eval_auc))                    
                    #self.evaluate_train(ep_probs,rel_dis)
                    if ep>=100000:
                        self.change_label_continue(ep_alpha,loader,logger)
                        logger.print('#########Label has changed!! cost time:{}'.format(time.time()-ts))
                    # logger.print('----->'+str(ep_y[-5:]))
                    # logger.print(ep_weight[-5:])
                    # logger.print('----->'+str(ep_y[:5]))
                    # logger.print(ep_weight[:5])

                    # if eval_auc > best_auc:
                    #     best_auc = eval_auc
                    #     best_count = 0
                    # else:
                    #     best_count +=1
                    #     if best_count > early_stop_rounds:
                    #         break

            except KeyboardInterrupt:
                logger.print('Training Interrupt!!')
                save_path = saver.save(sess,
                                       model_dir+'/{m}_killed_iter{i}'.format(m=name,i=global_batch_counter))
                with open(model_dir+'/{m}_killed_iter{i}-counter.pkl'.format(m=name,i=global_batch_counter), 'wb') as f:
                    pickle.dump([global_batch_counter, global_ep_counter], f)
                logger.print('Model saved in file: %s' % save_path)
            
            with open(stats_dir+'/auc_stats.txt','w') as f:
                f.write('test_auc'+'\t'+'eval_auc'+'\n')
                for i,j in self.auc_list:
                    f.write(str(i)+'\t'+str(j)+'\n')
                f.write('\n\n\n\n')

            self.auc_list.sort(key=lambda x:x[1],reverse=True)
            logger.print('--------->>>>>best auc:'+str(self.auc_list[0][1]))
            logger.print(comment)
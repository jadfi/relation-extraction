import numpy as np
import pickle
import time

"""
bag data loader for MIML
"""

class loader:
    def __init__(self, relation_file, label_data_file, unlabel_data_file = None, group_eval_data_file = None,
                 embed_dir = None, word_dir = None,
                 n_vocab = 80000, valid_split = 1000, max_len = 119, split_seed = 0,
                 use_DS_data = True):
        """
          relation_file: dictionary index of relation
          label_data_file: (list of) filename(s) of label_data pickle
                           we get <valid_split> from the first file for validation
          unlabel_data_file: filename of unlabel_data pickle
          group_eval_data_file: filename of evaluation data, which is in the form of bags of mentions for entities
          embed_dir: if None, then train from scratch, then word_dir must be not None
          word_dir: when embed_dir is not None, just ignore this
          n_vocab: vocab_size, default 80000
          valid_split: number of label data for validation
          max_len: maximum length of sentences
        """
        np.random.seed(split_seed)
        self.n_vocab = n_vocab
        self.n_test = valid_split
        self.max_len = max_len
        # load data
        print('load label data ....')
        ts = time.time()
        # relation index
        with open(relation_file, 'rb') as f:
            self.rel_ind = pickle.load(f)
        self.rel_name = dict()
        self.rel_num = len(self.rel_ind)
        for k,i in self.rel_ind.items():
            self.rel_name[i] = k
        #关系名称和序号
        # labeled data
        self.label_data = []
        self.test_data = []
        if not isinstance(label_data_file,list):
            label_data_file = [label_data_file]
        for i, fname in enumerate(label_data_file):
            with open(fname, 'rb') as f:
                text, entity, pos, rel = pickle.load(f)
            #过滤掉句子太长的
            curr_data = [(t,e,p,self.rel_ind[r]) for t,e,p,r in zip(text,entity,pos,rel) if len(t)<max_len]
            np.random.shuffle(curr_data)
            # for t,e,p,r in curr_data:
            #     if p[0] == p[2]:
            #         print(t,e,p,r)
            #         exit(0)
            # exit(0)
            if i == 0:
                # split valid data set
                self.test_data = curr_data[:valid_split]
                curr_data = curr_data[valid_split:]
            self.label_data += curr_data
        self.label_data_raw = self.label_data.copy()
        self.test_data_raw = self.test_data.copy()
        print('  -> done! elapsed = {}'.format(time.time()-ts))

        # unlabeled data
        self.unlabel_data = []
        if unlabel_data_file is not None and use_DS_data:
            print('load unlabel data ...')
            ts = time.time()
            if not isinstance(unlabel_data_file,list):
                unlabel_data_file = [unlabel_data_file]
            for fname in unlabel_data_file:
                with open(fname, 'rb') as f:
                    text, entity, pos, rel = pickle.load(f)
                self.unlabel_data += [(t,e,p,self.rel_ind[r]) for t,e,p,r in zip(text,entity,pos,rel) if len(t)<max_len]
                
            print('  -> done! elapsed = {}'.format(time.time()-ts))

        # evaluation data
        self.eval_data = None
        if group_eval_data_file is not None:
            print('load evaluation data ...')
            ts = time.time()
            self.eval_data = []
            with open(group_eval_data_file,'rb') as f:
                G = pickle.load(f)
            #感觉是每个实体对包含的句子来存的
            for e, dat in G.items():
                rel = set([self.rel_ind[r] for r in dat[0]])
                mention = [[t, p,[-1,0]] for t,p in zip(dat[1], dat[2]) if len(t)<max_len]
                if len(mention) > 0:
                    self.eval_data.append((rel,mention))
            print('  -> done! elapsed = {}'.format(time.time()-ts))
        
        print('load dictionary and embedding...')
        #加载词和词向量
        ts = time.time()
        if embed_dir is None:
            self.embed = None
            with open(word_dir,'rb') as f:
                self.vocab,_ = pickle.load(f)
        else:
            with open(embed_dir,'rb') as f:
                self.embed, self.vocab = pickle.load(f)
            self.embed = self.embed[:n_vocab,:]
            self.embed_dim = self.embed.shape[1]
        self.vocab = self.vocab[:n_vocab]
        self.word_ind = dict(zip(self.vocab, list(range(n_vocab))))
        self.init_extra_word()
        print('  -> done! elapsed = {}'.format(time.time()-ts))

    def init_extra_word(self):
        n = self.n_vocab
        self.n_extra = 3
        self.unk,self.eos,self.start=n,n+1,n+2
        self.pad=self.word_ind['<pad>']
        self.vocab += ['<unk>','<eos>','<start>']

    def group_data(self, raw_data, merge_by_entity = False):
        if len(raw_data) == 0:
            return []
        # data: list of (t, e, p, r)
        # return:
        #    a list [(list of relation id, list of (text, position))]
        count_metion=0
        count_one_mention=0
        max_mention=0
        if not merge_by_entity:
            # every single instance becomes a group
            data = [([r], [[t, p,[-1,0]]],e) for t, e, p, r in raw_data]   #(-1,0)初始化，表示该句子连续0次和标签-1最相近
        else:
            # merge mentions by entity names
            group = dict()
            for t,e,p,r in raw_data:
                if e not in group:
                    group[e] = (set(), [])
                group[e][0].add(r)
                group[e][1].append([t, p,[-1,0]])
            data = []
            for e, p in group.items():
                rel = sorted(list(p[0]))
                #如果有多种关系，去掉NA
                if rel [0] == 0 and len(rel) > 1:
                    rel = rel[1:]
                if len(rel)>1:
                    continue
                mention = p[1]
                
                #为啥要扔掉大包，因为噪声多
                if len(mention) > 300:
                    #print('Warninng!!! Super Large Bag!! N = {d}, Rel = {r}'.format(d=len(mention), r = rel))
                    continue
                max_mention=max(max_mention,len(mention))
                count_metion+=len(mention)
                if len(mention) == 1:
                    count_one_mention+=1
                np.random.shuffle(mention)
                data.append((rel, mention,e))
                
                # for t in mention:
                #     data.append((rel,[t],e))

        np.random.shuffle(data)
        print('count_metion:',count_metion)
        print('count_ent:',len(data))
        print('count_one_mention:',count_one_mention)
        print('max_mention:',max_mention)
        return data

    def init_data(self, bag_batch, seed = 3137, merge_by_entity = True):
        np.random.seed(seed)
        # group data into bags
        self.label_data = self.group_data(self.label_data_raw, True)
        self.test_data = self.group_data(self.test_data_raw, False) # do not merge entities
        self.train_data = self.group_data(self.unlabel_data, True) + self.label_data
        np.random.shuffle(self.train_data)
        # init params
        self.bag_batch = bag_batch   # number of bags processed per iteration
        self.train_n = len(self.train_data)
        self.test_n = len(self.test_data)
        self.train_batches = (self.train_n + bag_batch - 1) // bag_batch
        self.test_batches = (self.test_n + bag_batch - 1) // bag_batch

    def ID(self, c):
        if c in self.word_ind:
            return self.word_ind[c]
        return self.unk

    def new_epoch(self):
        np.random.shuffle(self.train_data)
        self.train_ptr = 0
        self.test_ptr = 0
        self.eval_ptr = 0
        self.train_batches = (len(self.train_data) + self.bag_batch - 1) // self.bag_batch

    def get_bag_n(self,feed_type='evel'):
        if feed_type=='eval':
            return len(self.eval_data)
        elif feed_type=='train':
            return len(self.train_data)
        else:
            return len(self.test_data)

    def get_bag_info(self, k,feed_type='eval'):
        # return:
        #    positive rel IDs
        if feed_type=='eval':
            dat = self.eval_data[k]
        elif feed_type=='train':
            dat = self.train_data[k]
        else:
            dat = self.test_data[k]
        return dat[0]

    def next_batch(self,num,data_source = 'train'):
        L = self.max_len
        # get training batch
        if data_source == 'train':
            curr_ptr, data = self.train_ptr, self.train_data
        elif data_source == 'test':
            curr_ptr, data = self.test_ptr, self.test_data
        else: # evaluation
            curr_ptr, data = self.eval_ptr, self.eval_data
        n = len(data)
        if curr_ptr==0:
            print('------------->>> len of data:',n)
        effective = min(self.bag_batch, n - curr_ptr)  #有效数据
        curr_bags = [data[(curr_ptr + i) % n] for i in range(self.bag_batch)]  #data里面一个元素就是一个实体对的所有rel和mention,也就是一个包
        #%n 表示最后一个batch不够会回到前面取数据
        batch_size = sum([len(d[1]) for d in curr_bags])#batch_size是batch里所有包的所有句子数量
        effective_size=sum([len(d[1]) for d in curr_bags[:effective]])
        Y = np.zeros((self.bag_batch, self.rel_num), dtype=np.float32)  #包和关系的映射矩阵
        X = np.ones((batch_size, L), dtype=np.int32) * self.eos  #词矩阵，矩阵的值是每个词的编号
        E = np.zeros((batch_size, L), dtype=np.int32)  #实体矩阵，标记哪个词是实体，1表示第一个实体，2表示第二个
        length = np.zeros((batch_size, ), dtype=np.int32)  #每个句子的长度

        mask = np.zeros((batch_size, L), dtype=np.float32)  
        shapes = np.zeros((self.bag_batch + 1), dtype=np.int32)
        shapes[self.bag_batch] = batch_size  #shape里存的是一个包的起始位置,shape[i+1]-shape[i]就是包i的大小
        k = 0

        self.cached_pos = []

        for i, bag in enumerate(curr_bags):
            rel = bag[0]
            for r in rel:
                Y[i][r] = 1
            mention = bag[1]
            shapes[i] = k
            for j in range(len(mention)):
                text, pos ,_ = mention[j]
                length[k + j] = len(text) + 1
                mask[k + j, : len(text)] = 1  # ignore the last eos symbol
                for l, c in enumerate(text):
                    X[k + j, l] = self.ID(c)
                X[k + j, len(text)] = self.eos
                E[k + j, pos[0]:pos[1]] = 1
                E[k + j, pos[2]:pos[3]] = 2
                self.cached_pos.append(pos)
            k += len(mention)

        if data_source == 'train':
            self.train_ptr += self.bag_batch
        elif data_source == 'test':
            self.test_ptr += self.bag_batch
        else:  # evaluation
            self.eval_ptr += self.bag_batch

        return effective,effective_size, X, Y, E, length, shapes, mask,batch_size



import numpy as np
import pickle
import time

"""
bag data loader for MIML
"""

class miml_loader():
    def __init__(self, relation_file, train_file, test_file,
                 embed_dir = None,
                 n_vocab = 80000, max_len = 145, max_bag_size = None):  # max_bag_size = 2500
        """
          relation_file: dictionary index of relation
          train_file, test_file: file name for the training/testing data
          group_eval_data_file: filename of evaluation data, which is in the form of bags of mentions for entities
          embed_dir: if None, then train from scratch, then word_dir must be not None
          word_dir: when embed_dir is not None, just ignore this
          n_vocab: vocab_size, default 80000
          max_len: maximum length of sentences
        """
        self.n_vocab = n_vocab
        self.max_len = max_len

        # load data
        print('load relation index ....')
        ts = time.time()
        # relation index
        with open(relation_file, 'rb') as f:
            self.rel_ind = pickle.load(f)
        assert ('NA' in self.rel_ind)
        assert (self.rel_ind['NA'] == 0)
        self.rel_name = dict()
        self.rel_num = len(self.rel_ind)
        for k, i in self.rel_ind.items():
            self.rel_name[i] = k
        print('  -> done! elapsed = {}'.format(time.time() - ts))

        print('load dictionary and embedding...')
        ts = time.time()
        with open(embed_dir, 'rb') as f:
            self.embed, self.vocab = pickle.load(f)
        self.embed = self.embed[:n_vocab, :]
        self.embed_dim = self.embed.shape[1]
        self.vocab = self.vocab[:n_vocab]
        self.word_ind = dict(zip(self.vocab, list(range(n_vocab))))
        self.init_extra_word()
        print('  -> done! elapsed = {}'.format(time.time() - ts))

        # train and test data
        def load_group_data(filename,single=False):
            with open(filename,'rb') as f:
                group = pickle.load(f)
            data = []
            count=0
            for e, dat in group.items():
                rel = set([self.rel_ind[r] for r in dat[0]])
                rel = sorted(list(rel))
                if rel [0] == 0 and len(rel) > 1:
                    rel = rel[1:]
                if len(rel)>1:
                    continue
                if single:
                    for t, p in zip(dat[1], dat[2]):
                        if len(t) < max_len:
                            mention = [[t, p,[-1,0]]]
                            data.append((rel,mention,e))
                    continue
                mention = [[t, p,[-1,0]] for t, p in zip(dat[1], dat[2]) if len(t) < max_len]
                count+=len(mention)
                if len(mention) > 0:
                    if (max_bag_size is not None) and (len(mention) > max_bag_size):
                        # split into 3 small bags, max training bag size is 5500
                        np.random.shuffle(mention)
                        n = len(mention)
                        ptr = 0
                        while ptr < n:
                            l = ptr
                            r = ptr + max_bag_size
                            if r >= n:
                                data.append((rel, mention[-max_bag_size:],e))
                                break
                            else:
                                data.append((rel, mention[l:r],e))
                            ptr = r
                    else:
                        data.append((rel, mention,e))
            print('len of data:',count)
            return data

        print('load training data ...')
        ts = time.time()
        self.train_data = load_group_data(train_file,False)
        print('  -> done! elapsed = {}'.format(time.time() - ts))

        print('load testing data ...')
        ts = time.time()
        self.eval_data = load_group_data(test_file,False)
        print('  -> done! elapsed = {}'.format(time.time() - ts))


    def init_extra_word(self):
        n = self.n_vocab
        self.n_extra = 3
        self.unk,self.eos,self.start=n,n+1,n+2
        self.vocab += ['<unk>','<eos>','<start>']

    def init_data(self, bag_batch, seed = 3137):
        np.random.seed(seed)
        # group data into bags
        np.random.shuffle(self.train_data)
        np.random.shuffle(self.eval_data)
        # init params
        self.bag_batch = bag_batch   # number of bags processed per iteration
        self.train_n = len(self.train_data)
        self.eval_n = len(self.eval_data)
        self.train_batches = (self.train_n + bag_batch - 1) // bag_batch
        self.eval_batches = (self.eval_n + bag_batch - 1) // bag_batch

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




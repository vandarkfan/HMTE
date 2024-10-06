# -- coding: utf-8 --

import numpy as np
import torch
import math
from collections import defaultdict
from collections import Counter
def top_n_elements(a, n):
    counter = Counter(a)
    top_three = counter.most_common(n)
    return [item[0] for item in top_three]
class Data:
    def __init__(self, data_dir='./data/JF17K/'):
        print('Loading the string facts from raw data...')
        train_strfacts, valid_strfacts, test_strfacts, rel_list, ent_list, self.max_ary = self.load_strfacts(data_dir=data_dir)
        self.rel2id, self.ent2id = self.str2id(rel_list, ent_list)
        train_facts = self.facts_str2idx(train_strfacts)
        valid_facts = self.facts_str2idx(valid_strfacts)
        test_facts = self.facts_str2idx(test_strfacts)

        all_facts = [[] for _ in range(2, self.max_ary + 1)]
        for ary in range(2, self.max_ary + 1):
            all_facts[ary - 2] = train_facts[ary - 2] + valid_facts[ary - 2] + test_facts[ary - 2]

        # 数据集测量
        # rel_ayy = {}
        # ent_ayy = {}
        # for ary in range(2, self.max_ary + 1):
        #     tmp_fact = all_facts[ary - 2]
        #     for i in range(len(tmp_fact)):
        #         tmp_tuple = tmp_fact[i]
        #         if tmp_tuple[0] not in rel_ayy.keys():
        #             rel_ayy[tmp_tuple[0]] = {}
        #             rel_ayy[tmp_tuple[0]][ary] = 1
        #         else:
        #             if ary not in rel_ayy[tmp_tuple[0]].keys():
        #                 rel_ayy[tmp_tuple[0]][ary] = 1
        #             else:
        #                 rel_ayy[tmp_tuple[0]][ary] += 1
        #         for j in range(1, ary + 1):
        #             if tmp_tuple[j] not in ent_ayy.keys():
        #                 ent_ayy[tmp_tuple[j]] = {}
        #                 ent_ayy[tmp_tuple[j]][ary] = 1
        #             else:
        #                 if ary not in ent_ayy[tmp_tuple[j]].keys():
        #                     ent_ayy[tmp_tuple[j]][ary] = 1
        #                 else:
        #                     ent_ayy[tmp_tuple[j]][ary] += 1
        #
        # ent_fouces = {}
        # dataset_fouces = 0
        # for ent in ent_ayy.keys():
        #     ent_fouces[ent] = 0
        #     tmp = 0
        #     count = 0
        #     for ary in ent_ayy[ent].keys():
        #         count = count + ent_ayy[ent][ary]
        #     for ary in ent_ayy[ent].keys():
        #         ent_ayy[ent][ary] = ent_ayy[ent][ary] / count
        #     for ary in ent_ayy[ent].keys():
        #         tmp = max(tmp, math.exp(ent_ayy[ent][ary]))
        #         ent_fouces[ent] += math.exp(ent_ayy[ent][ary])
        #     ent_fouces[ent] = tmp / ent_fouces[ent]
        #     dataset_fouces = dataset_fouces - math.log(ent_fouces[ent])
        # dataset_fouces = dataset_fouces / len(ent_ayy.keys())


        self.all_er_vocab_list = self.get_er_vocab(all_facts, self.max_ary)
        self.train_er_vocab_list = self.get_er_vocab(train_facts, self.max_ary)
        self.ent_relnel = self.get_relnel(train_facts)
        self.train_facts = [np.array(x).astype(np.int32) for x in train_facts]
        self.valid_facts = [np.array(x).astype(np.int32) for x in valid_facts]
        self.test_facts = [np.array(x).astype(np.int32) for x in test_facts]
        print('Loading data finished!')

    def get_relnel(self, train_facts):
        ent_dict_train={}
        for i in range(len(train_facts)):
            list_ary = train_facts[i]
            for j in range(len(list_ary)):
                triple = list_ary[j]
                for k in range(1, i+3):
                    ent = triple[k]
                    if ent not in ent_dict_train.keys():
                        ent_dict_train[ent] = []
                    ent_dict_train[ent].append(triple[0])
        ent_dict_three = torch.ones([len(self.ent2id), 3]) * int(len(self.rel2id))
        for j in ent_dict_train:
            tmp = list(set(ent_dict_train[j]))
            n = len(tmp)
            if n > 3:
                ent_dict_three[j] = torch.IntTensor(top_n_elements(ent_dict_train[j], 3))
            elif n == 3:
                ent_dict_three[j] = torch.IntTensor(tmp)
            elif n == 2:
                ent_dict_three[j][0] = tmp[0]
                ent_dict_three[j][1] = tmp[1]
            else:
                ent_dict_three[j][0] = tmp[0]
        return ent_dict_three



    def load_strfacts(self, data_dir='./data/JF17K/'):
        dataset = data_dir.strip().split('/')[-2]
        train_strfacts, valid_strfacts, test_strfacts =  [], [], []
        ent_list, rel_list = [], []
        max_ary = 0
        if dataset in ['JF17K', 'FB-AUTO', 'WikiPeople1', 'WikiPeople-3', 'WikiPeople-4', 'JF17K-3', 'JF17K-4']:
            for filename in ['train', 'valid', 'test']:
                with open(data_dir+filename+'.txt', 'r') as f:
                    lines = f.readlines()
                for k in range(len(lines)):
                    line = lines[k].strip().split('\t')
                    if len(line)-1 > max_ary:
                        max_ary = len(line)-1
                    rel = line[0]
                    rel_list.append(rel)
                    tmp_fact = [rel]
                    for id, ent in enumerate(line[1:]):
                        tmp_fact.append(ent)
                        ent_list.append(ent)
                    eval(filename+'_strfacts').append(tmp_fact)
        elif dataset in ['WikiPeople']:
            for filename in ['train', 'valid', 'test']:
                with open(data_dir+'n-ary_'+filename+'.json', 'r') as f:
                    lines = f.readlines()
                for k in range(len(lines)):
                    line = lines[k]
                    dic = eval(line)
                    if dic['N'] > max_ary:
                        max_ary = dic['N']
                    tmp_fact = []
                    str_list = [x.strip('":') for x in line.strip('\n{}').split()]
                    role_list = []
                    for tmp_role in str_list:
                        if tmp_role[0] == 'P':
                            if type(dic[tmp_role]) != list:
                                role_list.append(tmp_role)
                                tmp_fact.append(dic[tmp_role])
                                ent_list.append(dic[tmp_role])
                            else:
                                for val0 in dic[tmp_role]:
                                    role_list.append(tmp_role)
                                    tmp_fact.append(val0)
                                    ent_list.append(val0)
                    sorted_role = sorted(enumerate(role_list), key=lambda x:x[1])
                    role_list = [i[1] for i in sorted_role]
                    idx = [i[0] for i in sorted_role]
                    tmp_fact = [tmp_fact[id] for id in idx]
                    tmp_rel = '/'.join(role_list)
                    tmp_fact = [tmp_rel] + tmp_fact
                    rel_list.append(tmp_rel)
                    eval(filename+'_strfacts').append(tmp_fact)
        elif dataset in ['WN18', 'FB15K', 'WN18RR', 'FB15K-237']:
            for filename in ['train', 'valid', 'test']:
                with open(data_dir+filename, 'r') as f:
                    lines = f.readlines()
                for k in range(len(lines)):
                    line = lines[k].strip().split('\t')
                    if len(line)-1 > max_ary:
                        max_ary = len(line)-1
                    rel = line[1]
                    rel_list.append(rel)
                    tmp_fact = [rel, line[0], line[2]]
                    eval(filename+'_strfacts').append(tmp_fact)
                    ent_list.append(line[0])
                    ent_list.append(line[2])
        else:
            print('Hint: The used dataset is not predefined, please add it in load_data.py based on its form...')

        return train_strfacts, valid_strfacts, test_strfacts, sorted(list(set(rel_list))), sorted(list(set(ent_list))), max_ary

    def str2id(self, rel_list, ent_list):
        rel2id, ent2id = {}, {}
        for id, rel in enumerate(rel_list):
            rel2id[rel] = id
        for id, ent in enumerate(ent_list):
            ent2id[ent] = id
        return rel2id, ent2id

    def facts_str2idx(self, train_strs):
        train_facts = [[] for _ in range(2, self.max_ary+1)]
        for fact_strs in train_strs:
            tmp_fact = [self.rel2id[fact_strs[0]]] + [self.ent2id[x] for id, x in enumerate(fact_strs[1:])]
            train_facts[len(tmp_fact)-1-2].append(tmp_fact)
        return train_facts

    def get_er_vocab(self, facts, max_ary):
        er_vocab_list = [[defaultdict(list) for _ in range(ary)] for ary in range(2, max_ary + 1)]
        for ary in range(2, max_ary + 1):
            for k in range(len(facts[ary - 2])):
                for miss_ent_domain in range(1, ary + 1):
                    x = facts[ary - 2][k]
                    key_str = [int(x[0])]
                    for i, x0 in enumerate(x[1:]):
                        if i + 1 != miss_ent_domain:
                            key_str.append(int(x0))
                        else:
                            key_str.append(int(miss_ent_domain * 111111))
                    er_vocab_list[ary-2][miss_ent_domain-1][tuple(key_str)].append(int(x[miss_ent_domain]))
        return er_vocab_list

if __name__ == '__main__':
    data = Data(data_dir="./data/FB-AUTO/")

# -*- coding: utf-8 -*-

import random
from random import shuffle
random.seed(1)


#使用中文的停用词，这里使用百度的，更多见 "./data/stopwords/"
with open(r'.\utils\stopword.txt', 'r', encoding="utf8") as g:
    words = g.readlines()
_stop_words = [i.strip() for i in words]
_stop_words.extend(['.','（','）','-'])
_stop_words.remove("，")
_stop_words.remove("。")

#cleaning up text
import re
def get_only_chars(line):
    # 英文的清洗逻辑，这块看自己需求用或者不用吧，但是中文肯定是不能用的。
    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line

########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
# 同义词替换，原版本用的wordnet，也有中文的，第一次需要下载词库
########################################################################

import synonyms
import nltk
# 第一次使用请打开下面这一行，下载中文的wordnet
# nltk.download('omw')
from nltk.corpus import wordnet 

def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in _stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            #print("replaced", random_word, "with", synonym)
            num_replaced += 1
        if num_replaced >= n: #only replace up to n words
            break

    #this is stupid but we need it, trust me
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words

def get_synonyms(word):
    # 这里使用了word_net + synonyms, 将两者的同义词召回做合并
    synonyms_word = set()
    for syn in wordnet.synsets(word, lang='cmn'):
        synonyms_word = set(syn.lemma_names('cmn'))
    for w in synonyms.nearby(word)[0]:
        synonyms_word.add(w)
    return list(synonyms_word)

########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words, p):

    #obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words

    #randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    #if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    return new_words

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words

def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
    return new_words

########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################

def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words

def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)

########################################################################
# main data augmentation function
########################################################################

def eda(words, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=2):
    num_words = len(words)
    
    augmented_sentences = []
    num_new_per_technique = int(num_aug/2)
    n_sr = max(1, int(alpha_sr*num_words))
    n_ri = max(1, int(alpha_ri*num_words))
    n_rs = max(1, int(alpha_rs*num_words))

    # #sr 同义词替换
    # for _ in range(num_new_per_technique):
    #     a_words = synonym_replacement(words, n_sr)
    #     augmented_sentences.append(a_words)

    # #ri 随机插入单词
    # for _ in range(num_new_per_technique):
    #     a_words = random_insertion(words, n_ri)
    #     augmented_sentences.append(a_words)

    #rs 随机交换单词
    for _ in range(num_new_per_technique):
        a_words = random_swap(words, n_rs)
        augmented_sentences.append(a_words)

    #rd 随机删除单词
    for _ in range(num_new_per_technique):
        a_words = random_deletion(words, p_rd)
        augmented_sentences.append(a_words)

    # augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
    shuffle(augmented_sentences)

    #trim so that we have the desired number of augmented sentences
    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

    return augmented_sentences


words = ['2018', '年', '月', '15', '日', '14', '时', '10', '分许', '，', '被告人', '莫新国', '酒后', '驾驶', '湘', 'A', '×', '×', '×', '×', '×', '号', '小型', '轿车', '长沙市', '天心区', '伊莱克斯', '大道', '由南', '往北', '行驶', '水电', '八局', '基地', '路段', '时', '该处', '执勤', '长沙市', '公安局', '交通警察', '支队', '民警', '检查', '，', '现场', '酒精', '吹气', '检测', '，', '测试', '结果显示', '血液', '中', '乙醇', '含量', '195', '毫克', '／', '100', '毫升', '，', '随即', '被告人', '莫新国', '交警', '带', '湖南省', '融城', '医院', '抽取', '血样', '，', '血样', '送', '长沙市', '公安局', '物证', '鉴定', '检验', '，', '检验', '，', '血液', '中', '乙醇', '含量', '201.1', '毫克', '／', '100', '毫升', '。', '2009', '年', '11', '月', '15', '日', '，', '被告人', '莫新国经', '长沙市', '残疾人', '联合会', '审核', '精神', '残疾人', '。', '2018', '年', '月', '28', '日', '，', '湖南省', '芙蓉', '司法鉴定', '中心', '鉴定', '，', '被告人', '莫新国', '作案', '时', '处于', '普通', '醉酒', '状态', '，', '实施', '危害', '行为', '时有', '完全', '刑事责任', '能力', '。', '2018', '年', '月', '30', '日', '，', '被告人', '莫新国', '主动', '公安机关', '投案', '，', '归案', '如实', '供述', '罪行', '。']
print(eda(words))
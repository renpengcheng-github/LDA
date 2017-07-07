# -*- coding:utf-8 -*-
import numpy as np
import MySQLdb
from numpy import *
import ConfigParser
import random
import codecs
import os
import logging.config
logger = logging.getLogger()

class lda_model:
    def __init__(self, dbname,words_count,docs_count,doc,url_location):
        self.conn = MySQLdb.connect(user='root', passwd='mysql', host='localhost', db=dbname, charset="utf8")
        self.con = self.conn.cursor()  # con  replace  cur
        #-------------------------------------------------------------------------------------------------------------------
        #初始化变量
        self.K = 10
        self.beta = 0.1
        self.alpha = 0.1
        self.iter_times = 30#迭代次数

        self.docs = doc  # docs 为urlid 文档的真正编号 遍历时以下标遍历 字典结构
        #self.top_words_num = 20
        self.docs_count=docs_count  #len(self.docs)
        self.words_count=words_count  #wordcount是词汇总数（不重复）0~N 共N+1个数
        # nd[doc_count(M)][K],每个doc中各个topic的词的总数
        # ndsum[doc_count(M)],每各doc中词的总数
        # nw[wordcount(V)][K],词word在主题topic上的分布
        # nwsum[K],每各topic的词的总数

        self.nd = np.zeros((docs_count, self.K), dtype="int")
        # [[0 for t in range(K)] for d in range(docs_count) ]
        self.ndsum = np.zeros(docs_count, dtype="int")
        # [0 for d in range(docs_count)]
        self.nw = np.zeros((words_count, self.K), dtype="int")
        # [[0 for t in range(K)] for d in range(words_count) ]
        self.nwsum = np.zeros(self.K, dtype="int")
        # [0 for d in range(k)]

        self.p = np.zeros(self.K,dtype=float)
        # p,概率向量 double类型，存储采样的临时变量
        #self.z = np.array([[0 for m in xrange(docs_count)] for w in xrange(words_count)])
        self.z =[[0 for y in xrange(len(url_location[doc[m]]))] for m in xrange(docs_count)]
        # M*doc.size()，文档中词的主题分布

        self.theta = np.array([[0.0 for y in xrange(self.K)] for x in xrange(docs_count)])
        self.phi = np.array([[0.0 for y in xrange(words_count)] for x in xrange(self.K)])
        self.url_location=url_location
        # 文件变量
        # 分好词的文件trainfile
        # 词对应id文件wordidmapfile
        # 文章-主题分布文件thetafile
        # 词-主题分布文件phifile
        # 每个主题topN词文件topNfile
        # 最后分派结果文件tassginfile
        # 模型训练选择的参数文件paramfile
        #
        self.path = 'lda_logfile/'
        self.phifile = self.path + 'model_phi.dat'
        self.thetafile = self.path + 'model_theta.dat'
        self.topicwordfile=self.path + 'model_topicword.dat'
        self.test_topic=self.path + 'test_topic.dat'
        self.test_word=self.path + 'test_word.dat'

    def __del__(self):
        self.con.close()
    def dbcommit(self):
        self.conn.commit()

    # -------------------------------------------------------------------------------------------------------------------
    #1数据输入和Topic随机初始化：
    def data_preparing(self):
        # 初始化阶段：随机先分配类型
        print 'preparing data.....'
        for m in xrange(self.docs_count): #docs里面存储的是真正的urlid  self.docs_count=len(doc)
            true_urlid=self.docs[m]
            #------------------------------------------------------------------------------------
            #res=len(self.url_location[true_urlid])
            #self.ndsum[m] = res#某文档有多少个词
            #for i in self.url_location[true_urlid].keys():#对于每篇文档的每个locations
                #topicid = random.randint(0, self.K - 1)  # topic标号是0到k-1 随机分配topic
                #dict_wordid = self.url_location[true_urlid][i] #-1
                #self.z[i][m] = topicid  # 将此文档中每个词赋值一个topic
                #self.nw[dict_wordid][topicid] += 1
                #self.nwsum[topicid] += 1
                #self.nd[m][topicid] += 1
            # ------------------------------------------------------------------------------------
            res = len(self.url_location[true_urlid])
            self.ndsum[m] = res  # 某文档有多少个词
            for i in xrange(res):
                topicid = random.randint(0, self.K - 1)  # topic标号是0到k-1 随机分配topic
                dict_wordid = self.url_location[true_urlid][i] #-1
                self.z[m][i] = topicid  # 将此文档中每个词赋值一个topic
                self.nw[dict_wordid][topicid] += 1
                self.nwsum[topicid] += 1
                self.nd[m][topicid] += 1
            # ------------------------------------------------------------------------------------
            #self.con.execute('select count(wordid) from wordlocation where urlid=%d' %true_urlid)
            #res=self.con.fetchone()[0]#每篇doc文档有多少词（数量） 需要程序计数（从外界传入变量）
            #self.ndsum[m] =res
            #for w in xrange(int(res)):#此w是文档中的所有词语  即这篇文档一共有多少个词（包括重复词语）要对每一个词语查询字典id
                #if self.url_location[true_urlid].has_key(w):#若有这个词的location
                    #topicid = random.randint(0, self.K - 1)#topic标号是0到k-1
                    #dict_wordid = self.url_location[true_urlid][w]#-1
                    #self.z[dict_wordid][m] = topicid#将此文档中每个词赋值一个topic
                    #self.nw[dict_wordid][topicid] += 1
                    #self.nwsum[topicid] += 1
                    #self.nd[m][topicid]+=1
                #else:
                    #continue
        self.dbcommit()
        return 'prepare end!'

    # -------------------------------------------------------------------------------------------------------------------
    #计算alpha，输入nd, nd_sum
    def calc_theta(self):
        print 'calculating theta.....'
        nd=self.nd
        nd_sum=self.ndsum
        alpha=self.alpha
        topic_number=self.K
        doc_num=self.docs_count
        topic_alpha=topic_number * alpha  # K*alpha
        theta=[[0 for t in range(topic_number)]for d in range(doc_num)]
        for m in range(doc_num):
            for k in range(topic_number):
                theta[m][k]=(nd[m][k]+alpha)/(nd_sum[m]+topic_alpha)
        return theta
    #向量计算
    #def calc_theta(self):
        #print 'calculating theta.....'
        #for i in xrange(self.docs_count):#0到docs_count-1
            #self.theta[i] = (self.nd[i] + self.alpha) / (self.ndsum[i] + self.K * self.alpha)


    # -------------------------------------------------------------------------------------------------------------------
    #计算phi输入 nw,nw_sum
    #def calc_phi(self):
        #print 'calculating phi.....'
        #nw=self.nw
        #nw_sum=self.nwsum
        #beta=self.beta
        #topic_number = self.K
        #word_num = self.words_count
        #word_beta = word_num * beta  # V*beta
        #phi = [[0 for t in range(word_num)] for d in range(topic_number)]
        #for k in range(topic_number):
            #for w in range(word_num):
                #phi[k][w] = (nw[w][k] + beta) / (nw_sum[k] + word_beta)
        #return phi
    #向量计算
    def calc_phi(self):
        print 'calculating phi.....'
        nw = self.nw
        nw_sum = self.nwsum
        beta = self.beta
        topic_number = self.K
        word_num = self.words_count
        word_beta = word_num * beta  # V*beta
        phi = [[0 for t in range(word_num)] for d in range(topic_number)]
        for k in xrange(self.K):
            phi[k] = (nw.T[k] + beta) / (nw_sum[k] + word_beta)
        return phi

        #a[0]
        #Out[47]: array([1., 1., 1.])
        #In[48]: a[0] + 1
        #Out[48]: array([2., 2., 2.])
    # -------------------------------------------------------------------------------------------------------------------
    #吉比斯采样过程

    def sampling(self, m, dict_wordid,i):#对某个位置进行采样
        topic = self.z[m][i]
        self.nw[dict_wordid][topic] -= 1
        self.nd[m][topic] -= 1
        self.nwsum[topic] -= 1
        self.ndsum[m] -= 1

        Vbeta = self.words_count * self.beta
        Kalpha = self.K * self.alpha
        #吉比斯采样公式计算topic采样概率存入p中,以向量来计算  每迭代一次更新p向量
        self.p = (self.nw[dict_wordid] + self.beta) / (self.nwsum + Vbeta) * \
                 (self.nd[m] + self.alpha) / (self.ndsum[m] + Kalpha)

        # 在这里完成了一个掷骰子的过程，p[]向量是Topic的多项分布骰子，u是随便扔了个数，扔到p几就输出对应的Topic
        for k in xrange(1, self.K):
            self.p[k] += self.p[k - 1]#逐项累计 参考lda漫游p56  #p为长度为K的变量，存放各个主题的概率
        u = random.uniform(0, self.p[self.K - 1])
        new_topicid=0
        for i in xrange(self.K):
            if self.p[i] > u:
                new_topicid=i
                break
            else:
                continue
        self.nw[dict_wordid][new_topicid] += 1
        self.nwsum[new_topicid] += 1
        self.nd[m][new_topicid] += 1
        self.ndsum[m] += 1
        return new_topicid


    # -------------------------------------------------------------------------------------------------------------------
    #运行lda进行训练
    def runlda(self):
        self.data_preparing()
        # Consolelogger.info(u"迭代次数为%s 次" % self.iter_times)
        print 'prepare data end,starting simple iteration.....'
        for x in xrange(self.iter_times): #迭代次数
            print '迭代次数 ：'+str(x+1)
            for m in xrange(self.docs_count):#对每篇文档
                true_urlid=self.docs[m]
                #--------------------------------------------------------------------------------------
                res = len(self.url_location[true_urlid])
                #self.ndsum[m] = res  # 某文档有多少个词
                for i in xrange(res):
                    dict_wordid = self.url_location[true_urlid][i]
                    topic = self.sampling(m, dict_wordid,i)  # 对第m文档的w（位置location 从0开始）词进行采样 返回topicid
                    self.z[dict_wordid][m] = topic
                # --------------------------------------------------------------------------------------
                #for w in xrange(self.ndsum[m]):#对文档m中的每个词（包括重复词）都要进行采样，模拟投掷骰子生成词的逆过程
                    #dict_wordid=self.getword_dictid(self.docs[m],w)-1#减1同理
                    #if self.url_location[self.docs[m]].has_key(w):
                        #dict_wordid=self.url_location[self.docs[m]][w]#-1
                        #topic = self.sampling(m, dict_wordid)#对第m文档的w（位置location 从0开始）词进行采样 返回topicid
                        #self.z[dict_wordid][m] = topic
                    #else:
                        #continue
        self.theta=self.calc_theta()
        self.phi=self.calc_phi()
        print 'calculate end.....'
        print 'starting sign lda flag in mysql.....'
        for m in range(self.docs_count):
            self.updateandsave_topicword(m)
        self.save()
        return 'lda process end!'
 # -------------------------------------------------------------------------------------------------------------------
 #更新数据库ldaflag状态和高频主题词保存至本地
    def updateandsave_topicword(self,m):
        true_docid=self.docs[m]
        topk_word = 0
        top_topicid = self.theta[m].index(max(self.theta[m]))  # 概率最高的主题
        temp = list(enumerate(self.phi[top_topicid]))
        sorted_tmp=sorted(temp,key=lambda x: x[1], reverse=True) # 主题词排序 从高到底
        #temp.sort(key=lambda x: x[1], reverse=True)
        with codecs.open(self.topicwordfile, 'a') as f:
            f.write('真实文档号：'+str(true_docid) + '\n')#\n 换行
            for (wordid, p) in sorted_tmp:#phi的下标代表词典中的词id
                if wordid!=0:
                    self.con.execute('select word from wordlist where rowid=%d' % wordid)
                    word=self.con.fetchone()[0]
                    if topk_word < 10:
                        f.write('第' + str(topk_word + 1) + '主题词：' + str(word) + '\t' + '概率为：' + str(p) + '\n')  # \n 换行
                    #self.con.execute(
                    #'update wordlocation set lda_flag=1 where urlid=%d and wordid=%d ' % (true_docid, wordid))
                        topk_word += 1
                    else:
                        break
                else:
                    continue
        #------------------------------------------------------------------------------------
        #test
        topk_word = 0
        with codecs.open(self.test_word, 'a') as f:
            f.write('真实文档号：'+str(true_docid) + '\n')#\n 换行
            for (wordid, p) in sorted_tmp:#phi的下标代表词典中的词id
                if topk_word < 10:
                    f.write('第' + str(topk_word + 1) + '高频词：' + str(wordid) + '\t' + '概率为：' + str(p) + '\n')
                    topk_word += 1
                else:
                    break
        #------------------------------------------------------------------------------------
        topk_word = 0
        with codecs.open(self.test_topic, 'a') as f:
            f.write('真实文档号：'+str(true_docid) + '\n')#\n 换行
            if topk_word < 10:
                f.write('文档主题号'+str(top_topicid) + '\n')
                topk_word += 1
        self.dbcommit()
        return 'update and save done!'
 # -------------------------------------------------------------------------------------------------------------------
    # 保存theta文章-主题分布
    # 存储结果
    def save(self):
        print 'starting save.....'
        with codecs.open(self.thetafile, 'w') as f:
            for x in xrange(self.docs_count):
                for y in xrange(self.K):
                    if self.theta[x][y]!=None:
                        f.write(str(self.theta[x][y]) + '\t')
                    else:
                        f.write(str(0) + '\t')
                f.write('\n')
        # 保存phi词-主题分布
        #logger.info(u"词-主题分布已保存到%s" % self.phifile)
        with codecs.open(self.phifile, 'w') as f:
            for y in xrange(self.words_count):
                for x in xrange(self.K):
                    if self.phi[x][y]!=None:
                        f.write(str(self.phi[x][y]) + '\t')
                    else:
                        f.write(str(0) + '\t')
                f.write('\n')
        return 'save end'

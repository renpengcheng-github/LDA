#-*- coding:utf-8 -*-
import os
import jieba
import re
import json
import lda_model
import MySQLdb
import codecs
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

class lda:
    def __init__(self, dbname):
        self.conn = MySQLdb.connect(user='root', passwd='mysql', host='localhost', db=dbname, charset="utf8")
        self.con = self.conn.cursor()  # con  replace  cur
        self.stop_word = []

    def __del__(self):
        self.conn.close()

    def dbcommit(self):
        self.conn.commit()

    def readfile(self,path):
        fp = open(path, 'rb')
        content = fp.read()
        fp.close()
        return content

    def create_indextables(self):
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS urllist (rowid int(100) NOT NULL AUTO_INCREMENT PRIMARY KEY, url varchar(200),flag int(100),"
            "score_pagerank varchar(200),score_nn varchar(200),score_content varchar(200),score_lda varchar(200)) CHARACTER SET utf8")
        self.con.execute(
            'CREATE TABLE IF NOT EXISTS wordlist(rowid int(100) NOT NULL AUTO_INCREMENT PRIMARY KEY, word varchar(200)) CHARACTER SET utf8')
        self.con.execute(
            'CREATE TABLE IF NOT EXISTS wordlocation(urlid int(100) ,wordid int(100),location int(100),lda_flag int(100)) CHARACTER SET utf8')
        self.con.execute(
            'CREATE TABLE IF NOT EXISTS link(rowid int(100) NOT NULL AUTO_INCREMENT PRIMARY KEY,fromid int(100),toid int(100)) CHARACTER SET utf8')
        self.con.execute('create index wordidx on wordlist(word)')
        self.con.execute('create index urlidx on urllist(url)')
        self.con.execute('create index wordurlidx on wordlocation(wordid)')
        self.con.execute('create index urltoidx on link(toid)')
        self.con.execute('create index urlfromidx on link(fromid)')
        self.dbcommit()

    def separatewords(self,text):
        # print 'separating'
        null=[]
        a = re.findall(ur"[\u4e00-\u9fa5]+", text)
        if a:
            for word in a:#在切分词时候添加停用词检测
                if word in self.stop_word:
                    return null
                else:
                    return a

    def addtodb(self):
        #加载停用词，在入数据库之前忽略停用词，在所有进程开始前调用一次
        path_stopwoed = 'lda_logfile/'
        stopwordfile = path_stopwoed + 'stopwords.txt'
        with codecs.open(stopwordfile, 'rb', 'utf-8') as f:
            stop_words = f.readlines()
        for sword in stop_words:
            strword = sword.replace('\r\n', '').replace(' ','').strip()
            if strword != '':
                self.stop_word.append(strword)#全局变量  self.stop_word
            else:
                continue

        path = 'text_test1/'  # 'ldatext/'
        catlist = os.listdir(path)
        for dir in catlist:
            class_path = path + dir + '/'
            # if not os.path.exists(class_path):
            file_list = os.listdir(class_path)
            for file in file_list:
                fullname = class_path + file
                if fullname!='':
                    self.addtoindex(fullname)

                #content = self.readfile(fullname).strip()
                #content = content.replace("\r\n", "").strip()
                #content_seg = jieba.cut(content, cut_all=False)
                #for word in content_seg:
                    # w.append(separatewords(word))
                    #w.extend(self.separatewords(word))
                #print  json.dumps(w, encoding="UTF-8", ensure_ascii=False) + '\t'
                # print " ".join(content_seg)+'\t'
                #print fullname + '\t'

    def isindexed(self, url):
        # print 'checking'
        self.con.execute("select flag from urllist where url='%s' " % url)
        u = self.con.fetchone()
        if u!= None:
            if u[0] == 1:  # if only if flag=1
                return True
            else:
                return False
        else:
            return False

    def getentryid(self, table, field, value,flag=0, createnew=True):
        #update flag
        if flag == 1:
            self.con.execute("update %s set %s=1 where url='%s' " % (table, field, value))
            self.dbcommit()
        #search and insert
        elif flag == 0:
            self.con.execute("select rowid from %s where %s='%s'" % (table, field, value))
            res = self.con.fetchone()
            if res == None:
                self.con.execute("insert into %s (%s) values ('%s')" % (table, field, value))
                v = self.con.lastrowid
                self.dbcommit()
                return v
            else:
                # return int(res[0])
                return res[0]

    def addtoindex(self, url):
        words=[]
        if self.isindexed(url): return
        print 'Indexing ' + url
        content = self.readfile(url).strip()
        content = content.replace("\r\n", "").strip()
        content_seg = jieba.cut(content, cut_all=False)

        for word in content_seg:
            if word:#判断列表是否为空 若word非空
                words.extend(self.separatewords(word))
            else:
                continue

        # 得到这个 URL的 rowid
        urlid = self.getentryid('urllist', 'url', url)
        #flag=1 表示已经插入过这个url
        self.getentryid('urllist', 'flag', url, 1)

        # Link each word to this url
        for i in range(len(words)):
            word = words[i]
            if word.replace(' ','')=='' or word in self.stop_word: continue
            # 忽略停用词，但是location空出来了 是否合适  还是在加入words列表时就去除
            wordid = self.getentryid('wordlist', 'word', word)#得到词典中词的id（包括去重过程）
            self.con.execute("insert into wordlocation(urlid,wordid,location) values (%d,%d,%d)" % (urlid, wordid, i))
        self.dbcommit()



#-----------------------------------------------------------------------------------------------------------------------
#添加lda模块
    def run_lda(self):
        doc = {}
        i=0
        self.con.execute('select max(rowid) from wordlist')  # count(distinct(rowid))
        words_count = self.con.fetchone()[0]  # wordcount是词汇总数（不重复）
        self.con.execute('select DISTINCT (urlid) from wordlocation')#
        cur = self.con.fetchall()
        for (urlid,) in cur:
            #doc.append(urlid)#得到列表 doc
            doc[i]=urlid#得到字典doc  从0开始存储真正的docid
            i+=1
        docs_count = len(doc)
        a = {}
        url_location={}
        print 'preparing url_wordlocation.....'
        for docid in doc.values():  # 将docid,location,wordid载入内存，避免mysql瓶颈 返回self.url_location
            # self.url_location.setdefault(docid, {})
            self.con.execute('select wordid,location from wordlocation where urlid=%d' % docid)
            res = self.con.fetchall()
            j=0
            for (wordid, location) in res:
                a[j] = wordid
                j+=1

            url_location[docid] = a  # url_location存的是docs中的编号，即true_urlid
        mylda = lda_model.lda_model('lda', words_count+1, docs_count, doc,url_location)
        mylda.runlda()





#---------------------------------------------------------------------------------------------------------------------
#demo
    def set_stopwords(self):
        w=[]
        path='lda_logfile/'
        trainfile=path+'stopwords.txt'
        with codecs.open(trainfile, 'rb', 'utf-8') as f:
            words = f.readlines()
        for word in words:
            strword=word.replace('\r\n','').strip()
            if strword!='':
                w.append(strword)
            else:
                continue
            #print word.strip()
            #self.con.execute("select rowid from wordlist where word='%s' " % str(strword))
            #res=self.con.fetchone()
            #if res!=None:
                #print res[0]
                #self.con.execute("update wordlist set stopword_flag=1 where rowid=%d " % res[0])
                #self.dbcommit()
            #else:
                #continue
        #self.dbcommit()
        print json.dumps(w, encoding="UTF-8", ensure_ascii=False)+'\t'
        print ("和" in w)

# LDA
# lda 算法以及使用示例
# 1在数据库中建立表 ldasql 指定utf8
import cn_lda_text.py
content=cn_lda_text.lda('ldasql')
# 2建立表
content.create_indextables()
# 3读取训练集到数据库 文件在两个zip解压后直接使用  放在同一项目目录下
# 注意训练语料路径和停用词路径  在addtodb中设置
content.addtodb()
# 4运行lda_model
content.run_lda()

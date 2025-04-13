import jieba
import jieba.posseg as posseg

text = "贝壳省心租业务服务有保障"
seg = jieba.cut(text)
print(' '.join(seg))

seg2 = posseg.cut(text)
print([se for se in seg2])


print("--------")
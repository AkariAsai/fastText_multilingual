from fasttext import FastVector
import json

ja_dic = FastVector(vector_file='../vecmap/data/wiki.ja.vec')
en_dic = FastVector(vector_file='../vecmap/data/wiki.en.vec')
print("loaded the dictionaries")

ja_dic.apply_transform('alignment_matrices/ja.txt')
en_dic.apply_transform('alignment_matrices/en.txt')
print("transformed the dictionaries")

en_word_list = ["cat", "dog", "apple", "car",
                "train", "school", "student", "teacher"]
ja_word_list = ["猫", "犬", "りんご", "車", "電車", "学校", "生徒", "先生"]

result_f = open("multi_fast.txt", "w")
# Ja_word_list 10 nearest neighbor
for ja_word in ja_word_list:
    en_words = en_dic.translate_k_nearest_neighbour(ja_dic[ja_word], k=20)
    result[ja_word] = en_words
    resut_str = ",".join(result[ja_word])
    result_f.write(ja_word + "," + resut_str + "\n")


# En_word_list 10 nearest neighbor
for en_word in en_word_list:
    ja_words = ja_dic.translate_k_nearest_neighbour(en_dic[en_word], k=20)
    result[en_word] = ja_words
    resut_str = ",".join(result[en_word])
    result_f.write(en_word + "," + resut_str + "\n")

result_f.close()
#
# text = json.dump(result, open("result.json", "w"),
#                  ensure_ascii=False, indent=2)

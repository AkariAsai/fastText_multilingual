from fasttext import FastVector
import json

ja_dic = FastVector(vector_file='../vecmap/data/wiki.ja.vec')
en_dic = FastVector(vector_file='../vecmap/data/wiki.en.vec')
print("loaded the dictionaries")

ja_dic.apply_transform('alignment_matrices/ja.txt')
en_dic.apply_transform('alignment_matrices/ja.txt')
print("transformed the dictionaries")

en_word_list = ["cat", "dog", "apple", "car",
                "train", "school", "student", "teacher"]
ja_word_list = ["猫", "犬", "りんご", "車", "電車", "学校", "生徒", "先生"]


# Ja_word_list 10 nearest neighbor
result = {}
for ja_word in ja_word_list:
    en_words = en_dic.translate_k_nearest_neighbour(ja_dic[ja_word], k=10)
    result[ja_word] = en_words

# En_word_list 10 nearest neighbor
for en_word in en_word_list:
    ja_words = ja_dic.translate_k_nearest_neighbour(en_dic[en_word], k=10)
    result[en_word] = ja_words


text = json.dump(result, open("result.json", "w"),
                 ensure_ascii=False, indent=2)

# with open("result.json", "w") as fh:
# #     fh.write(text.encode("utf-8"))
#
#
# def main():
#
#
# if __name__ == "__main__":
#     main()

from fasttext import FastVector
import json
from tqdm import tqdm


def main():
    ja_dic = FastVector(vector_file='../vecmap/data/wiki.ja.vec')
    en_dic = FastVector(vector_file='../vecmap/data/wiki.en.vec')
    print("loaded the dictionaries")

    ja_dic.apply_transform('alignment_matrices/ja.txt')
    en_dic.apply_transform('alignment_matrices/en.txt')
    print("transformed the dictionaries")

    idx = 0
    result = {}
    result_f = open("en_ja_multifast.txt", "w")
    en_word_list = list(en_dic.word2id.keys())
    print("The total length of English pretrained vector : " + str(len(en_word_list)))

    for en_word in tqdm(en_word_list):
        ja_words = ja_dic.translate_k_nearest_neighbour(en_dic[en_word], k=15)
        result[en_word] = ja_words
        idx += 1
        result[en_word] = ja_words
        resut_str = ",".join(result[en_word])
        result_f.write(str(idx) + "," + en_word + "," + resut_str + "\n")
        if idx > 5000:
            exit()

    result_f.close()


if __name__ == "__main__":
    main()

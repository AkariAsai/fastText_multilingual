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
    for ja_word in tqdm(list(ja_dic.keys())):
        en_words = en_dic.translate_k_nearest_neighbour(ja_dic[ja_word], k=15)
        result[ja_word] = en_words

    text = json.dump(result, open("result.json", "w"),
                     ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

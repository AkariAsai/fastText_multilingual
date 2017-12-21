import numpy as np
import json

class VecMapVector:
    def __init__(self, vector_file='', transform=None):
        """Read in word vectors in fasttext format"""
        self.word2id = {}

        # Captures word order, for export() and translate methods
        self.id2word = []

        print('reading word vectors from %s' % vector_file)
        with open(vector_file, 'r') as f:
            (self.n_words, self.n_dim) = \
                (int(x) for x in f.readline().rstrip('\n').split(' '))
            self.embed = np.zeros((self.n_words, self.n_dim))
            for i, line in enumerate(f):
                elems = line.rstrip('\n').split(' ')
                self.word2id[elems[0]] = i
                self.embed[i] = elems[1:self.n_dim + 1]
                self.id2word.append(elems[0])

        # Used in translate_inverted_softmax()
        self.softmax_denominators = None

    def translate_nearest_neighbour(self, source_vector):
        """Obtain translation of source_vector using nearest neighbour retrieval"""
        similarity_vector = np.matmul(
            FastVector.normalised(self.embed), source_vector)
        target_id = np.argmax(similarity_vector)
        return self.id2word[target_id]

    def translate_k_nearest_neighbour(self, source_vector, k=10):
        """Obtain translation of source_vector using nearest neighbour retrieval"""
        similarity_vector = np.matmul(
            VecMapVector.normalised(self.embed), source_vector)
        target_ids = similarity_vector.argsort()[::-1][:k]

        word_list = []
        for target_id in target_ids:
            word_list.append(self.id2word[target_id])

        return word_list


    def translate_inverted_softmax(self, source_vector, source_space, nsamples,
                                   beta=10., batch_size=100, recalculate=True):
        """
        Obtain translation of source_vector using sampled inverted softmax retrieval
        with inverse temperature beta.

        nsamples vectors are drawn from source_space in batches of batch_size
        to calculate the inverted softmax denominators.
        Denominators from previous call are reused if recalculate=False. This saves
        time if multiple words are translated from the same source language.
        """
        embed_normalised = VecMapVector.normalised(self.embed)
        # calculate contributions to softmax denominators in batches
        # to save memory
        if self.softmax_denominators is None or recalculate is True:
            self.softmax_denominators = np.zeros(self.embed.shape[0])
            while nsamples > 0:
                # get batch of randomly sampled vectors from source space
                sample_vectors = source_space.get_samples(
                    min(nsamples, batch_size))
                # calculate cosine similarities between sampled vectors and
                # all vectors in the target space
                sample_similarities = \
                    np.matmul(embed_normalised,
                              VecMapVector.normalised(sample_vectors).transpose())
                # accumulate contribution to denominators
                self.softmax_denominators \
                    += np.sum(np.exp(beta * sample_similarities), axis=1)
                nsamples -= batch_size
        # cosine similarities between source_vector and all target vectors
        similarity_vector = np.matmul(embed_normalised,
                                      source_vector / np.linalg.norm(source_vector))
        # exponentiate and normalise with denominators to obtain inverted
        # softmax
        softmax_scores = np.exp(beta * similarity_vector) / \
            self.softmax_denominators
        # pick highest score as translation
        target_id = np.argmax(softmax_scores)
        return self.id2word[target_id]

    def get_samples(self, nsamples):
        """Return a matrix of nsamples randomly sampled vectors from embed"""
        sample_ids = np.random.choice(
            self.embed.shape[0], nsamples, replace=False)
        return self.embed[sample_ids]

    @classmethod
    def normalised(cls, mat, axis=-1, order=2):
        """Utility function to normalise the rows of a numpy array."""
        norm = np.linalg.norm(
            mat, axis=axis, ord=order, keepdims=True)
        norm[norm == 0] = 1
        return mat / norm

    @classmethod
    def cosine_similarity(cls, vec_a, vec_b):
        """Compute cosine similarity between vec_a and vec_b"""
        return np.dot(vec_a, vec_b) / \
            (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

    def __contains__(self, key):
        return key in self.word2id

    def __getitem__(self, key):
        return self.embed[self.word2id[key]]

def main():
    ja_dic = VecMapVector(vector_file='/home/dl-exp/vecmap/data/wiki.ja.mapped.txt')
    en_dic = VecMapVector(vector_file='/home/dl-exp/vecmap/data/wiki.en.mapped.txt')
    print("loaded the dictionaries")

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
        if idx > 500:
            break

    result_f.close()

    idx = 0
    result = {}
    result_f = open("ja_en_multifast.txt", "w")
    ja_word_list = list(ja_dic.word2id.keys())

    for ja_word in tqdm(ja_word_list):
        en_words = en_dic.translate_k_nearest_neighbour(ja_dic[ja_word], k=15)
        result[en_word] = ja_words
        idx += 1
        result[ja_word] = en_words
        resut_str = ",".join(result[ja_word])
        result_f.write(str(idx) + "," + en_word + "," + resut_str + "\n")
        if idx > 500:
            break

    result_f.close()

if __name__ == "__main__":
    main()

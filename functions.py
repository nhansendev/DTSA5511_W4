import numpy as np
import torch


def pad_encode_tensor(
    sentences, vector_len=96, nlp=None, word_encoder=None, to_cuda=True
):
    # When creating a tensor out of sentences of different lengths we must
    # pad each sentence to match the length of the longest sentence.
    # After passing the sentences through a RNN we take the output at
    # the index that matches the original sentence length.

    tmp_sentences = []
    sen_lens = []
    # Encode sentences while tracking their lengths
    for sentence in sentences:
        if nlp is not None:
            tmp = [word.vector for word in nlp(" ".join(sentence))]
        elif word_encoder is not None:
            tmp = []
            for w in sentence:
                try:
                    tmp.append(word_encoder.get_vector(w, norm=True))
                except KeyError:
                    # Word not known to encoder
                    tmp.append(np.zeros(vector_len))
        else:
            raise ValueError("Must provide either nlp or word_encoder.")

        sen_lens.append(len(tmp))
        tmp_sentences.append(tmp)

    maxlen = max(sen_lens)

    # Pad all sentences to the same length
    padding_vector = np.zeros(vector_len)
    for sen in tmp_sentences:
        for _ in range(maxlen - len(sen)):
            sen.append(padding_vector)

    # Shape: batch, sequence length, features
    padded = torch.FloatTensor(np.array(tmp_sentences))

    if to_cuda:
        padded = padded.cuda()

    return padded, np.array([i - 1 for i in sen_lens], "int")


def encode_KL(dataset, vector_len=96, nlp=None, word_encoder=None, to_cuda=True):
    # Encode keywords and locations, replacing words not known to the encoder with zero vectors
    encoded = []
    for i in range(len(dataset)):
        if nlp is not None:
            A = nlp(dataset.at[i, "keyword"])[0].vector
            B = nlp(dataset.at[i, "location"])[0].vector
        elif word_encoder is not None:
            try:
                A = word_encoder.get_vector(dataset.at[i, "keyword"], norm=True)
            except KeyError:
                A = np.zeros(vector_len)
            try:
                B = word_encoder.get_vector(dataset.at[i, "location"], norm=True)
            except KeyError:
                B = np.zeros(vector_len)
        else:
            raise ValueError("Must provide either nlp or word_encoder.")
        encoded.append(np.append(A, B))

    if to_cuda:
        return torch.FloatTensor(np.array(encoded)).cuda()
    else:
        return torch.FloatTensor(np.array(encoded))

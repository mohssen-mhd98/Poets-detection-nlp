import os
import math


def main():
    poets_probs = train_poets_models()
    hemistich, poets_name = read_test_data('test_set\\test_file.txt')
    epsilon_values = [0.0001, 0.0000001]
    lambda_values = [(0.001, 0.233, 0.766), (0.0001, 0.1999, 0.8), (0, 0.5, 0.5), (0, 0.001, 0.999)]
    for eps in epsilon_values:
        for l in lambda_values:
            print("Precision of model with eps = {},"
                  " and lambdas = (λ1={}, λ2={}, λ3={}) is:".format(eps, l[0], l[1], l[2]))
            precision = test_model(poets_probs, hemistich, poets_name, l1=l[0],
                                   l2=l[1], l3=l[2], eps=eps)
            print('BackOff Accracy:', precision, "\n")


# Read the data from text file from its path address.
def read_data(addr):
    f = open(addr, "r", encoding='utf-8')
    return f


# Returns two correspond lists,
# that each element in poets_name is correspond to each hemistich in hemistiches list.
def read_test_data(addr):
    data = read_data(addr)
    file_lines = data.readlines()
    y_labels = ['ferdowsi', 'hafez', 'molavi']
    hemistiches = []
    poets_name = []
    for l in file_lines:
        data_part = l.split('\t')
        hemistiches.append(data_part[1].replace('\n', ''))
        poets_name.append(y_labels[int(data_part[0]) - 1])
    return hemistiches, poets_name


# Returns a dictionary that contains number of repetition of each word.
def cal_words_vocab(lines):
    vocab = {}
    for l in lines:
        words = l.split(' ')
        for w in words:
            p_word = w.replace('\n', '')
            if p_word not in vocab.keys():
                vocab[p_word] = 1
            else:
                vocab[p_word] += 1
    return vocab


# Returns vocabulary, without words are repeated in train set file.
def non_repetitive_words_removal(vocab, min_count=2):
    repetitive_words = {key: val for key, val in vocab.items() if val > min_count}
    return repetitive_words


# Returns the number of two unit repetition in lines of a specific train file.
# (e.g repetition a|b, that a, b are units)
def bigram_counter(lines):
    _2units_count = {}
    for l in lines:
        modified = '<s> ' + l.replace('\n', '') + ' <\s>'
        words = modified.split()
        num_of_line_words = len(words)
        for w in range(1, num_of_line_words):
            prob_name = words[w] + '|' + words[w - 1]
            if prob_name not in _2units_count.keys():
                _2units_count[prob_name] = 1
            else:
                _2units_count[prob_name] += 1
    return _2units_count


# Returns the probability's of Bigram model
def calculate_bigram_probs(_2units_count, num_of_rep_of_words, number_of_lines):
    bigram_probs = {}
    for key in list(_2units_count.keys()):
        previous_word = key.split('|')[1]
        if previous_word == '<s>':
            bigram_probs[key] = math.log2(_2units_count[key] / number_of_lines)
        else:
            bigram_probs[key] = math.log2(_2units_count[key] / num_of_rep_of_words[previous_word])
    return bigram_probs


# This method calculates the Unigram and Bigram models for a poet from its train data file.
# And returns a model-dictionary contains Unigran and Bigram probability dictionaries.
def make_model(data):
    vocabulary = cal_words_vocab(data)
    # Words that repeated more than two times in file.
    valid_words_invocab = non_repetitive_words_removal(vocabulary)
    word_count = len(valid_words_invocab)
    num_of_words = sum(valid_words_invocab.values())  # Number of all words exists in this specific txt file.
    _2units_count = bigram_counter(data)
    bigram_probs = calculate_bigram_probs(_2units_count, vocabulary, len(data))
    poet_model = {
                  'Unigram': {key: math.log2(val / num_of_words) for key, val in valid_words_invocab.items()},
                  'Bigram': bigram_probs}

    return poet_model, word_count


# Returns the final poets models.
def train_poets_models():
    model = {}
    total_number_of_words = 0
    poet_words_count = {}
    for file in os.listdir("train_set"):
        if file.endswith(".txt"):
            file_addr = os.path.join("train_set", file)
            poet = file.split('_')[0]
            data = read_data(file_addr).readlines()
            # Num of valid words in current file.
            poet_model, word_count = make_model(data)
            model[poet] = poet_model
            poet_words_count[poet] = word_count
            total_number_of_words += word_count

    model['PoetProb'] = {key: math.log2(val / total_number_of_words) for key, val in poet_words_count.items()}
    return model


# Calculate The Unigram Probability Of One Class For One Line In The Test DataSet
# Returns the Unigram probability of a hemistich from test file.
def unigram_prob_calculator(hemistich, unigram_model):
    words = hemistich.split()
    unigram_prob = 1
    for w in words:
        if w in unigram_model.keys():
            unigram_prob *= 2 ** unigram_model[w]
        else:
            unigram_prob *= 0.00001

    return unigram_prob


# Calculate The Bigram Probability Of One Class For One Line In The Test DataSet
def backoff_prob_calculator(hemistich, model, l1, l2, l3, eps):
    modified = '<s> ' + hemistich.replace('\n', '') + ' <\s>'
    words = modified.split()
    num_of_line_words = len(words)
    backoff_prob = 1
    if1 = 0
    if2 = 0
    if3 = 0
    else1 = 0
    for i in range(1, num_of_line_words):
        unigram_prob = 0.0001
        bigram_prob = 0.0001
        prob_name = words[i] + '|' + words[i - 1]
        if (prob_name in model["Bigram"].keys()) and (words[i] not in model["Unigram"].keys()):
            bigram_prob = 2 ** model["Bigram"][prob_name]
            backoff_prob *= l3 * bigram_prob + l2 * unigram_prob + l1 * eps

        elif (prob_name not in model["Bigram"].keys()) and (words[i] in model["Unigram"].keys()):
            unigram_prob = 2 ** model["Unigram"][words[i]]
            backoff_prob *= l3 * bigram_prob + l2 * unigram_prob + l1 * eps

        elif (prob_name in model["Bigram"].keys()) and (words[i] in model["Unigram"].keys()):
            bigram_prob = 2 ** model["Bigram"][prob_name]
            unigram_prob = 2 ** model["Unigram"][words[i]]
            backoff_prob *= l3 * bigram_prob + l2 * unigram_prob + l1 * eps

        else:
            bigram_prob = 0.00001
            unigram_prob = 0.00001
            backoff_prob *= l3 * bigram_prob + l2 * unigram_prob + l1 * eps

    return backoff_prob


# Returns the precision of model.
def test_model(model, hemistiches, poets_name, l1=0.001, l2=0.299, l3=0.7, eps=0.000001):
    num_of_data = len(poets_name)  # Num of hemistiches.
    u_true_predicted = 0
    b_true_predicted = 0
    for i in range(num_of_data):
        max_prob_backoff = 0
        max_prob_unigram = 0
        predicted_hemistich_unigram = ''
        predicted_hemistich_backoff = ''
        for key, probs_dict in model.items():
            if key != 'PoetProb':
                poet_prob = 2 ** model['PoetProb'][key]
                unigram_prob = poet_prob * unigram_prob_calculator(hemistiches[i], probs_dict['Unigram'])
                b = backoff_prob_calculator(hemistiches[i], probs_dict, l1, l2, l3, eps)
                hemistich_prob = poet_prob * b
                # print(x[i], iflist, y[i])

                if unigram_prob > max_prob_unigram:
                    max_prob_unigram = unigram_prob
                    predicted_hemistich_unigram = key

                if max_prob_backoff < hemistich_prob:
                    max_prob_backoff = hemistich_prob
                    predicted_hemistich_backoff = key

        # print(x[i], y[i], predicted_y_backoff)
        if predicted_hemistich_backoff == poets_name[i]:
            u_true_predicted += 1
        if predicted_hemistich_backoff == poets_name[i]:
            b_true_predicted += 1

    unigram_precision = u_true_predicted / num_of_data
    backoff_precision = b_true_predicted / num_of_data

    return backoff_precision


if __name__ == '__main__':
    main()

import time

start = time.time()
import nltk
import numpy as np
import random
import math
from collections import Counter

# takes a sent and returns several paraphrases of it
def paraphrase(sent):
    # what descriptions are outstanding -> show me what descriptions are outstanding
    out = org_ngram_noise_scaling([sent])

    end = time.time()
    duration = end - start
    duration = round(duration, 4)
    print(str(duration) + 's')

    print("Original: ", sent, "\nParaphrase: ", out)
    return out


def org_ngram_noise_scaling(sentlist):
    org_corpus_word_counter, org_bigram_counter, org_trigram_counter, org_fourgram_counter, \
    bigram_mult, trigram_mult, fourgram_mult = determine_noiseboost_multi_orig_corpus(sentlist)
    bigram_sets = set(list(bigram_mult.keys()))
    trigram_sets = set(list(trigram_mult.keys()))
    fourgram_sets = set(list(fourgram_mult.keys()))

    boosted_para_list = noise_boost_sent_list(bigram_mult, trigram_mult, fourgram_mult, bigram_sets, trigram_sets,
                                              fourgram_sets, sentlist)

    return boosted_para_list


def noise_boost_sent_list(bigram_mult, trigram_mult, fourgram_mult, bigram_sets, trigram_sets, fourgram_sets, sentlist):
    org_eng = [sentlist[0]]
    #sentlist = sentlist[0]
    ngrams = list(map(lambda x: get_ngram_sent(x), sentlist))

    ngram_mults = list(
        map(lambda x: (list(set(x[0]).intersection(bigram_sets)), list(set(x[1]).intersection(trigram_sets)),
                       list(set(x[2]).intersection(fourgram_sets))), ngrams))

    ngram_mults = list(map(lambda x: ({k: v for k, v in bigram_mult.items() if k in x[0]},
                                      {k: v for k, v in trigram_mult.items() if k in x[1]},
                                      {k: v for k, v in fourgram_mult.items() if k in x[2]}), ngram_mults))

    ngram_mults = [((list(x[0].values()) + list(x[1].values()) + list(x[2].values()))) for x in ngram_mults]
    ngram_mults = [det_multiplier_logic(x, sentlist[i]) for i, x in enumerate(ngram_mults)]
    eliminations = [x for x in ngram_mults if x == "eliminate"]
    sentlist = list(zip(sentlist, ngram_mults, org_eng))
    sentlist = list(map(lambda x: noise_boost_sents(x), sentlist))
    sentlist = [item for sublist in sentlist for item in sublist]
    return sentlist


def noise_boost_sents(input):
    sentence = input[0].strip()
    multiplier = input[1]
    origial = input[2].strip()

    if multiplier == "eliminate":
        multiplier = 2
    elif multiplier < 0:
        out = [sentence]
        return out
    out = multiply_sentence(sentence, multiplier)
    if sentence in out:
        out.remove(sentence)
    out = list(set([sentence] + out))
    return out


def multiply_sentence(sentence, multiplier):
    multiplier = int(multiplier)
    sentence = sentence.strip()
    if multiplier == 0:
        return [sentence]
    out = []
    # split_sent= sentence.split()

    all_QWORDS_phrases = ['what', 'return', 'give', 'show all', 'show'
                          'print all', 'find all', 'list all', 'break down of all',
                          'all', 'find', 'return all', 'what is the', 'print', 'give me',
                          'what are the', 'give all',
                          'find me', 'list all of']

    # all_QWORDS_phrases=" ".join(all_QWORDS_phrases)

    for i in range(multiplier):
        hasalready = False
        for x in all_QWORDS_phrases:
            if " " + x + " " in " " + sentence + " ":
                hasalready = True
                out += replace_remove_list_phrases(sentence, x)
                break

        if not hasalready:
            tmp = random.choice(all_QWORDS_phrases) + " " + sentence
            out.append(tmp)

    return out


def replace_remove_list_phrases(sent, phr):
    showme_replacements_list = {}
    showme_replacements_list["list_phrases1"] = ["list", "show", "all"]

    showme_replacements_list["count"] = ["count", "number of"]
    showme_replacements_list["mean"] = ["average", "mean", "avg"]

    showme_replacements_list["aggregate_phrases"] = ["total", "sum"]

    sent = " " + sent + " "
    out = []
    # if phr in replaceable_QWORDS:
    tmp = sent.replace(" " + phr + " ", " ")
    out.append(tmp)
    for vllist in showme_replacements_list.values():
        if phr in vllist:
            ch = random.choice(vllist)
            tmp = sent.replace(" " + phr + " ", " " + ch + " ")
            out.append(tmp)
    out = [" ".join(x.split()) for x in out]
    return out


def special_sentence_condition_boosting(sentence):
    split = sentence.split()
    split = [(x, split[i + 1]) for i, x in enumerate(split) if
             (i < len(split) - 1 and is_duckling_vl(x) and is_duckling_vl(split[i + 1]))]
    if len(split) > 0:
        return 1
    return 0


def det_multiplier_logic(multipliers, sentence):
    if len(multipliers) == 0:
        mult = special_sentence_condition_boosting(sentence)
        if mult == 0:
            mult = np.random.choice([1, -1], 1, p=[0.3, 0.7]).item()
        return mult
    zeros = [x for x in multipliers if x == 0]
    if len(zeros) > 0:
        if len(zeros) >= 2:
            return "eliminate"
        else:
            return max(multipliers)
    mult = max(multipliers)
    if mult <= 0:
        return "eliminate"
    mult = round(mult + special_sentence_condition_boosting(sentence))
    return mult


def determine_noiseboost_multi_orig_corpus(sentencelist):
    corpus_word_counter = Counter((" ".join(sentencelist)).split())
    bigram_counter, trigram_counter, fourgram_counter = get_ngrams_counter(sentencelist)
    k = list(fourgram_counter.keys())
    v = list(fourgram_counter.values())
    mean = np.mean(v)
    std = np.std(v)
    dev1 = [i for i, x in enumerate(v) if x <= math.ceil((mean - 2 * std))]
    dev2 = [i for i, x in enumerate(v) if x <= math.ceil((mean - std)) if i not in dev1]
    fourgram_mult = {}
    fourgram_mult.update({k[i]: 2 for i in dev1})
    fourgram_mult.update({k[i]: 1 for i in dev2})
    k = list(trigram_counter.keys())
    v = list(trigram_counter.values())
    mean = np.mean(v)
    std = np.std(v)
    dev1 = [i for i, x in enumerate(v) if x <= math.ceil((mean - 2 * std))]
    dev2 = [i for i, x in enumerate(v) if x <= math.ceil((mean - std)) if i not in dev1]
    trigram_mult = {}
    trigram_mult.update({k[i]: 2 for i in dev1})
    trigram_mult.update({k[i]: 1 for i in dev2})
    k = list(bigram_counter.keys())
    v = list(bigram_counter.values())
    mean = np.mean(v)
    std = np.std(v)
    dev1 = [i for i, x in enumerate(v) if x <= math.ceil((mean - 2 * std))]
    dev2 = [i for i, x in enumerate(v) if x <= math.ceil((mean - std)) if i not in dev1]
    bigram_mult = {}
    bigram_mult.update({k[i]: 3 for i in dev1})
    bigram_mult.update({k[i]: 2 for i in dev2})
    return corpus_word_counter, bigram_counter, trigram_counter, fourgram_counter, bigram_mult, trigram_mult, fourgram_mult


def get_ngrams_counter(sentencelist):
    sentencelist = " <EOD> ".join(sentencelist)
    ngrams = get_ngram_sent(sentencelist)
    bigram_counter = ngrams[0]
    bigram_counter = [x for x in bigram_counter if "<EOD>" not in x]
    bigram_counter = Counter(bigram_counter)

    trigram_counter = ngrams[1]
    trigram_counter = [x for x in trigram_counter if "<EOD>" not in x]
    trigram_counter = Counter(trigram_counter)

    fourgram_counter = ngrams[2]
    fourgram_counter = [x for x in fourgram_counter if "<EOD>" not in x]
    fourgram_counter = Counter(fourgram_counter)
    return bigram_counter, trigram_counter, fourgram_counter


def get_ngram_sent(x):
    bigrams, trigrams, fourgrams = get_ngram_sent_nofilter(x)
    return bigrams, trigrams, fourgrams


def is_duckling_vl(inp):
    if "value_label" in inp:
        return True
    if "duckling" in inp:
        return True
    return False


def get_ngram_sent_nofilter(x):
    x = x.split()
    bigrams = nltk.ngrams(x, 2)
    trigrams = nltk.ngrams(x, 3)
    fourgrams = nltk.ngrams(x, 4)
    bigrams = list(map(lambda x: " ".join(x), bigrams))
    trigrams = list(map(lambda x: " ".join(x), trigrams))
    fourgrams = list(map(lambda x: " ".join(x), fourgrams))
    return bigrams, trigrams, fourgrams


if __name__ == '__main__':
    #paraphrase("what invoices are outstanding")
    #paraphrase("highest outstanding specifications by month")
    paraphrase("profits by month")
# Based on Isabelle Dautriche's code: https://github.com/SbllDtrch/NullLexicons/blob/master/extract_lex_celex.py

import os
import re
import argparse
import pandas as pd


VOWELS = set("IE{VQU@i#$u312456789cq0~iI1!eE2@aAoO3#4$6^uU7&5+8*9(0)<>[]{}")


# Word cleaning functions
def clean_words(corpus):
    for x in corpus:
        x['phones'], x['syllables'] = clean_word(x['phones'])


def clean_word(word_in):
    """Remove stress and syllable boundaries."""
    word_in = re.sub("'", "", word_in)
    syllables = re.sub('"', "", word_in)
    word = re.sub("-", "", syllables)
    return word, syllables


def substitute_diphthongs(corpus, language):
    new_corpus = []
    for x in corpus:
        phones = x['phones']
        if check_valid_phones(phones, language):
            x['phones'] = celex_diphthong_sub(phones)
            x['syllables'] = celex_diphthong_sub(x['syllables'])
            new_corpus += [x]
    return new_corpus


def check_valid_phones(word, language):
    filtered_chars = set(['c', 'q', '0', '~', '^', '*', '<'])
    contains_invalid = filtered_chars & set(word)

    if not contains_invalid and \
            ((language == 'english') | ("_" not in word)):
        return True
    return False


def celex_diphthong_sub(word):
    """ Do celex dipthong subs. """
    word = re.sub("2", "#I", word)
    word = re.sub("4", "QI", word)
    word = re.sub("6", "#U", word)
    word = re.sub("7", "I@", word)
    word = re.sub("8", "E@", word)
    word = re.sub("9", "U@", word)
    word = re.sub("X", "Oy", word)
    word = re.sub("W", "ai", word)
    word = re.sub("B", "au", word)
    word = re.sub("K", "EI", word)
    return word


# Celex reading functions
def get_cv(word):
    cv_info = ""
    for letter in word:
        if letter in VOWELS:
            cv_info += "V"
        else:
            cv_info += "C"

    return cv_info


def get_celex_path(path, language, dtype, lemma):
    """ Return celex path, given root, lemma, language. """
    return os.path.join(
        path, language,
        "{lang}{dtype}{lem}/{lang}{dtype}{lem}.cd"
        .format(lang=language[0], dtype=dtype[0], lem=lemma[0]))


def read_in_celex_lines(path):
    """ Return lines in given celex file. """
    return [line.strip().split("\\") for line in open(path, "r").readlines()]


def get_celex_monos(path_base, language):
    """ Return list of celex ids for monomorphemes. """
    path = get_celex_path(path_base, language, 'morphology', 'lemma')
    lines = read_in_celex_lines(path)

    # 0 is ID and 3 is if it is monomorphemic
    return {x[0] for x in lines if x[3] == "M"}


def get_celex_class(path_base, language):
    """ Return list of tuples of (celex ids; class) """
    path = get_celex_path(path_base, language, 'morphology', 'lemma')
    lines = read_in_celex_lines(path)

    # 0 is ID and 21/13/12 is pos tag  information
    if language == "english":
        return dict((x[0], re.sub("]", "", x[21].split("[")[-1])) for x in lines)
    if language == "german":
        return dict((x[0], re.sub("]", "", x[13].split("[")[-1])) for x in lines)
    return dict((x[0], re.sub("]", "", x[12].split("[")[-1])) for x in lines)


def get_celex_freq(path_base, language):
    """ Return tuples of celex ids and freq per million word. """
    path = get_celex_path(path_base, language, 'frequency', 'lemma')
    lines = read_in_celex_lines(path)

    # 0 is ID and 2 is frequency in the corpus
    return dict((x[0], x[2]) for x in lines)


def celex_pron_loc(language, lemma):
    """ Return location of pronunciation in celex, given language. """
    locations = {
        'english': 5,
        'german': 3,
        'dutch': 3,
    }
    pron = locations[language]

    if lemma == "wordform":
        pron += 1
    return pron


def get_main_celex_corpus(path_base, language, lemma):
    path = get_celex_path(path_base, language, 'phonology', lemma)
    print(path)
    return read_in_celex_lines(path)


def filter_monos(lines, path, language):
    monos = get_celex_monos(path, language)
    return [line for line in lines if line[0] in monos]


def extract_celex_info(line, freqs, gram, language="english", lemma="lemma"):
    """ Return celex word (ortho or phonemic) and its freq from celex line. """
    word = line[1]

    filtered_chars = set(['-', '.', '\'', ' '])
    contains_invalid = filtered_chars & set(word)

    if word.isalpha() and ((word.islower()) | (language == 'german')) and \
            not contains_invalid:
        idx = line[0]
        return {
            'phones': line[celex_pron_loc(language, lemma)],
            'freq': int(freqs[idx]),
            'word': word,
            'pos': gram[idx],
        }
    return None


def extract_corpus(lines, freqs, gram, language, lemma):
    corpus = [extract_celex_info(line, freqs, gram, language, lemma) for line in lines]
    return [x for x in corpus if x is not None]


def build_celex_corpus(path, language, lemma, mono):
    """ Return corpus from celex, given path and parameters. """
    lines = get_main_celex_corpus(path, language, lemma)
    if mono:
        lines = filter_monos(lines, path, language)

    freqs = get_celex_freq(path, language)
    gram = get_celex_class(path, language)

    return extract_corpus(lines, freqs, gram, language, lemma)


def remove_zero_frequency(corpus):
    return [c for c in corpus if c['freq'] > 0]


def filter_length(corpus, minlength, maxlength):
    return [x for x in corpus
            if (len(re.sub("-", "", x['phones'])) >= minlength and
                len(re.sub("-", "", x['phones'])) <= maxlength)]


def clean_pronunciation(corpus, language, minlength, maxlength):
    # reduce celex to just pronunciation
    clean_words(corpus)

    corpus = substitute_diphthongs(corpus, language)
    corpus = filter_length(corpus, minlength, maxlength)
    return corpus


def filter_homophony(df):
    phones_freq_agg = df.groupby('phones').agg('sum')['freq'].to_dict()
    phones_count = df.groupby('phones').agg('count')['freq'].to_dict()

    df.sort_values('freq', ascending=False, inplace=True)
    df.drop_duplicates('phones', keep='first', inplace=True)

    df['freq'] = df.phones.apply(lambda x: phones_freq_agg[x])
    df['homo'] = df.phones.apply(lambda x: phones_count[x])
    return df


def build_real_lex(path, lemma, language, mono, homophily, minlength, maxlength):
    corpus = build_celex_corpus(path, language, lemma, mono)
    print("number of words:", len(corpus))

    corpus = remove_zero_frequency(corpus)
    print("number of words in lex after selecting words frequency > 0:", len(corpus))

    corpus = clean_pronunciation(corpus, language, minlength, maxlength)
    print("number of words in lex after cleaning pronunciation: %d" %
          len(corpus))

    df = pd.DataFrame(corpus)

    if not homophily:
        df = filter_homophony(df)

    print(">>>TOTAL NB OF WORDS", df.shape[0])
    return df


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-path', type=str,
                        help='The path with raw celex data')
    parser.add_argument('--tgt-path', type=str,
                        help='The tgt path where to save data')

    parser.add_argument('--lemma', type=str, nargs='?',
                        help='lemma or wordform in celex', default="lemma")
    parser.add_argument('--language', type=str,
                        default="eng")

    parser.add_argument('--mono', action='store_true', default=True,
                        help='Get only monomorphemic words (this is the default)')
    parser.add_argument('--not-mono', action='store_false', dest='mono',
                        help='Get non monomorphemic words')

    parser.add_argument('--homophily', action='store_true', default=False,
                        help='Flag to allow homophones in celex')
    parser.add_argument('--minlength', type=int, default=0,
                        help='minimum length of word allowed from celex')
    parser.add_argument('--maxlength', type=int, default=float('inf'),
                        help='maximum length of word allowed from celex')

    return parser.parse_args()


def main():
    args = get_args()

    language_dict = {
        'en': 'english',
        'eng': 'english',
        'de': 'german',
        'deu': 'german',
        'nl': 'dutch',
        'nld': 'dutch',
    }
    language = language_dict[args.language]

    celex_path = os.path.join(args.src_path, 'CELEX_V2')

    celex_list = [
        args.language, args.lemma, args.mono,
        args.homophily, args.minlength, args.maxlength]
    fname = "_".join([str(i) for i in celex_list]) + ".tsv"
    fpath = os.path.join(args.tgt_path, fname)

    df = build_real_lex(
        celex_path, args.lemma, language, args.mono, args.homophily,
        args.minlength, args.maxlength)

    df.to_csv(fpath, sep='\t')


if __name__ == "__main__":
    main()

# This is a script to calculate sentiment
# score (Positive, Negative) for a dataset
# using ArSenL_v1.0A

import pandas as pd
import argparse
import io
import pyaramorph
from collections import namedtuple
import util

Segment = namedtuple('Segment', 'prefix stem suffix')

def word2buck(word):
	uni2buck = {u'\u0621': "'",
	 u'\u0622': '|',
	 u'\u0623': '>',
	 u'\u0624': '&',
	 u'\u0625': '<',
	 u'\u0626': '}',
	 u'\u0627': 'A',
	 u'\u0628': 'b',
	 u'\u0629': 'p',
	 u'\u062a': 't',
	 u'\u062b': 'v',
	 u'\u062c': 'j',
	 u'\u062d': 'H',
	 u'\u062e': 'x',
	 u'\u062f': 'd',
	 u'\u0630': '*',
	 u'\u0631': 'r',
	 u'\u0632': 'z',
	 u'\u0633': 's',
	 u'\u0634': '$',
	 u'\u0635': 'S',
	 u'\u0636': 'D',
	 u'\u0637': 'T',
	 u'\u0638': 'Z',
	 u'\u0639': 'E',
	 u'\u063a': 'g',
	 u'\u0640': '_',
	 u'\u0641': 'f',
	 u'\u0642': 'q',
	 u'\u0643': 'k',
	 u'\u0644': 'l',
	 u'\u0645': 'm',
	 u'\u0646': 'n',
	 u'\u0647': 'h',
	 u'\u0648': 'w',
	 u'\u0649': 'Y',
	 u'\u064a': 'y',
	 u'\u064b': 'F',
	 u'\u064c': 'N',
	 u'\u064d': 'K',
	 u'\u064e': 'a',
	 u'\u064f': 'u',
	 u'\u0650': 'i',
	 u'\u0651': '~',
	 u'\u0652': 'o',
	 u'\u0670': '`'
	}
	for k, v in uni2buck.items():
		word = word.replace(k, v)
	return word

def clean(word):
	word = word.split("_")[0]
	return word

def computeSentence(df, line, analyzer):
	"""Compute sentiment score for a sentence."""
	# DF is the lexicon
	# Line is the input
	# Analyzer is the pyaramorph analyzer
	words = line.split(" ")
	words = [word2buck(word) for word in words]
	extra_segments = []
	for word in words:
		extra_segments.extend(analyze_word(analyzer, word))
	words.extend(extra_segments)
	pos_score = 0
	neg_score = 0
	for word in words:
		if word in df.index:
			pos_score += df.loc[word2buck(word)][0]
			neg_score += df.loc[word2buck(word)][1]
	pos_score_mean = pos_score / float(len(words))
	neg_score_mean = neg_score / float(len(words))
	return (pos_score_mean, neg_score_mean)

def computeDataset(df, data, analyzer):
	"""Compute sentiment score for the whole data."""
	output = []
	for item in data:
		output.append(computeSentence(df, item, analyzer))
	return output

def _check_segment(self, prefix, stem, suffix):
    """See if the prefix, stem, and suffix are compatible."""
    analyses = []

    # Loop through the possible prefix entries
    for pre_entry in self.prefixes[prefix]:
        (voc_a, cat_a, gloss_a, pos_a) = pre_entry[1:5]

        # Loop through the possible stem entries
        for stem_entry in self.stems[stem]:
            (voc_b, cat_b, gloss_b, pos_b, lemmaID) = stem_entry[1:]

            # Check the prefix + stem pair
            pairAB = "%s %s" % (cat_a, cat_b)
            if not pairAB in self.tableAB: continue

            # Loop through the possible suffix entries
            for suf_entry in self.suffixes[suffix]:
                (voc_c, cat_c, gloss_c, pos_c) = suf_entry[1:5]

                # Check the prefix + suffix pair
                pairAC = "%s %s" % (cat_a, cat_c)
                if not pairAC in self.tableAC: continue

                # Check the stem + suffix pair
                pairBC = "%s %s" % (cat_b, cat_c)
                if not pairBC in self.tableBC: continue

                # Ok, it passed!
                buckvoc = "%s%s%s" % (voc_a, voc_b, voc_c)
                analyses.append(buckvoc)

    return analyses

def analyze_word(self, word):
    """Return all possible analyses for the given word."""
    analyses = []
    count = 0
    segments = _build_segments(self, word)

    for prefix, stem, suffix in segments:
        analyses.extend(_check_segment(self, prefix, stem, suffix))

    return analyses

def _build_segments(self, word):
    """Returns all possible segmentations of the given word."""
    segments = []

    for stem_idx, suf_idx in util.segment_indexes(len(word)):
        prefix = word[0:stem_idx]
        stem = word[stem_idx:suf_idx]
        suffix = word[suf_idx:]

        segment = Segment(prefix, stem, suffix)
        if self._valid_segment(segment):
            segments.append(segment)

    return segments


# Read lexicon
df = pd.read_csv("ArSenL_v1.0A.txt", delimiter=";")

# Remove the _1 _2 _3 extension (Not sure what this is)
df.Aramorph_lemma = df.Aramorph_lemma.apply(clean)

# Compute mean score for each word
df = df.groupby("Aramorph_lemma").mean()

# Parse the input to the script using argparse
parser = argparse.ArgumentParser(
	description='Compute sentiment score for arabic text')

parser.add_argument('input',
	help='input file of sentences')

in_file = parser.parse_args().input

# Initialize analyzer
# I'm modifying the core functionality
# of this analyzer so in order not to
# modify the package I import it and
# pass the class to the methods
# after modifying them
analyzer = pyaramorph.Analyzer()

with io.open(in_file, "r", encoding="utf-8") as fp:
	with open("output.csv", "w") as op:
		op.write("Positive sentiment score,Negative sentiment score\n")
		for sentence in fp:
			print(sentence)
			score = computeSentence(df, sentence, analyzer)
			op.write(str(score[0]) + "," + str(score[1]) + "\n")

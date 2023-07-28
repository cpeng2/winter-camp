
from typing import List
import csv
import json
import collections

from nltk import word_tokenize
from features import extract
from case import get_cc, apply_cc, get_tc, apply_tc


def main():
    # test the get_tc function
    print(get_tc("Mary"))
    print(get_tc("LaTeX"))
    filepath = "wsj/wsj_train.txt"
    mcdict = collections.defaultdict(collections.Counter)
    most_frequent = {}
    with open(filepath, "r") as source:
        with open("features.tsv", "w") as sink:
            writer = csv.writer(sink, delimiter="\t")
            for line in source:
                tokens = word_tokenize(line)
                # extract the features and save them to a file
                features = extract(tokens)
                writer.writerow(features)
                writer.writerow("")
                # identify mixed-cased tokens and add them to mcdict
                for token in tokens:
                    token.replace(":", "_")
                    TokenCase, _ = get_tc(token)
                    if TokenCase == TokenCase.MIXED:
                        # print(token)
                        mcdict[token.casefold()][token] += 1
    keys = mcdict.keys()
    for key in keys:
        [(pattern, freq)] = mcdict[key].most_common(1)
        if freq >= 2:
            most_frequent[key] = pattern
    print(most_frequent)
    # Use json.dump to write the dictionary to the file in JSON format
    with open("most_frequent", "w") as json_file:
        json.dump(most_frequent, json_file)


if __name__ == '__main__':
    main()
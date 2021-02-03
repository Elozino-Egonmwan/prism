import os
import sentencepiece as spm
import argparse
import sys
from mech_paraphraser_mini import paraphrase
lang='en'

sp = spm.SentencePieceProcessor()
sp.Load(os.environ['MODEL_DIR'] + '/spm.model')

def getSents(sents):
    sents=paraphrase(sents[0])
    
    sp_sents = [' '.join(sp.EncodeAsPieces(sent)) for sent in sents]

    with open('test.src', 'wt') as fout:
        for sent in sp_sents:
            fout.write(sent + '\n')

    # we also need a dummy output file with the language tag
    with open('test.tgt', 'wt') as fout:
        for sent in sp_sents:
            fout.write(f'<{lang}> \n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sents', type=str, nargs='+')
    args = parser.parse_args()
    getSents(args.sents)

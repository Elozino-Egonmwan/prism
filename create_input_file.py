import os
import sentencepiece as spm

lang='en'

sp = spm.SentencePieceProcessor()
sp.Load(os.environ['MODEL_DIR'] + '/spm.model')
sents = ['Among other things, the developments in terms of turnover, employment, warehousing and prices are recorded.', ]
sp_sents = [' '.join(sp.EncodeAsPieces(sent)) for sent in sents]

with open('test.src', 'wt') as fout:
    for sent in sp_sents:
        fout.write(sent + '\n')

# we also need a dummy output file with the language tag
with open('test.tgt', 'wt') as fout:
    for sent in sp_sents:
        fout.write(f'<{lang}> \n')
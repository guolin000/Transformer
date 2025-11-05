import math
from nltk.translate.bleu_score import corpus_bleu

def compute_bleu(references, hypotheses):
    return corpus_bleu([[ref.split()] for ref in references],
                       [hyp.split() for hyp in hypotheses])

def compute_perplexity(loss):
    return math.exp(loss)

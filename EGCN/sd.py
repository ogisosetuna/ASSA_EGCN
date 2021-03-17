import numpy as np
import spacy
import pickle


nlp = spacy.load('en_core_web_sm')


def dependency_adj_matrix(text):

    document = nlp(text)
    seq_len = len(text.split())
    matrix = np.zeros((seq_len, seq_len)).astype('float32')

    for token in document:
        if token.i < seq_len:
            matrix[token.i][token.i] = 1

            for child in token.children:
                print("child:",child," ","token",token," ","sd:",child.dep_)
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1
                    matrix[child.i][token.i] = 1
                    if child.dep_ == 'acomp':
                        matrix[token.i][child.i] = 1.5
                        matrix[child.i][token.i] = 1.5
                    if child.dep_ == 'det' or child.dep_ == 'prep':
                        matrix[token.i][child.i] = 0.25
                        matrix[child.i][token.i] = 0.25

    return matrix

def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename + '.graph', 'wb')
    # for i in range(0,len(lines),3):
    for i in range(0, 3, 3):
        text_left, deli, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        adj_matrix = dependency_adj_matrix(text_left+' '+aspect+' '+text_right)
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph,fout)
    fout.close()
    import spacy
    from spacy import displacy

    nlp = spacy.load("en_core_web_sm")
    doc = nlp("But the staff was so horrible to us .")
    displacy.serve(doc, style="dep")

if __name__ == '__main__':
    # process('./datasets/acl-14-short-data/train.raw')
    process('./datasets/semeval14/restaurant_train.raw')
    # process('./datasets/semeval14/restaurant_test.raw')
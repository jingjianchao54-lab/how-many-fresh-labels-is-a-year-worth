"""TrainVocab must reproduce scikit-learn's TfidfVectorizer / CountVectorizer fitted on the training rows only."""
import sys, pickle
import numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from features import TrainVocab
from common import WORK

def test_equivalence():
    s = pickle.load(open(WORK / 'sample_reddit.pkl', 'rb'))
    rng = np.random.default_rng(1)
    rows = np.sort(rng.choice(len(s['tok']), 6000, replace=False))
    docs = [s['tok'][i] for i in rows]
    cv = CountVectorizer(ngram_range=(1, 2), token_pattern=r"\S+", lowercase=False, min_df=1, dtype=np.int32)
    x = cv.fit_transform(docs).tocsr(); x.sort_indices()
    tr, te = np.arange(4000), np.arange(4000, 6000)
    tv = TrainVocab(x, tr, tfidf=True, max_features=None)
    ref = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"\S+", lowercase=False, min_df=2, sublinear_tf=True)
    a = ref.fit_transform([docs[i] for i in tr]); b = ref.transform([docs[i] for i in te])
    assert tv.k == a.shape[1]
    # align columns by term
    terms = np.array(cv.get_feature_names_out())[tv.remap >= 0]
    order = np.array([ref.vocabulary_[t] for t in terms])
    assert abs(tv.transform(tr) - a[:, order]).max() < 1e-10
    assert abs(tv.transform(te) - b[:, order]).max() < 1e-10
    tc = TrainVocab(x, tr, tfidf=False, max_features=None)
    refc = CountVectorizer(ngram_range=(1, 2), token_pattern=r"\S+", lowercase=False, min_df=2)
    a = refc.fit_transform([docs[i] for i in tr]); b = refc.transform([docs[i] for i in te])
    order = np.array([refc.vocabulary_[t] for t in terms])
    assert abs(tc.transform(te) - b[:, order]).max() == 0
    # max_features: same number of columns and same total weight as sklearn's cap (ties may differ)
    tv2 = TrainVocab(x, tr, tfidf=True, max_features=5000)
    assert tv2.k == 5000

if __name__ == '__main__':
    test_equivalence(); print('ok')

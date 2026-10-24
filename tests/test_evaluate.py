import sys
import numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
from sklearn.metrics import f1_score
import evaluate as ev

class FakeD:
    K = 4; years = [2000]
    def __init__(self):
        rng = np.random.default_rng(0)
        self.cls = np.repeat(np.arange(4), 50)
        self.pred = np.where(rng.random(200) < 0.6, self.cls, rng.integers(0, 4, 200))
    def rows(self, part, u): return np.arange(200)

def test_scorer_matches_sklearn_and_manual_bootstrap():
    ev.B = 50
    d = FakeD(); sc = ev.Scorer(d, 2)
    f, a, fb, pc, pcb = sc.score(2000, d.pred)
    assert abs(f - f1_score(d.cls, d.pred, average='macro')) < 1e-12
    assert abs(a - np.mean(d.cls == d.pred)) < 1e-12
    w = sc.w[2000]
    for b in range(50):
        idx = np.repeat(np.arange(200), w[b].astype(int))
        assert len(idx) == 200
        assert np.all(np.bincount(d.cls[idx], minlength=4) == 50)           # stratified by class
        assert abs(fb[b] - f1_score(d.cls[idx], d.pred[idx], average='macro')) < 1e-12

def test_breakeven():
    S = [25, 50, 100, 200]
    assert ev.breakeven(S, np.array([.1, .2, .3, .4]), 0.05) == 0.0
    assert np.isinf(ev.breakeven(S, np.array([.1, .2, .3, .4]), 0.5))
    assert abs(ev.breakeven(S, np.array([.1, .2, .3, .4]), 0.2) - 50) < 1e-9
    v = ev.breakeven(S, np.array([.1, .2, .3, .4]), 0.25)
    assert abs(v - np.exp((np.log(50) + np.log(100)) / 2)) < 1e-9

if __name__ == '__main__':
    test_scorer_matches_sklearn_and_manual_bootstrap(); test_breakeven(); print('ok')

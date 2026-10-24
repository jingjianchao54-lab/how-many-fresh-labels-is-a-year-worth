"""One-off patch applying amendment A-02 (reduced M5 design) to encode_minilm.py and run.py."""
p = "encode_minilm.py"
s = open(p).read()
old = s[s.index("def encode(dataset):"):s.index("if __name__")]
new = '''def encode(dataset):
    """A-02: encode validation, test and replicate-1 training items only, at 128 word pieces."""
    from run import Data
    s = pickle.load(open(WORK / f"sample_{dataset}.pkl", "rb"))
    texts = s["nat"]
    d = Data(dataset)
    need = np.zeros(len(texts), dtype=bool)
    need[d.part > 0] = True
    for t in d.years:
        need[d.train_rows(t, 1, d.N)] = True
    idx = np.flatnonzero(need)
    out_dir = WORK / f"minilm_{dataset}_chunks"
    out_dir.mkdir(exist_ok=True)
    model = MiniLM(DATA / "minilm", max_len=128)
    for a in range(0, len(idx), CHUNK):
        path = out_dir / f"{a:07d}.npy"
        if path.exists():
            continue
        np.save(path, model.encode([texts[i] for i in idx[a:a + CHUNK]], batch_size=32))
        print(a, len(idx), flush=True)
    full = np.full((len(texts), 384), np.nan, dtype=np.float32)
    full[idx] = np.concatenate([np.load(out_dir / f"{a:07d}.npy") for a in range(0, len(idx), CHUNK)])
    np.save(WORK / f"minilm_{dataset}.npy", full)
    print("done", len(idx), flush=True)


'''
if "A-02" not in s:
    open(p, "w").write(s.replace(old, new))
r = open("run.py").read()
old = '        if os.environ.get("SKIP_E3") == "1":'
new = '''        if model == "M5":                       # A-02: M5 on replicate 1 only
            jobs = [j for j in jobs if j[1] == 1]
        if os.environ.get("SKIP_E3") == "1":'''
if "A-02" not in r:
    assert old in r
    open("run.py", "w").write(r.replace(old, new))
print("patched")

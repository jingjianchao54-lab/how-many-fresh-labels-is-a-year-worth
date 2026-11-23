"""Record the software and hardware environment of the RV-02 runs: python env_rv02.py <results_dir>"""
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

out = {"python": platform.python_version(), "platform": platform.platform(), "recorded": time.strftime("%Y-%m-%d %H:%M:%S %z")}
for m in ("numpy", "scipy", "sklearn", "torch", "transformers", "tokenizers", "pyarrow", "huggingface_hub"):
    try:
        out[m] = __import__(m).__version__
    except Exception as e:  # noqa: BLE001
        out[m] = f"not importable: {e.__class__.__name__}"
try:
    import torch
    out["cuda_available"] = torch.cuda.is_available()
    out["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    out["bf16_supported"] = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
except Exception:
    pass
out["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader"],
                                   capture_output=True, text=True).stdout.strip()
json.dump(out, open(Path(sys.argv[1]) / "env_rv02.json", "w"), indent=1)
print(json.dumps(out, indent=1))

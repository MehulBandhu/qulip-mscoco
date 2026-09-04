"""Bug fixes for the QuLIP repo, applied to a fresh clone.

Run once after cloning:

    python scripts/patches/apply_fixes.py

Each fix is separate and idempotent, so re-running is safe and reports what was
already done. Every one of these is upstream-worthy; the intention is that this
file eventually becomes a commit series rather than living here forever.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
applied: list[str] = []
skipped: list[str] = []


def patch(rel_path: str, label: str, old: str, new: str, marker: str | None = None):
    """Replace `old` with `new` in one file. `marker` is a string that will be
    present if the fix has already been applied."""
    path = REPO / rel_path
    src = path.read_text()

    if (marker or new) in src:
        skipped.append(label)
        return

    n = src.count(old)
    if n != 1:
        sys.exit(f"FAILED [{label}] in {rel_path}: expected 1 match, found {n}.\n"
                 f"Looking for:\n{old}")

    path.write_text(src.replace(old, new))
    applied.append(label)


# --------------------------------------------------------------- 1. InfoNCE
# The big one. Without normalisation these logits are a raw dot product between
# a unit-norm text vector and a raw CLIP image vector, so the image magnitude
# acts as a per-sample temperature. DisCoCLIP's InfoNCE normalises both. On ARO
# this is the difference between 0.50 (chance) and 0.61.
patch(
    "modules/models/fusion/criteria.py",
    "InfoNCE: cosine similarity",
    """        logits = (text_emb @ image_emb.T) / self.temperature""",
    """        # Cosine, not raw dot product. Image embeddings come in at varying
        # magnitudes, and without this that magnitude ends up scaling the
        # logits per sample, which stops the model training at all.
        text_emb = F.normalize(text_emb, dim=-1)
        image_emb = F.normalize(image_emb, dim=-1)
        logits = (text_emb @ image_emb.T) / self.temperature""",
    marker="Cosine, not raw dot product",
)

_crit = REPO / "modules/models/fusion/criteria.py"
_src = _crit.read_text()
if "import torch.nn.functional as F" not in _src:
    _crit.write_text(_src.replace("import torch.nn as nn",
                                  "import torch.nn as nn\nimport torch.nn.functional as F", 1))

# ----------------------------------------------------- 2. partial batches
# compile_fmap builds the contraction expression once using the configured
# batch size, so the last batch of an epoch dies on a shape check whenever the
# dataset doesn't divide evenly. Cache one expression per shape instead.
patch(
    "modules/models/vision/quantum_map.py",
    "vision: handle short final batch",
    "        return self.contraction_path(*tensor_arr)",
    """        # The compiled path is tied to one batch size, and the last batch of
        # an epoch is usually short. Keep one compiled path per shape.
        shapes = tuple(tuple(t.shape) for t in tensor_arr)
        if not hasattr(self, "_paths"):
            self._paths = {}
        if shapes not in self._paths:
            self._paths[shapes] = contract_expression(self.einsum_expr, *shapes)
        return self._paths[shapes](*tensor_arr)""",
    marker="Keep one compiled path per shape",
)

# ------------------------------------------------------- 3. COCO image ids
# Embeddings are keyed by the integer COCO image_id. file_name is a path like
# 'train2017/000000000599.jpg', which int() can't parse.
patch(
    "modules/data_pipeline/datasets.py",
    "CocoDataset: look up by image_id",
    "        image_ref = row_compiled.get('file_name')",
    """        # Embeddings are keyed by integer image_id; file_name is a path.
        image_ref = row_compiled.get('image_id', row_compiled.get('file_name'))""",
    marker="Embeddings are keyed by integer image_id",
)

# --------------------------------------------- 4. benchmark eval signature
# global_retrieval needs the batch mapper; the hard-negative tasks don't, and
# they hand back (metrics, diagnostics) rather than a bare dict.
patch(
    "scripts/benchmark.py",
    "benchmark: eval call and tuple result",
    "            out = eval_fn(eval_loader)",
    """            if split_info['task'] == 'global_retrieval':
                out = eval_fn(eval_loader, data_engine.eval_mapper)
            else:
                out = eval_fn(eval_loader)
            if isinstance(out, tuple):
                out = out[0]""",
    marker="if split_info['task'] == 'global_retrieval'",
)

# ------------------------------------------------- 5. describe_einsum: COCO
# On COCO each row holds five captions, so both columns are nested a level
# deeper than the ARO rows this was written for.
patch(
    "modules/data_pipeline/engine.py",
    "describe_einsum: multi-caption rows",
    "            tn_arr = list(zip(einsum_arr, symbol_arr))",
    """            tn_arr = []
            for expr, arr in zip(einsum_arr, symbol_arr):
                # COCO rows carry a list of captions each; ARO rows carry one.
                if isinstance(expr, str):
                    tn_arr.append((expr, arr))
                else:
                    tn_arr.extend(zip(expr, arr))""",
    marker="COCO rows carry a list of captions each",
)

# --------------------------------- 6. describe_einsum: non-quantum towers
# Qubits, gates and depth only mean something for the quantum ansatz. TN and
# MLP nodes carry 'name' and 'shape', so report what they do have.
patch(
    "modules/data_pipeline/engine.py",
    "describe_einsum: classical towers",
    "            metrics = tn_metadata(tn_arr)",
    """            first = tn_arr[0][1][0] if tn_arr and tn_arr[0][1] else {}
            if 'op_type' not in first:
                sizes = [len(arr) for _, arr in tn_arr]
                print(f"Network Metrics for {model.__class__.__name__}:")
                print(f"  networks {len(tn_arr)} | tensors per network: "
                      f"max {max(sizes)} avg {sum(sizes) / len(sizes):.1f}")
                return None if return_metrics else None
            metrics = tn_metadata(tn_arr)""",
    marker="Network Metrics for",
)

# --------------------------------- 7. describe_einsum: frozen image encoder
patch(
    "modules/data_pipeline/engine.py",
    "describe_einsum: frozen encoders",
    '            raise ValueError(f"Model {model.__class__.__name__} does not support einsum description.")',
    """            # FrozenCLIP has no circuit to describe, which is the point of it.
            print(f"  {model.__class__.__name__}: no circuit, skipping.")
            return None""",
    marker="FrozenCLIP has no circuit to describe",
)

# --------------------------------------------------------- 8. ansatz choice
# The config's `ansatz` key was being ignored, which made Sim14 (the paper's
# SAP family) unreachable without editing this file.
patch(
    "modules/utils/factory.py",
    "factory: honour the ansatz config key",
    "        ansatz = CustomV5Ansatz(obmap=obmap, layers=compiler['layers'])",
    """        from modules.compilation.quantum.ansatz import (
            BrickworkAnsatz, IQPAnsatz, Sim14Ansatz,
        )
        ansatze = {
            'custom_v5': CustomV5Ansatz,
            'iqp': IQPAnsatz,
            'brickwork': BrickworkAnsatz,
            'sim14': Sim14Ansatz,
        }
        name = compiler.get('ansatz', 'custom_v5')
        if name not in ansatze:
            raise ValueError(f"unknown ansatz '{name}', options: {sorted(ansatze)}")
        print(f"  text ansatz : {name}")
        ansatz = ansatze[name](obmap=obmap, layers=compiler['layers'])""",
    marker="unknown ansatz",
)

# ------------------------------------------------------- 9. device override
# Complex64 support on MPS is patchy and the text tower is complex throughout,
# so being able to force CPU without editing code saves a lot of time.
patch(
    "modules/utils/general.py",
    "device override via QULIP_DEVICE",
    """def get_device():
    if torch.backends.mps.is_available():""",
    """def get_device():
    override = os.getenv("QULIP_DEVICE")
    if override:
        return torch.device(override)
    if torch.backends.mps.is_available():""",
    marker="QULIP_DEVICE",
)

_gen = REPO / "modules/utils/general.py"
_src = _gen.read_text()
if not _src.lstrip().startswith("import os") and "\nimport os" not in _src:
    _gen.write_text("import os\n" + _src)

# --------------------------------------------------- 10. lazy heavy imports
# quantum_ops pulls in qiskit at module level, but only tn2qiskit needs it, and
# that's circuit export rather than training. Same story with spacy in
# grammar.py: it was being loaded as a default argument, so importing the
# module cost you a model load whether you parsed anything or not.
_ops = REPO / "modules/utils/quantum_ops.py"
_src = _ops.read_text()
if _src.startswith("import torch, math\nfrom qiskit"):
    _src = _src.replace(
        "import torch, math\nfrom qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister\n"
        "from qiskit.circuit import Parameter\n",
        "import torch, math\n")
    _src = _src.replace(
        "def tn2qiskit(einsum_expr, gate_arr, param_dict={}, meas_output=True):",
        "def tn2qiskit(einsum_expr, gate_arr, param_dict={}, meas_output=True):\n"
        "    # Only needed for circuit export, so don't make training depend on it.\n"
        "    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister\n"
        "    from qiskit.circuit import Parameter\n")
    _ops.write_text(_src)
    applied.append("quantum_ops: lazy qiskit import")
else:
    skipped.append("quantum_ops: lazy qiskit import")

_gram = REPO / "modules/symbolic/grammar.py"
_src = _gram.read_text()
_default_arg = 'def lemmatise_sent(caption, nlp=spacy.load("en_core_web_sm")):'
if _default_arg in _src:
    _gram.write_text(_src.replace(
        _default_arg,
        "_NLP = None\n\n\n"
        "def lemmatise_sent(caption, nlp=None):\n"
        "    # Loading the spacy model in the signature meant every import of this\n"
        "    # module paid for it, including callers that only want strip_sent.\n"
        "    global _NLP\n"
        "    if nlp is None:\n"
        "        if _NLP is None:\n"
        '            _NLP = spacy.load("en_core_web_sm")\n'
        "        nlp = _NLP"))
    applied.append("grammar: load spacy on demand")
else:
    skipped.append("grammar: load spacy on demand")


print(f"applied {len(applied)}:")
for label in applied:
    print(f"  + {label}")
if skipped:
    print(f"already present {len(skipped)}:")
    for label in skipped:
        print(f"  . {label}")

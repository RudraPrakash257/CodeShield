# CodeShield AI — Build Guide

> An explainable, ML-based system that flags suspiciously similar Python assignments even when variable names, comments, formatting or statement structure have been disguised. **It assists instructors; it never accuses students.**

This document is both a **human build guide** and a **spec you can paste into an AI coding assistant** (Claude Code, Cursor, etc.). Every code block that starts with `# file: <path>` is a complete file. Copy the blocks into that path and you have a working MVP.

---

## 0. Read this first

| Item | Decision |
|---|---|
| Language scope | Python submissions only |
| Core ML | Random Forest on multi-view pairwise features, probability-calibrated |
| Explainability | Per-pair SHAP reasons + matched-code highlighting + "likely disguise" panel |
| Novelty | Mutation engine (training data) + canonicalization (reverses common disguises) + class-relative context |
| UI | Streamlit + Plotly |
| Not in MVP | CodeBERT, code execution, Java support, deep-model training |
| Build order | Working end-to-end first, then upgrades. Never sacrifice the demo for a fancy model |

**Assistant prompt (optional).** Paste this into your coding assistant, then attach this file:

```text
Build the CodeShield AI project exactly as specified in the attached guide.
Work phase by phase (Section 4). After each phase, run the "Checkpoint" command
and fix failures before moving on. Do not add features that are not in the guide
until every Phase 1-8 checkpoint passes.
```

---

## 1. What the system does

```
ZIP / .py files (+ optional starter code)
        │
        ▼
Validate → strip comments/docstrings → parse AST
        │
        ▼
Canonicalize  (undo common disguises: x += 1, not x <= 0, temp vars, dead code, main() wrappers)
        │
        ▼
Normalize identifiers (VAR_1, FUNC_1) → per-file feature store
        │
        ▼
All unique pairs N(N-1)/2 → 12 pairwise features
        │
        ▼
Random Forest → calibrated probability → Low / Review / High / Very high
        │
        ▼
SHAP reasons · matched blocks · likely disguises · class percentile · group graph
        │
        ▼
Instructor review (confirm / dismiss) → CSV export
```

**Design principle:** a score is a reason to *look*, never proof of misconduct. Every alert must show its evidence.

---

## 2. Upgrades over the original guidebook

| Weakness in the base plan | Fix included here |
|---|---|
| Train and test on the same mutation engine → circular results | Hand-made **real-disguise holdout set** evaluated separately |
| "Why ML?" has no answer | **Baseline comparison** (difflib, TF-IDF, AST-only, mean-of-signals) at an equal false-positive budget |
| `feature_importances_` is global, not per pair | **SHAP TreeExplainer** per-pair reasons |
| Forest probabilities are not true probabilities | **Calibration** on a held-out family split |
| Natural similarity on short assignments | **Hard negatives**, starter-code subtraction, **class-relative percentile** |
| Disguises must be reversed to be detected | **Canonicalization pass** that mirrors the mutation engine |
| No proof that the model beats simple tools | **Per-mutation detection chart** (the demo's strongest slide) |
| Static "behavior" claimed as behavior | Honestly labelled **behavioral fingerprint (syntactic approximation)** |

---

## 3. Setup

**Python 3.10+** (3.12 recommended).

```bash
mkdir codeshield-ai && cd codeshield-ai
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
mkdir -p src tests models data/raw_submissions data/holdout data/generated_pairs data/demo_submissions
touch src/__init__.py tests/__init__.py
```

```text
# file: requirements.txt
streamlit>=1.40
pandas>=2.0
numpy>=1.26
scikit-learn>=1.6
networkx>=3.2
plotly>=5.20
joblib>=1.3
shap>=0.46
pytest>=8
```

```bash
pip install -r requirements.txt
```

`scikit-learn>=1.6` is required for `FrozenEstimator` (used for calibration).

### Project structure

```
codeshield-ai/
├── app.py                     # Streamlit dashboard
├── requirements.txt
├── README.md
├── data/
│   ├── raw_submissions/<problem>/<solution>.py   # training originals
│   ├── holdout/                                  # hand-made real-disguise pairs
│   │   ├── pairs.csv                             # columns: a,b,label
│   │   └── *.py
│   ├── generated_pairs/                          # feature tables written by training
│   ├── demo_submissions/                         # files for the live demo
│   └── feedback.csv                              # instructor confirm/dismiss log
├── models/                    # written by train_model.py
├── src/
│   ├── preprocessing.py       # load, validate, parse, pseudonymise
│   ├── ast_features.py        # canonicalize + normalize + per-file features
│   ├── behavioral_features.py # syntactic behavior fingerprint
│   ├── mutation_engine.py     # disguise generator (training data)
│   ├── pair_features.py       # 12 pairwise features
│   ├── train_model.py         # dataset build, RF, calibration, evaluation
│   ├── predict.py             # analyze() used by the dashboard
│   ├── explain.py             # SHAP reasons, disguise heuristics, matched blocks
│   └── graph_analysis.py      # Louvain groups + graph figure
└── tests/test_smoke.py
```

---

## 4. Build phases

Each phase ends with a **Checkpoint**. Do not skip them.

### Phase 1: Load, validate, parse (Day 1 morning)

```python
# file: src/preprocessing.py
from __future__ import annotations

import ast
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

MAX_FILES = 500
MAX_BYTES = 200_000


@dataclass
class Submission:
    name: str
    raw: str


def load_folder(folder: str | Path) -> list[Submission]:
    subs = []
    for p in sorted(Path(folder).rglob("*.py"))[:MAX_FILES]:
        if p.stat().st_size <= MAX_BYTES:
            subs.append(Submission(p.name, p.read_text(encoding="utf-8", errors="replace")))
    return subs


def load_zip(data: bytes) -> list[Submission]:
    """Read .py files straight from memory (nothing is extracted to disk)."""
    subs = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            if info.is_dir() or not info.filename.endswith(".py"):
                continue
            if info.filename.startswith("__MACOSX") or info.file_size > MAX_BYTES:
                continue
            subs.append(Submission(Path(info.filename).name,
                                   z.read(info).decode("utf-8", errors="replace")))
            if len(subs) >= MAX_FILES:
                break
    return subs


def pseudonymise(subs: list[Submission]):
    """Student_01.py, Student_02.py ... The mapping is kept in memory only."""
    mapping, out = {}, []
    for i, s in enumerate(subs, 1):
        new = f"Student_{i:02d}.py"
        mapping[new] = s.name
        out.append(Submission(new, s.raw))
    return out, mapping


def parse_source(raw: str):
    """Return (tree, error). Comments disappear here: the AST never stores them."""
    try:
        return ast.parse(raw), None
    except SyntaxError as e:
        return None, f"Syntax error on line {e.lineno}: {e.msg}"
    except (ValueError, RecursionError, MemoryError) as e:
        return None, f"Cannot parse file: {e}"


def strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            b = node.body
            if (b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant)
                    and isinstance(b[0].value.value, str)):
                node.body = b[1:] or [ast.Pass()]
    return tree
```

### Phase 2: Behavioral fingerprint

This is a **static, syntactic** approximation of behavior (the order of INPUT / LOOP / CONDITION / UPDATE / OUTPUT operations). Present it honestly as such; running student code is out of scope for the MVP.

```python
# file: src/behavioral_features.py
from __future__ import annotations

import ast


def behavior_fingerprint(tree: ast.AST) -> list[str]:
    seq: list[str] = []

    def visit(node: ast.AST, in_loop: bool) -> None:
        if isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
            seq.append("LOOP")
            in_loop = True
        elif isinstance(node, ast.If):
            seq.append("COND")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "input":
                seq.append("INPUT")
            elif node.func.id == "print":
                seq.append("OUTPUT")
        elif isinstance(node, (ast.Assign, ast.AugAssign)) and in_loop:
            seq.append("UPDATE")
        elif isinstance(node, ast.Return):
            seq.append("RETURN")
        elif isinstance(node, (ast.Break, ast.Continue)):
            seq.append("JUMP")
        for child in ast.iter_child_nodes(node):
            visit(child, in_loop)

    visit(tree, False)
    return seq
```

### Phase 3: Canonicalize + normalize + per-file features

This is the heart of the disguise resistance. The canonicalizer **reverses the same disguises your mutation engine creates**, so equivalent programs collapse to the same form.

| Disguise | Canonical form |
|---|---|
| `x += 1` | `x = x + 1` |
| `if not x <= 0` | `if 0 < x` (also `a > b` becomes `b < a`) |
| `t = a + b; r = t` | `r = a + b` (single-use temporaries inlined) |
| `unused = 0` never read | removed |
| `def main(): ...` + `main()` / `if __name__ == "__main__"` | body inlined at module level |
| `b + a` vs `a + b` | operands sorted (best effort, comparison only) |
| Renamed identifiers | `VAR_1`, `FUNC_1` |
| Starter/template code | removed (optional) |

```python
# file: src/ast_features.py
from __future__ import annotations

import ast
import builtins
import hashlib
import io
import tokenize
from collections import Counter
from dataclasses import dataclass

from .behavioral_features import behavior_fingerprint
from .preprocessing import parse_source, strip_docstrings

BUILTINS = set(dir(builtins))

# Logical negation of each comparison operator
INVERT = {ast.Lt: ast.GtE, ast.LtE: ast.Gt, ast.Gt: ast.LtE, ast.GtE: ast.Lt,
          ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Is: ast.IsNot, ast.IsNot: ast.Is,
          ast.In: ast.NotIn, ast.NotIn: ast.In}
COMMUTATIVE = (ast.Add, ast.Mult, ast.BitAnd, ast.BitOr, ast.BitXor)
CONTROL_NODES = {"For", "While", "If", "Try", "Return", "Break", "Continue",
                 "With", "FunctionDef", "Lambda", "ListComp", "Call"}
COMMON_ATTRS = {"append", "extend", "pop", "join", "split", "strip", "format", "get",
                "items", "keys", "values", "sort", "lower", "upper", "count", "index"}
NESTING = (ast.For, ast.While, ast.If, ast.Try, ast.With, ast.AsyncFor, ast.AsyncWith)
_PURE = (ast.Constant, ast.Name, ast.BinOp, ast.UnaryOp, ast.List, ast.Tuple,
         ast.Load, ast.operator, ast.unaryop)


# ------------------------------------------------------------------ helpers
def dfs(node: ast.AST):
    yield node
    for child in ast.iter_child_nodes(node):
        yield from dfs(child)


def _is_str(n: ast.AST) -> bool:
    return isinstance(n, ast.Constant) and isinstance(n.value, str)


def fix_empty(tree: ast.AST) -> ast.AST:
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.For,
                          ast.While, ast.If, ast.With, ast.Try, ast.ExceptHandler)):
            if isinstance(n.body, list) and not n.body:
                n.body = [ast.Pass()]
    return tree


# ------------------------------------------------------- canonicalization
class Canonicalizer(ast.NodeTransformer):
    def visit_AugAssign(self, node):
        self.generic_visit(node)
        if isinstance(node.target, ast.Name):
            new = ast.Assign(
                targets=[ast.Name(id=node.target.id, ctx=ast.Store())],
                value=ast.BinOp(left=ast.Name(id=node.target.id, ctx=ast.Load()),
                                op=node.op, right=node.value))
            return ast.copy_location(new, node)
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        if len(node.ops) == 1 and isinstance(node.ops[0], (ast.Gt, ast.GtE)):
            swap = ast.Lt if isinstance(node.ops[0], ast.Gt) else ast.LtE
            node.left, node.comparators = node.comparators[0], [node.left]
            node.ops = [swap()]
        return node

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if (isinstance(node.op, ast.Not) and isinstance(node.operand, ast.Compare)
                and len(node.operand.ops) == 1 and type(node.operand.ops[0]) in INVERT):
            cmp_ = node.operand
            cmp_.ops = [INVERT[type(cmp_.ops[0])]()]
            return self.visit_Compare(cmp_)
        return node


def _is_main_guard(s: ast.stmt) -> bool:
    return (isinstance(s, ast.If) and isinstance(s.test, ast.Compare)
            and isinstance(s.test.left, ast.Name) and s.test.left.id == "__name__"
            and not s.orelse)


def unwrap_main(tree: ast.Module) -> ast.Module:
    """Inline `if __name__ == "__main__"` blocks and zero-arg functions called once."""
    body = []
    for s in tree.body:
        body.extend(s.body if _is_main_guard(s) else [s])
    tree.body = body

    loads = Counter(n.id for n in ast.walk(tree)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load))
    for f in list(tree.body):
        a = getattr(f, "args", None)
        if not (isinstance(f, ast.FunctionDef) and not f.decorator_list and loads[f.name] == 1
                and not (a.args or a.vararg or a.kwonlyargs or a.kwarg or a.posonlyargs)):
            continue
        if any(isinstance(n, (ast.Return, ast.Global, ast.Nonlocal, ast.Yield, ast.YieldFrom))
               for n in ast.walk(f)):
            continue
        for i, c in enumerate(tree.body):
            if (isinstance(c, ast.Expr) and isinstance(c.value, ast.Call)
                    and isinstance(c.value.func, ast.Name) and c.value.func.id == f.name
                    and not c.value.args and not c.value.keywords):
                tree.body[i:i + 1] = f.body
                tree.body.remove(f)
                break
    return tree


def inline_temps(tree: ast.AST) -> None:
    """t = expr ; y = t   ->   y = expr   (when t is used exactly once)."""
    loads = Counter(n.id for n in ast.walk(tree)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load))
    stores = Counter(n.id for n in ast.walk(tree)
                     if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store))
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            body = getattr(node, field, None)
            if not isinstance(body, list):
                continue
            out, i = [], 0
            while i < len(body):
                s = body[i]
                nxt = body[i + 1] if i + 1 < len(body) else None
                if (isinstance(s, ast.Assign) and len(s.targets) == 1
                        and isinstance(s.targets[0], ast.Name)
                        and isinstance(nxt, ast.Assign) and isinstance(nxt.value, ast.Name)
                        and nxt.value.id == s.targets[0].id
                        and loads[nxt.value.id] == 1 and stores[nxt.value.id] == 1):
                    nxt.value = s.value
                    out.append(nxt)
                    i += 2
                else:
                    out.append(s)
                    i += 1
            setattr(node, field, out)


def remove_dead_code(tree: ast.AST) -> ast.AST:
    """Drop assignments of pure values to names that are never read."""
    class Dead(ast.NodeTransformer):
        def __init__(self, loaded):
            self.loaded = loaded

        def visit_ClassDef(self, node):      # class attributes are read via attributes
            return node

        def visit_Assign(self, node):
            t = node.targets
            if (len(t) == 1 and isinstance(t[0], ast.Name) and t[0].id not in self.loaded
                    and all(isinstance(n, _PURE) for n in ast.walk(node.value))):
                return None
            return node

    for _ in range(2):
        loaded = {n.id for n in ast.walk(tree)
                  if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        tree = Dead(loaded).visit(tree)
    return tree


def canonicalize(tree: ast.Module) -> ast.Module:
    tree = unwrap_main(tree)
    tree = Canonicalizer().visit(tree)
    inline_temps(tree)
    tree = remove_dead_code(tree)
    fix_empty(tree)
    return ast.fix_missing_locations(tree)


def remove_boilerplate(tree: ast.Module, template: ast.Module) -> ast.Module:
    """Delete statements that exactly match the instructor's starter code."""
    sigs = {ast.dump(n) for n in ast.walk(template) if isinstance(n, ast.stmt)}

    class R(ast.NodeTransformer):
        def visit(self, node):
            if isinstance(node, ast.stmt) and ast.dump(node) in sigs:
                return None
            return super().visit(node)

    tree = R().visit(tree)
    fix_empty(tree)
    return tree


# ---------------------------------------------------------- normalization
class IdentNormalizer(ast.NodeTransformer):
    """Consistently rename identifiers: VAR_n, FUNC_n, CLASS_n. Keeps builtins and imports."""

    def __init__(self, tree: ast.AST):
        stored = {n.id for n in ast.walk(tree)
                  if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        self.keep = BUILTINS - stored
        self.funcs: dict[str, str] = {}
        self.vars: dict[str, str] = {}
        fi = ci = 0
        for n in ast.walk(tree):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                self.keep |= {(a.asname or a.name).split(".")[0] for a in n.names}
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name not in self.funcs:
                fi += 1
                self.funcs[n.name] = f"FUNC_{fi}"
            elif isinstance(n, ast.ClassDef) and n.name not in self.funcs:
                ci += 1
                self.funcs[n.name] = f"CLASS_{ci}"

    def _map(self, name: str) -> str:
        if name in self.keep or (name.startswith("__") and name.endswith("__")):
            return name
        if name in self.funcs:
            return self.funcs[name]
        return self.vars.setdefault(name, f"VAR_{len(self.vars) + 1}")

    def visit_Name(self, n):
        n.id = self._map(n.id)
        return n

    def visit_arg(self, n):
        n.arg = self._map(n.arg)
        return n

    def visit_FunctionDef(self, n):
        n.name = self._map(n.name)
        self.generic_visit(n)
        return n

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef

    def visit_ExceptHandler(self, n):
        if n.name:
            n.name = self._map(n.name)
        self.generic_visit(n)
        return n


class SortCommutative(ast.NodeTransformer):
    """Best-effort operand ordering. Used for comparison only; never executed."""

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, COMMUTATIVE) and not (
                isinstance(node.op, ast.Add) and (_is_str(node.left) or _is_str(node.right))):
            if ast.dump(node.left) > ast.dump(node.right):
                node.left, node.right = node.right, node.left
        return node


# ------------------------------------------------------- per-file features
@dataclass
class FileFeatures:
    name: str
    raw: str
    normalized: str
    tokens: list
    ast_seq: list
    ast_counts: Counter
    subtree: Counter
    control: Counter
    behavior: list
    complexity: dict
    raw_idents: set
    rare: set


def tokenize_code(code: str) -> list[str]:
    toks: list[str] = []
    try:
        for t in tokenize.generate_tokens(io.StringIO(code).readline):
            if t.type in (tokenize.NAME, tokenize.OP):
                toks.append(t.string)
            elif t.type == tokenize.NUMBER:
                toks.append("NUM")
            elif t.type == tokenize.STRING:
                toks.append("STR")
    except (tokenize.TokenError, IndentationError):
        pass
    return toks


def subtree_hashes(tree: ast.AST, min_size: int = 4) -> Counter:
    """Bag of structural subtree hashes: robust to reordering and helper extraction."""
    out: Counter = Counter()

    def visit(node):
        if isinstance(node, ast.expr_context):
            return None, 0
        label = type(node).__name__
        if isinstance(node, ast.Name):
            label += ":" + node.id
        elif isinstance(node, ast.Constant):
            label += ":" + type(node.value).__name__
        elif isinstance(node, ast.Attribute):
            label += ":" + node.attr
        kids, size = [], 1
        for c in ast.iter_child_nodes(node):
            h, s = visit(c)
            if h is not None:
                kids.append(h)
                size += s
        h = hashlib.md5((label + "(" + ",".join(kids) + ")").encode()).hexdigest()[:10]
        if size >= min_size and isinstance(node, (ast.stmt, ast.expr)):
            out[h] += 1
        return h, size

    visit(tree)
    return out


def max_nesting(node: ast.AST, d: int = 0) -> int:
    best = d
    for c in ast.iter_child_nodes(node):
        best = max(best, max_nesting(c, d + isinstance(c, NESTING)))
    return best


def rare_tokens(tree: ast.AST) -> set:
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant):
            v = n.value
            if isinstance(v, str) and len(v) >= 3:
                out.add(("s", v))
            elif isinstance(v, (int, float)) and not isinstance(v, bool) and v not in (0, 1, 2, -1):
                out.add(("n", v))
        elif isinstance(n, ast.Attribute) and n.attr not in COMMON_ATTRS:
            out.add(("a", n.attr))
    return out


def collect_idents(tree: ast.AST) -> set:
    ids = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id not in BUILTINS:
            ids.add(n.id)
        elif isinstance(n, ast.arg):
            ids.add(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            ids.add(n.name)
    return ids


def extract_features(name: str, raw: str, template_src: str | None = None):
    """Return (FileFeatures | None, error_message | None)."""
    tree, err = parse_source(raw)
    if err:
        return None, err
    tree = strip_docstrings(tree)
    raw_idents = collect_idents(tree)
    tree = canonicalize(tree)

    if template_src:
        ttree, terr = parse_source(template_src)
        if ttree is not None:
            tree = remove_boilerplate(tree, canonicalize(strip_docstrings(ttree)))

    rare = rare_tokens(tree)
    behavior = behavior_fingerprint(tree)
    control = Counter(type(n).__name__ for n in ast.walk(tree)
                      if type(n).__name__ in CONTROL_NODES)

    tree = IdentNormalizer(tree).visit(tree)
    tree = SortCommutative().visit(tree)
    ast.fix_missing_locations(tree)
    normalized = ast.unparse(tree)

    ast_seq = [type(n).__name__ for n in dfs(tree) if not isinstance(n, ast.expr_context)]
    complexity = {
        "lines": len(normalized.splitlines()),
        "functions": sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                         for n in ast.walk(tree)),
        "nesting": max_nesting(tree),
        "nodes": len(ast_seq),
    }
    return FileFeatures(
        name=name, raw=raw, normalized=normalized, tokens=tokenize_code(normalized),
        ast_seq=ast_seq, ast_counts=Counter(ast_seq), subtree=subtree_hashes(tree),
        control=control, behavior=behavior, complexity=complexity,
        raw_idents=raw_idents, rare=rare), None
```

**Checkpoint 3**

```bash
python - <<'EOF'
from src.ast_features import extract_features
a = "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n"
b = "def add_all(items):\n    acc = 0  # start\n    for it in items:\n        acc = acc + it\n    return acc\n"
fa, _ = extract_features("a", a); fb, _ = extract_features("b", b)
print(fa.normalized == fb.normalized)   # expect True
print(fa.normalized)
EOF
```

### Phase 4: Mutation engine (Day 1 evening)

The mutation engine creates realistic disguised copies **and records which disguise was used**. That record powers the per-mutation robustness chart.

```python
# file: src/mutation_engine.py
from __future__ import annotations

import ast
import builtins
import keyword
import random

BUILTINS = set(dir(builtins))
INVERT = {ast.Lt: ast.GtE, ast.LtE: ast.Gt, ast.Gt: ast.LtE, ast.GtE: ast.Lt,
          ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
WORDS = ["alpha", "beta", "gamma", "delta", "item", "data", "value", "output", "result_val",
         "cur", "acc", "idx", "elem", "num", "buffer", "tally", "node", "entry", "chunk", "temp_val"]
COMMENTS = ["# compute result", "# loop over items", "# helper", "# TODO: clean up",
            "# check condition", "# update value", "# main logic"]


class _Mut(ast.NodeTransformer):
    def __init__(self, rng: random.Random):
        self.rng, self.changed, self.n = rng, False, 0


def _run(cls, src: str, rng: random.Random):
    tree = ast.parse(src)
    t = cls(rng)
    tree = t.visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) if t.changed else None


# 1. identifier renaming ------------------------------------------------------
class _Renamer(ast.NodeTransformer):
    def __init__(self, m):
        self.m = m

    def visit_Name(self, n):
        n.id = self.m.get(n.id, n.id)
        return n

    def visit_arg(self, n):
        n.arg = self.m.get(n.arg, n.arg)
        return n

    def visit_FunctionDef(self, n):
        n.name = self.m.get(n.name, n.name)
        self.generic_visit(n)
        return n

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef

    def visit_ExceptHandler(self, n):
        if n.name:
            n.name = self.m.get(n.name, n.name)
        self.generic_visit(n)
        return n


def m_rename(src, rng):
    tree = ast.parse(src)
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            imported |= {(a.asname or a.name).split(".")[0] for a in n.names}
    kw = {k.arg for n in ast.walk(tree) if isinstance(n, ast.Call) for k in n.keywords if k.arg}
    targets = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            targets.add(n.id)
        elif isinstance(n, ast.arg):
            targets.add(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            targets.add(n.name)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            targets.add(n.name)
    targets = {t for t in targets - imported - kw
               if not t.startswith("__") and t not in ("self", "cls")}
    if not targets:
        return None
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | BUILTINS | set(keyword.kwlist)
    pool = [w for w in WORDS + [f"v{i}" for i in range(200)] if w not in used]
    rng.shuffle(pool)
    mapping = dict(zip(sorted(targets), pool))
    return ast.unparse(_Renamer(mapping).visit(tree))


# 2. comments / formatting (text-level) -----------------------------------------
def m_comments(src, rng):
    lines = ast.unparse(ast.parse(src)).splitlines()
    out = []
    for ln in lines:
        indent = ln[: len(ln) - len(ln.lstrip())]
        if rng.random() < 0.3:
            out.append(indent + rng.choice(COMMENTS))
        if rng.random() < 0.15:
            out.append("")
        tail = "  # " + rng.choice(["ok", "step", "note"]) if rng.random() < 0.15 else ""
        out.append(ln + tail)
    return "\n".join(out)


# 3. equivalent assignment -----------------------------------------------------
class AugCompress(_Mut):
    def visit_Assign(self, node):
        self.generic_visit(node)
        t, v = node.targets, node.value
        if (len(t) == 1 and isinstance(t[0], ast.Name) and isinstance(v, ast.BinOp)
                and isinstance(v.left, ast.Name) and v.left.id == t[0].id):
            self.changed = True
            return ast.copy_location(
                ast.AugAssign(target=ast.Name(id=t[0].id, ctx=ast.Store()), op=v.op, value=v.right),
                node)
        return node


class AugExpand(_Mut):
    def visit_AugAssign(self, node):
        self.generic_visit(node)
        if isinstance(node.target, ast.Name):
            self.changed = True
            return ast.copy_location(ast.Assign(
                targets=[ast.Name(id=node.target.id, ctx=ast.Store())],
                value=ast.BinOp(left=ast.Name(id=node.target.id, ctx=ast.Load()),
                                op=node.op, right=node.value)), node)
        return node


def m_augassign(src, rng):
    return _run(AugCompress, src, rng) or _run(AugExpand, src, rng)


# 4. condition transform -------------------------------------------------------
class CondFlip(_Mut):
    def visit_If(self, node):
        self.generic_visit(node)
        t = node.test
        if (isinstance(t, ast.Compare) and len(t.ops) == 1 and type(t.ops[0]) in INVERT
                and self.rng.random() < 0.8):
            t.ops = [INVERT[type(t.ops[0])]()]
            node.test = ast.UnaryOp(op=ast.Not(), operand=t)   # not (negated comparison)
            self.changed = True
        return node


def m_condition(src, rng):
    return _run(CondFlip, src, rng)


# 5. temporary variable --------------------------------------------------------
class TempVar(_Mut):
    def visit_Assign(self, node):
        self.generic_visit(node)
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, (ast.BinOp, ast.Call)) and self.rng.random() < 0.6):
            self.n += 1
            tmp = f"tmp_{self.n}"
            self.changed = True
            return [ast.Assign(targets=[ast.Name(id=tmp, ctx=ast.Store())], value=node.value),
                    ast.Assign(targets=[node.targets[0]], value=ast.Name(id=tmp, ctx=ast.Load()))]
        return node


def m_temp(src, rng):
    return _run(TempVar, src, rng)


# 6. dead code insertion ---------------------------------------------------------
class DeadCode(_Mut):
    def _inject(self, body):
        if not body:
            return body
        self.n += 1
        stmt = ast.Assign(targets=[ast.Name(id=f"unused_{self.n}", ctx=ast.Store())],
                          value=ast.Constant(value=self.rng.randint(0, 99)))
        pos = self.rng.randint(0, len(body))
        self.changed = True
        return body[:pos] + [stmt] + body[pos:]

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if self.rng.random() < 0.7:
            node.body = self._inject(node.body)
        return node

    def visit_Module(self, node):
        self.generic_visit(node)
        if not self.changed or self.rng.random() < 0.5:
            node.body = self._inject(node.body)
        return node


def m_dead(src, rng):
    return _run(DeadCode, src, rng)


# 7. function extraction: wrap script body in main()/run()/solve() -----------------
def m_wrap_main(src, rng):
    tree = ast.parse(src)
    keep, move = [], []
    for s in tree.body:
        (keep if isinstance(s, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))
         else move).append(s)
    if len(move) < 2:
        return None
    name = rng.choice(["main", "run", "solve", "program", "start"])
    if any(getattr(s, "name", None) == name for s in keep):
        return None
    assigned = {n.id for s in move for n in ast.walk(s)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    read_in_defs = {n.id for s in keep if isinstance(s, (ast.FunctionDef, ast.ClassDef))
                    for n in ast.walk(s) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    if assigned & read_in_defs:          # would change variable scoping
        return None
    fn = ast.parse("def f():\n    pass").body[0]
    fn.name, fn.body = name, move
    tail = (ast.parse(f'if __name__ == "__main__":\n    {name}()').body[0]
            if rng.random() < 0.5 else ast.parse(f"{name}()").body[0])
    tree.body = keep + [fn, tail]
    return ast.unparse(ast.fix_missing_locations(tree))


# 8. function reordering ---------------------------------------------------------
def m_reorder(src, rng):
    tree = ast.parse(src)
    idx = [i for i, s in enumerate(tree.body) if isinstance(s, ast.FunctionDef)]
    if len(idx) < 2:
        return None
    funcs = [tree.body[i] for i in idx]
    rng.shuffle(funcs)
    for i, f in zip(idx, funcs):
        tree.body[i] = f
    return ast.unparse(tree)


MUTATIONS = {"rename": m_rename, "comments_formatting": m_comments, "augassign": m_augassign,
             "condition_flip": m_condition, "temp_variable": m_temp, "dead_code": m_dead,
             "wrap_main": m_wrap_main, "reorder_functions": m_reorder}


def mutate(src: str, rng: random.Random, max_k: int = 3):
    """Apply 1..max_k stacked disguises. Returns (new_source, [applied mutation names])."""
    names = list(MUTATIONS)
    rng.shuffle(names)
    k, applied, cur = rng.randint(1, max_k), [], src
    for n in names:
        if len(applied) >= k:
            break
        out = MUTATIONS[n](cur, rng)
        if out is None:
            continue
        try:
            ast.parse(out)
        except SyntaxError:
            continue
        cur = out
        applied.append(n)
    return cur, applied


def partial_copy(src_a: str, src_b: str, rng: random.Random, frac: float = 0.6) -> str:
    """Keep the first `frac` of A and append unrelated code from B (a partial-copy positive)."""
    a, b = ast.parse(src_a), ast.parse(src_b)
    k = max(1, int(len(a.body) * frac))
    extra = [s for s in b.body if not isinstance(s, (ast.Import, ast.ImportFrom))]
    a.body = a.body[:k] + extra[: max(1, len(extra) // 2)]
    return ast.unparse(a)
```

> Mutations are validated by **parsing**, not by proving semantic equivalence. Spot-check a few mutated files by running them.

**Checkpoint 4**

```bash
python - <<'EOF'
import random
from src.mutation_engine import mutate
src = "def total(xs):\n    s = 0\n    for x in xs:\n        if x > 0:\n            s = s + x\n    return s\n"
for seed in range(3):
    out, muts = mutate(src, random.Random(seed))
    print(muts); print(out); print("---")
EOF
```

### Phase 5: Pairwise features

Twelve model features plus one diagnostic (`raw_text_ratio`, the naive "text diff" baseline, not fed to the model).

```python
# file: src/pair_features.py
from __future__ import annotations

import itertools
from collections import Counter
from difflib import SequenceMatcher

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_FEATURES = [
    "tfidf_cosine", "ngram3_jaccard", "ngram5_jaccard", "ast_bag_sim", "ast_seq_ratio",
    "ast_freq_sim", "control_flow_sim", "behavior_sim", "complexity_diff", "rare_overlap",
    "length_ratio", "raw_ident_jaccard",
]
DIAG_FEATURES = ["raw_text_ratio"]


def make_vectorizer() -> TfidfVectorizer:
    # Tokens are already normalized ("VAR_1", "for", "+", "NUM"), so split on whitespace only.
    return TfidfVectorizer(token_pattern=r"\S+", lowercase=False,
                           ngram_range=(1, 3), sublinear_tf=True)


def ngrams(tokens: list, n: int) -> set:
    return set(zip(*(tokens[i:] for i in range(n))))


def jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def wjaccard(a: Counter, b: Counter) -> float:
    keys = a.keys() | b.keys()
    den = sum(max(a[k], b[k]) for k in keys)
    return sum(min(a[k], b[k]) for k in keys) / den if den else 0.0


def seq_ratio(a: list, b: list, cap: int = 3000) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a[:cap], b[:cap], autojunk=False).ratio()


def complexity_diff(a: dict, b: dict) -> float:
    """0 = identical profile; 1 = completely different."""
    return sum(abs(a[k] - b[k]) / max(a[k], b[k], 1) for k in a) / len(a)


class PairFeatureBuilder:
    def __init__(self, vectorizer: TfidfVectorizer):
        self.vec = vectorizer

    def build(self, feats: list, pairs: list[tuple[int, int]] | None = None) -> pd.DataFrame:
        """`feats` is a list of FileFeatures (None allowed only if no pair uses it)."""
        if pairs is None:
            pairs = list(itertools.combinations(range(len(feats)), 2))
        X = self.vec.transform([" ".join(f.tokens) if f else "" for f in feats]).tocsr()
        ng3 = [ngrams(f.tokens, 3) if f else set() for f in feats]
        ng5 = [ngrams(f.tokens, 5) if f else set() for f in feats]
        rawl = [[l.strip() for l in f.raw.splitlines() if l.strip()] if f else [] for f in feats]

        rows = []
        for i, j in pairs:
            a, b = feats[i], feats[j]
            la, lb = a.complexity["lines"], b.complexity["lines"]
            rows.append({
                "file_a": a.name, "file_b": b.name,
                "tfidf_cosine": float(X[i].multiply(X[j]).sum()),
                "ngram3_jaccard": jaccard(ng3[i], ng3[j]),
                "ngram5_jaccard": jaccard(ng5[i], ng5[j]),
                "ast_bag_sim": wjaccard(a.subtree, b.subtree),
                "ast_seq_ratio": seq_ratio(a.ast_seq, b.ast_seq),
                "ast_freq_sim": wjaccard(a.ast_counts, b.ast_counts),
                "control_flow_sim": wjaccard(a.control, b.control),
                "behavior_sim": seq_ratio(a.behavior, b.behavior),
                "complexity_diff": complexity_diff(a.complexity, b.complexity),
                "rare_overlap": jaccard(a.rare, b.rare),
                "length_ratio": min(la, lb) / max(la, lb, 1),
                "raw_ident_jaccard": jaccard(a.raw_idents, b.raw_idents),
                "raw_text_ratio": seq_ratio(rawl[i], rawl[j]),
            })
        return pd.DataFrame(rows)
```

### Phase 6: Training and evaluation

**Training data you must prepare (about 1 hour of team time):**

```
data/raw_submissions/
  factorial/   sol_a.py sol_b.py sol_c.py sol_d.py   ← independent solutions
  max_in_list/ sol_a.py ... 
  palindrome/  ...
  (aim for 8+ problems × 4+ independent solutions)
```

Suggested problems: factorial, max in list, palindrome, Fibonacci, prime check, bubble sort, GCD, word count, digit sum, anagram check. Get solutions from teammates writing independently, or from permitted open datasets (check licenses). **Never use real student work without approval.**

**Splitting:** by *original family* (default) or by whole *problem* (`--split-by problem`, stricter). Pairs are only formed *inside* each split, so no leakage. The calibration split is separate from the test split.

**Evaluation includes:**
- precision / recall / F1 / PR-AUC / ROC-AUC / confusion matrix
- baselines at an **equal false-positive budget** (5% FPR chosen on the calibration split)
- per-mutation detection rate (the robustness chart)
- false-positive rate on **hard negatives** vs easy negatives
- runtime per pair
- optional **real-disguise holdout** (`data/holdout/pairs.csv` with columns `a,b,label`, files sitting beside it)

```python
# file: src/train_model.py
"""Usage:
    python -m src.train_model --data data/raw_submissions --holdout data/holdout --variants 6
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)

from .ast_features import extract_features
from .mutation_engine import MUTATIONS, mutate, partial_copy
from .pair_features import MODEL_FEATURES, PairFeatureBuilder, make_vectorizer

FPR_BUDGET = 0.05


# ------------------------------------------------------------ dataset build
def load_originals(root):
    return [{"problem": p.parent.name, "family": f"{p.parent.name}/{p.stem}",
             "src": p.read_text(encoding="utf-8")}
            for p in sorted(Path(root).glob("*/*.py"))]


def build_files(originals, variants, rng):
    files = []
    for o in originals:
        files.append({**o, "kind": "original", "mutations": []})
        made = tries = 0
        while made < variants and tries < variants * 6:
            tries += 1
            src, muts = mutate(o["src"], rng)
            if muts:
                files.append({**o, "src": src, "kind": "variant", "mutations": muts})
                made += 1
        others = [x for x in originals if x["problem"] != o["problem"]]
        if others:
            files.append({**o, "kind": "partial", "mutations": ["partial_copy"],
                          "src": partial_copy(o["src"], rng.choice(others)["src"], rng)})
    return files


def assign_splits(files, key, rng):
    groups = sorted({f[key] for f in files})
    if len(groups) < 5:
        raise SystemExit(f"Need at least 5 distinct {key} groups to make train/cal/test splits.")
    rng.shuffle(groups)
    n = len(groups)
    a, b = max(1, int(n * 0.6)), max(2, int(n * 0.8))
    return {g: ("train" if i < a else "cal" if i < b else "test") for i, g in enumerate(groups)}


def sample_pairs(files, idxs, rng, max_pos=10, neg_ratio=2.0):
    by_fam = defaultdict(list)
    for i in idxs:
        by_fam[files[i]["family"]].append(i)
    pos = []
    for ids in by_fam.values():
        cand = list(itertools.combinations(ids, 2))
        rng.shuffle(cand)
        pos += [(a, b, 1, "positive") for a, b in cand[:max_pos]]

    clean = [i for i in idxs if files[i]["kind"] != "partial"]
    hard, easy = [], []
    for a, b in itertools.combinations(clean, 2):
        if files[a]["family"] == files[b]["family"]:
            continue
        (hard if files[a]["problem"] == files[b]["problem"] else easy).append((a, b))
    rng.shuffle(hard)
    rng.shuffle(easy)
    n_neg = int(len(pos) * neg_ratio)
    n_hard = min(len(hard), int(n_neg * 2 / 3))
    neg = ([(a, b, 0, "hard_neg") for a, b in hard[:n_hard]] +
           [(a, b, 0, "easy_neg") for a, b in easy[: max(0, n_neg - n_hard)]])
    return pos + neg


# --------------------------------------------------------------- evaluation
def best_f1(y, s):
    p, r, _ = precision_recall_curve(y, s)
    f = 2 * p * r / np.clip(p + r, 1e-9, None)
    return float(f.max())


def block(y, p, thr):
    yhat = (p >= thr).astype(int)
    return {"threshold": thr, "precision": float(precision_score(y, yhat, zero_division=0)),
            "recall": float(recall_score(y, yhat, zero_division=0)),
            "f1": float(f1_score(y, yhat, zero_division=0)),
            "pr_auc": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)),
            "confusion_matrix": confusion_matrix(y, yhat, labels=[0, 1]).tolist()}


def evaluate(df_ca, p_ca, df_te, p_te):
    y_ca, y_te = df_ca["label"].to_numpy(), df_te["label"].to_numpy()
    scorers = {
        "CodeShield (Random Forest)": (p_ca, p_te),
        "Raw text diff (difflib)": (df_ca["raw_text_ratio"].to_numpy(), df_te["raw_text_ratio"].to_numpy()),
        "TF-IDF cosine only": (df_ca["tfidf_cosine"].to_numpy(), df_te["tfidf_cosine"].to_numpy()),
        "AST subtree overlap only": (df_ca["ast_bag_sim"].to_numpy(), df_te["ast_bag_sim"].to_numpy()),
    }
    mean3 = lambda d: d[["tfidf_cosine", "ast_bag_sim", "ast_seq_ratio"]].mean(axis=1).to_numpy()
    scorers["Simple mean of 3 signals"] = (mean3(df_ca), mean3(df_te))

    baselines, per_mut = [], []
    mut_sets = df_te["mutations"].fillna("").map(lambda s: set(filter(None, s.split("|"))))
    for name, (s_ca, s_te) in scorers.items():
        baselines.append({"method": name, "pr_auc": float(average_precision_score(y_te, s_te)),
                          "best_f1": best_f1(y_te, s_te)})
        thr = float(np.quantile(s_ca[y_ca == 0], 1 - FPR_BUDGET))    # equal FPR budget
        det = s_te > thr
        for mut in list(MUTATIONS) + ["partial_copy"]:
            mask = (y_te == 1) & mut_sets.map(lambda s: mut in s).to_numpy()
            if mask.sum():
                per_mut.append({"method": name, "mutation": mut,
                                "detection_rate": float(det[mask].mean()), "n": int(mask.sum())})

    neg_types = {}
    for t in ("hard_neg", "easy_neg"):
        m = (df_te["pair_type"] == t).to_numpy()
        if m.sum():
            neg_types[t] = float((p_te[m] >= 0.7).mean())        # share wrongly rated High+
    return {"test_at_0.5": block(y_te, p_te, 0.5), "test_at_0.7": block(y_te, p_te, 0.7),
            "fpr_budget": FPR_BUDGET, "baselines": baselines, "per_mutation": per_mut,
            "false_alarm_rate_by_negative_type": neg_types}


def eval_holdout(folder, builder, cal):
    folder = Path(folder)
    if not (folder / "pairs.csv").exists():
        return None
    pairs = pd.read_csv(folder / "pairs.csv")
    names = sorted(set(pairs["a"]) | set(pairs["b"]))
    feats, index = [], {}
    for n in names:
        f, err = extract_features(n, (folder / n).read_text(encoding="utf-8"))
        if f:
            index[n] = len(feats)
            feats.append(f)
    idx_pairs, labels = [], []
    for r in pairs.itertuples():
        if r.a in index and r.b in index:
            idx_pairs.append((index[r.a], index[r.b]))
            labels.append(int(r.label))
    if not idx_pairs or len(set(labels)) < 2:
        return None
    p = cal.predict_proba(builder.build(feats, idx_pairs)[MODEL_FEATURES])[:, 1]
    return {"n_pairs": len(labels), **block(np.array(labels), p, 0.5)}


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw_submissions")
    ap.add_argument("--holdout", default="data/holdout")
    ap.add_argument("--out", default="models")
    ap.add_argument("--variants", type=int, default=6)
    ap.add_argument("--split-by", choices=["family", "problem"], default="family")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    originals = load_originals(args.data)
    if not originals:
        raise SystemExit(f"No training programs found in {args.data}/<problem>/*.py")
    files = build_files(originals, args.variants, rng)
    split_of = assign_splits(files, args.split_by, rng)

    feats = []
    for i, f in enumerate(files):
        ff, _ = extract_features(f"{f['family']}#{i}", f["src"])
        feats.append(ff)

    vec = make_vectorizer().fit([" ".join(f.tokens) for f, o in zip(feats, files)
                                 if f and split_of[o[args.split_by]] == "train"])
    builder = PairFeatureBuilder(vec)

    frames, timing = {}, 0.0
    for split in ("train", "cal", "test"):
        idxs = [i for i, f in enumerate(files) if split_of[f[args.split_by]] == split and feats[i]]
        pairs = sample_pairs(files, idxs, rng)
        t0 = time.perf_counter()
        df = builder.build(feats, [(a, b) for a, b, *_ in pairs])
        if split == "test":
            timing = (time.perf_counter() - t0) / max(len(pairs), 1) * 1000
        df["label"] = [p[2] for p in pairs]
        df["pair_type"] = [p[3] for p in pairs]
        df["mutations"] = ["|".join(sorted(set(files[a]["mutations"]) | set(files[b]["mutations"])))
                           for a, b, *_ in pairs]
        df["split"] = split
        frames[split] = df
        print(f"{split}: {len(df)} pairs ({df['label'].sum()} positive)")

    Path("data/generated_pairs").mkdir(parents=True, exist_ok=True)
    pd.concat(frames.values()).to_csv("data/generated_pairs/pair_features.csv", index=False)

    tr, ca, te = frames["train"], frames["cal"], frames["test"]
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced",
                                random_state=args.seed, n_jobs=-1)
    rf.fit(tr[MODEL_FEATURES], tr["label"])
    method = "isotonic" if len(ca) >= 1000 else "sigmoid"
    cal = CalibratedClassifierCV(FrozenEstimator(rf), method=method)
    cal.fit(ca[MODEL_FEATURES], ca["label"])

    p_ca = cal.predict_proba(ca[MODEL_FEATURES])[:, 1]
    p_te = cal.predict_proba(te[MODEL_FEATURES])[:, 1]
    metrics = evaluate(ca, p_ca, te, p_te)
    metrics.update({
        "n_pairs": {k: len(v) for k, v in frames.items()},
        "split_by": args.split_by, "calibration": method,
        "runtime_ms_per_pair": round(timing, 3),
        "feature_importance": dict(zip(MODEL_FEATURES, map(float, rf.feature_importances_))),
        "holdout": eval_holdout(args.holdout, builder, cal),
    })

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    joblib.dump(rf, out / "random_forest.joblib")
    joblib.dump(cal, out / "calibrated_model.joblib")
    joblib.dump(vec, out / "tfidf_vectorizer.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    t = metrics["test_at_0.5"]
    print(f"\nTEST @0.5  precision={t['precision']:.3f} recall={t['recall']:.3f} "
          f"f1={t['f1']:.3f} PR-AUC={t['pr_auc']:.3f}")
    for b in metrics["baselines"]:
        print(f"  {b['method']:<30} PR-AUC={b['pr_auc']:.3f}  best-F1={b['best_f1']:.3f}")
    if metrics["holdout"]:
        print(f"HOLDOUT F1={metrics['holdout']['f1']:.3f} (n={metrics['holdout']['n_pairs']})")
    print(f"\nSaved model files to {out}/")


if __name__ == "__main__":
    main()
```

```bash
python -m src.train_model --data data/raw_submissions --holdout data/holdout --variants 6
```

**Checkpoint 6:** the run prints per-split pair counts and a baseline table. **If Random Forest does not beat every baseline on PR-AUC, fix that before building the UI** (more problems, more variants, check the canonicalizer, look at `pair_features.csv`).

### Phase 7: Explanations, groups, prediction

```python
# file: src/explain.py
from __future__ import annotations

import difflib
import html

import numpy as np
import pandas as pd

from .pair_features import MODEL_FEATURES

FEATURE_TEXT = {
    "tfidf_cosine": "Normalised-token TF-IDF cosine",
    "ngram3_jaccard": "Shared 3-token sequences",
    "ngram5_jaccard": "Shared 5-token sequences",
    "ast_bag_sim": "Shared AST subtrees",
    "ast_seq_ratio": "AST node-order similarity",
    "ast_freq_sim": "AST node-type frequency similarity",
    "control_flow_sim": "Loop / branch profile similarity",
    "behavior_sim": "Behavioural fingerprint similarity (static)",
    "complexity_diff": "Complexity difference (lower = more alike)",
    "rare_overlap": "Shared rare constants / library calls",
    "length_ratio": "Length ratio",
    "raw_ident_jaccard": "Original identifier-name overlap",
}


def shap_values(rf, X: pd.DataFrame) -> pd.DataFrame:
    """Per-pair SHAP values for the class-1 probability of the (uncalibrated) forest.
    Calibration is a monotone map, so direction and ranking of reasons are unchanged."""
    try:
        import shap
        sv = shap.TreeExplainer(rf).shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
    except Exception:                       # crude fallback if shap is unavailable
        sv = (X.values - X.values.mean(axis=0)) * rf.feature_importances_
    return pd.DataFrame(sv, columns=X.columns, index=X.index)


def top_reasons(shap_row: pd.Series, x_row: pd.Series, k: int = 4) -> list[str]:
    order = shap_row.abs().sort_values(ascending=False).index[:k]
    return [f"{FEATURE_TEXT[f]} = {x_row[f]:.2f} — "
            f"{'raises' if shap_row[f] > 0 else 'lowers'} the risk ({shap_row[f]:+.2f})"
            for f in order]


def infer_disguises(r) -> list[str]:
    """Heuristic (tune these thresholds on your mutation logs)."""
    out = []
    structural = r["ast_bag_sim"] >= 0.7
    if structural and r["raw_ident_jaccard"] < 0.5:
        out.append("identifier renaming")
    if structural and r["raw_text_ratio"] < 0.7:
        out.append("formatting / comment / statement-level rewrites")
    if structural and 0.02 < r["complexity_diff"] < 0.25 and r["length_ratio"] < 0.97:
        out.append("dead-code or temporary-variable insertion")
    if structural and r["ast_bag_sim"] - r["ast_seq_ratio"] > 0.15:
        out.append("function / statement reordering or extraction")
    return out


def matched_blocks(a: str, b: str, min_lines: int = 2):
    """Line indices of shared runs in the normalised code (a simple stand-in for winnowing)."""
    la, lb = a.splitlines(), b.splitlines()
    ma, mb = set(), set()
    for blk in difflib.SequenceMatcher(None, la, lb, autojunk=False).get_matching_blocks():
        if blk.size >= min_lines:
            ma.update(range(blk.a, blk.a + blk.size))
            mb.update(range(blk.b, blk.b + blk.size))
    return ma, mb


def render_code(code: str, matched: set) -> str:
    rows = []
    for i, ln in enumerate(code.splitlines()):
        bg = "background:rgba(255,200,60,0.35);" if i in matched else ""
        rows.append(f'<div style="{bg}white-space:pre;font-family:monospace;font-size:13px;'
                    f'padding:0 6px">{html.escape(ln) or "&nbsp;"}</div>')
    return "".join(rows)
```

```python
# file: src/graph_analysis.py
from __future__ import annotations

import networkx as nx
import plotly.graph_objects as go


def build_graph(pairs, threshold: float) -> nx.Graph:
    G = nx.Graph()
    for r in pairs.itertuples():
        if r.probability >= threshold:
            G.add_edge(r.file_a, r.file_b, weight=float(r.probability))
    return G


def analyse_groups(G: nx.Graph, n_students: int, convergent_share: float = 0.25) -> list[dict]:
    if G.number_of_edges() == 0:
        return []
    out = []
    for c in nx.community.louvain_communities(G, weight="weight", seed=42):
        if len(c) < 2:
            continue
        sub = G.subgraph(c)
        w = [d["weight"] for *_, d in sub.edges(data=True)]
        if not w:
            continue
        dens = nx.density(sub)
        if len(c) >= 4 and len(c) / n_students >= convergent_share:
            kind = "Convergent cluster (possible shared source or tutorial solution)"
        elif len(c) >= 3 and dens >= 0.8:
            kind = "Tight ring (every pair highly similar)"
        elif len(c) == 2:
            kind = "Pair"
        else:
            kind = "Chain / partial group"
        out.append({"members": sorted(c), "size": len(c), "density": round(dens, 2),
                    "mean_probability": round(sum(w) / len(w), 2), "type": kind})
    return sorted(out, key=lambda g: (-g["mean_probability"], -g["size"]))


def graph_figure(G: nx.Graph, groups: list[dict]) -> go.Figure:
    pos = nx.spring_layout(G, seed=42, weight="weight")
    member_group = {m: i + 1 for i, g in enumerate(groups) for m in g["members"]}
    ex, ey = [], []
    for u, v in G.edges():
        ex += [pos[u][0], pos[v][0], None]
        ey += [pos[u][1], pos[v][1], None]
    nodes = list(G.nodes())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", hoverinfo="none", showlegend=False,
                             line=dict(width=1.5, color="#999")))
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text",
        text=nodes, textposition="top center", showlegend=False,
        hovertext=[f"{n}: {G.degree(n)} link(s)" for n in nodes], hoverinfo="text",
        marker=dict(size=[16 + 6 * G.degree(n) for n in nodes],
                    color=[member_group.get(n, 0) for n in nodes], colorscale="Turbo")))
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False),
                      margin=dict(l=0, r=0, t=10, b=0), height=520)
    return fig
```

```python
# file: src/predict.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from .ast_features import extract_features
from .explain import infer_disguises, shap_values, top_reasons
from .pair_features import MODEL_FEATURES, PairFeatureBuilder

LABEL_EDGES = [0.0, 0.40, 0.70, 0.90, 1.0001]
LABELS = ["Low", "Review", "High", "Very high"]


@dataclass
class Analysis:
    pairs: pd.DataFrame       # one row per pair, sorted by probability
    shap: pd.DataFrame        # aligned with `pairs` (same index)
    files: dict               # name -> FileFeatures
    errors: list              # [(name, message)]


def analyze(subs, template: str | None = None, model_dir: str = "models") -> Analysis:
    md = Path(model_dir)
    try:
        cal = joblib.load(md / "calibrated_model.joblib")
        rf = joblib.load(md / "random_forest.joblib")
        vec = joblib.load(md / "tfidf_vectorizer.joblib")
    except FileNotFoundError as e:
        raise FileNotFoundError("Model files not found. Train first: python -m src.train_model") from e

    feats, errors, seen = [], [], {}
    for s in subs:
        name = s.name
        if name in seen:                       # keep names unique
            seen[name] += 1
            name = f"{Path(name).stem}_{seen[name]}.py"
        else:
            seen[name] = 1
        f, err = extract_features(name, s.raw, template)
        if err:
            errors.append((name, err))
        else:
            feats.append(f)
    if len(feats) < 2:
        raise ValueError("Need at least two valid Python files to compare.")

    df = PairFeatureBuilder(vec).build(feats)
    X = df[MODEL_FEATURES]
    df["probability"] = cal.predict_proba(X)[:, 1]
    df["label"] = pd.cut(df["probability"], LABEL_EDGES, labels=LABELS, right=False)

    # Class-relative context: where does this pair sit among ALL pairs in this batch?
    # Shown to the instructor but NOT fed to the model (class prevalence differs from training).
    composite = df[["tfidf_cosine", "ast_bag_sim", "ast_seq_ratio"]].mean(axis=1)
    df["top_pct"] = composite.rank(ascending=False, method="min") / len(df) * 100

    sv = shap_values(rf, X)
    df["reasons"] = [top_reasons(sv.loc[i], X.loc[i]) for i in df.index]
    df["disguises"] = [infer_disguises(r) for _, r in df.iterrows()]

    order = df["probability"].sort_values(ascending=False).index
    df = df.loc[order].reset_index(drop=True)
    sv = sv.loc[order].reset_index(drop=True)
    return Analysis(df, sv, {f.name: f for f in feats}, errors)


def log_feedback(a: str, b: str, prob: float, verdict: str, path: str = "data/feedback.csv"):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{"ts": datetime.now().isoformat(timespec="seconds"),
                         "file_a": a, "file_b": b, "probability": prob, "verdict": verdict}])
    row.to_csv(p, mode="a", header=not p.exists(), index=False)
```

**Checkpoint 7:**

```bash
python - <<'EOF'
from src.preprocessing import load_folder
from src.predict import analyze
res = analyze(load_folder("data/demo_submissions"))
print(res.pairs[["file_a", "file_b", "probability", "label", "top_pct"]].head())
print(res.pairs.loc[0, "reasons"], res.pairs.loc[0, "disguises"])
EOF
```

### Phase 8: Streamlit dashboard (Day 2 morning)

```python
# file: app.py
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.explain import matched_blocks, render_code
from src.graph_analysis import analyse_groups, build_graph, graph_figure
from src.predict import analyze, log_feedback
from src.preprocessing import Submission, load_zip, pseudonymise

st.set_page_config(page_title="CodeShield AI", page_icon="🛡️", layout="wide")
st.title("🛡️ CodeShield AI")
st.caption("Instructor-assistance tool. A score is a reason to look, never proof of misconduct.")

ss = st.session_state
tabs = st.tabs(["Upload", "Overview", "Suspicious pairs", "Pair evidence", "Group graph", "Evaluation"])

# ------------------------------------------------------------------ Upload
with tabs[0]:
    files = st.file_uploader("Student submissions (.zip or .py files)", type=["zip", "py"],
                             accept_multiple_files=True)
    tpl = st.file_uploader("Starter / template code to ignore (optional)", type=["py"])
    st.slider("Group-graph threshold", 0.40, 0.95, 0.70, 0.05, key="threshold")
    anon = st.checkbox("Pseudonymise file names (recommended)", value=True)
    if st.button("Analyze", type="primary", disabled=not files):
        subs = []
        for f in files:
            data = f.getvalue()
            if f.name.lower().endswith(".zip"):
                subs += load_zip(data)
            else:
                subs.append(Submission(f.name, data.decode("utf-8", errors="replace")))
        if anon:
            subs, ss["name_map"] = pseudonymise(subs)
        try:
            with st.spinner("Analysing submissions..."):
                ss["result"] = analyze(subs, tpl.getvalue().decode("utf-8", "replace") if tpl else None)
            st.success("Done. Open the other tabs.")
        except (ValueError, FileNotFoundError) as e:
            st.error(str(e))
    res = ss.get("result")
    if res and res.errors:
        st.warning("Skipped files:\n" + "\n".join(f"- {n}: {e}" for n, e in res.errors))

res = ss.get("result")
NEED = "Run an analysis on the Upload tab first."

# ---------------------------------------------------------------- Overview
with tabs[1]:
    if res is None:
        st.info(NEED)
    else:
        p = res.pairs
        c = st.columns(4)
        c[0].metric("Submissions", len(res.files))
        c[1].metric("Pairs compared", len(p))
        c[2].metric("High or Very high", int((p["probability"] >= 0.70).sum()))
        c[3].metric("Review or above", int((p["probability"] >= 0.40).sum()))
        st.plotly_chart(px.histogram(p, x="probability", nbins=20, title="Score distribution"))

# -------------------------------------------------------- Suspicious pairs
with tabs[2]:
    if res is None:
        st.info(NEED)
    else:
        min_p = st.slider("Minimum probability", 0.0, 1.0, 0.40, 0.05)
        view = res.pairs[res.pairs["probability"] >= min_p].copy()
        view["main_reason"] = view["reasons"].map(lambda r: r[0] if r else "")
        view["likely_disguises"] = view["disguises"].map(", ".join)
        cols = ["file_a", "file_b", "probability", "label", "top_pct", "ast_bag_sim",
                "tfidf_cosine", "behavior_sim", "likely_disguises", "main_reason"]
        st.dataframe(view[cols].round(3))
        st.download_button("Download CSV report", view[cols].round(3).to_csv(index=False),
                           "codeshield_report.csv", "text/csv")

# ------------------------------------------------------------ Pair evidence
with tabs[3]:
    if res is None:
        st.info(NEED)
    else:
        labels = [f"{r.file_a} ↔ {r.file_b}  ({r.probability:.2f})" for r in res.pairs.itertuples()]
        pick = st.selectbox("Pair", labels)
        i = labels.index(pick)
        row = res.pairs.iloc[i]
        c = st.columns(3)
        c[0].metric("Suspicious-similarity probability", f"{row.probability:.2f}")
        c[1].metric("Risk category", str(row.label))
        c[2].metric("Similarity rank in this class", f"top {row.top_pct:.1f}% of pairs")
        st.markdown("**Why this pair was flagged**")
        for r in row.reasons:
            st.write("• " + r)
        st.markdown("**Likely disguise techniques (heuristic):** " +
                    (", ".join(row.disguises) or "none detected"))
        a, b = res.files[row.file_a], res.files[row.file_b]
        ma, mb = matched_blocks(a.normalized, b.normalized)
        st.markdown("**Normalised code, matching blocks highlighted**")
        c1, c2 = st.columns(2)
        c1.caption(row.file_a)
        c1.html(render_code(a.normalized, ma))
        c2.caption(row.file_b)
        c2.html(render_code(b.normalized, mb))
        with st.expander("Original code"):
            o1, o2 = st.columns(2)
            o1.code(a.raw, language="python")
            o2.code(b.raw, language="python")
        s = res.shap.iloc[i].sort_values()
        st.plotly_chart(px.bar(s, orientation="h", title="Feature contributions (SHAP)",
                               labels={"value": "effect on risk", "index": "feature"}))
        st.info("Instructor review required. This is not an automatic verdict.")
        b1, b2 = st.columns(2)
        for col, verdict in ((b1, "confirmed"), (b2, "dismissed")):
            if col.button(f"Mark {verdict}", key=f"{verdict}-{pick}"):
                log_feedback(row.file_a, row.file_b, float(row.probability), verdict)
                col.success("Saved to data/feedback.csv")

# -------------------------------------------------------------- Group graph
with tabs[4]:
    if res is None:
        st.info(NEED)
    else:
        G = build_graph(res.pairs, ss["threshold"])
        if G.number_of_edges() == 0:
            st.info("No pairs above the threshold.")
        else:
            groups = analyse_groups(G, len(res.files))
            st.plotly_chart(graph_figure(G, groups))
            st.dataframe(pd.DataFrame(groups))

# --------------------------------------------------------------- Evaluation
with tabs[5]:
    mp = Path("models/metrics.json")
    if not mp.exists():
        st.info("Train the model first: python -m src.train_model")
    else:
        m = json.loads(mp.read_text())
        t = m["test_at_0.5"]
        c = st.columns(5)
        for col, k in zip(c, ["precision", "recall", "f1", "pr_auc", "roc_auc"]):
            col.metric(k.upper().replace("_", "-"), f"{t[k]:.3f}")
        st.caption(f"Split by {m['split_by']} · calibration: {m['calibration']} · "
                   f"{m['runtime_ms_per_pair']} ms per pair · test pairs: {m['n_pairs']['test']}")
        cm = pd.DataFrame(t["confusion_matrix"], index=["actual independent", "actual suspicious"],
                          columns=["predicted independent", "predicted suspicious"])
        st.plotly_chart(px.imshow(cm, text_auto=True, title="Confusion matrix (threshold 0.5)"))
        st.subheader(f"Detection rate per disguise (all methods at {int(m['fpr_budget'] * 100)}% false-positive budget)")
        st.plotly_chart(px.bar(pd.DataFrame(m["per_mutation"]), x="mutation", y="detection_rate",
                               color="method", barmode="group"))
        st.subheader("Baselines vs Random Forest")
        st.dataframe(pd.DataFrame(m["baselines"]).round(3))
        st.write("Share of independent pairs wrongly rated High or above:",
                 m["false_alarm_rate_by_negative_type"])
        fi = pd.Series(m["feature_importance"]).sort_values()
        st.plotly_chart(px.bar(fi, orientation="h", title="Global feature importance"))
        if m.get("holdout"):
            h = m["holdout"]
            st.subheader("Real-disguise holdout (never seen by the mutation engine)")
            st.write({k: round(v, 3) for k, v in h.items() if isinstance(v, float)}, f"n = {h['n_pairs']}")
```

```bash
streamlit run app.py
```

### Phase 9: Tests

```python
# file: tests/test_smoke.py
import ast
import random

from src.ast_features import extract_features
from src.mutation_engine import MUTATIONS, mutate
from src.pair_features import PairFeatureBuilder, make_vectorizer

A = '''
def biggest(nums):
    best = nums[0]
    for n in nums:
        if n > best:
            best = n
    return best

data = [3, 9, 2]
print(biggest(data))
'''

B = '''
values = [4, 1, 7]
values.sort()
print(values[-1])
'''


def test_normalization_ignores_names_and_comments():
    a, _ = extract_features("a", "def f(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n")
    b, _ = extract_features("b", "def g(items):\n    acc = 0  # start\n    for it in items:\n        acc = acc + it\n    return acc\n")
    assert a.normalized == b.normalized


def test_syntax_error_is_reported_not_raised():
    f, err = extract_features("bad", "def broken(:\n")
    assert f is None and "Syntax error" in err


def test_every_mutation_yields_valid_python():
    rng = random.Random(1)
    for _ in range(30):
        out, muts = mutate(A, rng)
        ast.parse(out)


def test_disguised_copy_scores_above_independent_solution():
    fa, _ = extract_features("a", A)
    fb, _ = extract_features("b", B)
    vec = make_vectorizer().fit([" ".join(fa.tokens), " ".join(fb.tokens)])
    rng, feats = random.Random(3), [fa, fb]
    for i in range(8):
        src, _ = mutate(A, rng)
        f, _ = extract_features(f"m{i}", src)
        feats.append(f)
    df = PairFeatureBuilder(vec).build(feats, [(0, k) for k in range(2, len(feats))] + [(0, 1)])
    assert df["ast_bag_sim"].iloc[:-1].mean() > df["ast_bag_sim"].iloc[-1]
    assert df["tfidf_cosine"].iloc[:-1].mean() > df["tfidf_cosine"].iloc[-1]


def test_all_mutations_are_registered():
    assert {"rename", "dead_code", "temp_variable", "wrap_main"} <= set(MUTATIONS)
```

```bash
pytest -q
```

---

## 5. Two-day schedule

| When | Do | Done when |
|---|---|---|
| **Day 1 morning** | Repo, `preprocessing.py`, `ast_features.py`, `behavioral_features.py` | Checkpoint 3 prints `True` |
| **Day 1 afternoon** | `pair_features.py`; **write the training solutions and 10–20 real-disguise holdout files by hand** (parallelize this across teammates) | CSV of pair features; baseline ranking works |
| **Day 1 evening** | `mutation_engine.py`, `train_model.py`, first training run | RF beats baselines; `metrics.json` exists |
| **Day 2 morning** | `explain.py`, `graph_analysis.py`, `predict.py`, `app.py` (tabs 0–3) | Upload → ranked table → evidence screen works |
| **Day 2 afternoon** | Group graph, Evaluation tab, per-mutation chart, CSV export, feedback buttons, polish | Full demo run-through without errors |
| **Only if time remains** | Section 9 extras | Bonus only |

**Team split for 3–4 people:** one on data + mutation engine + evaluation, one on canonicalization + features, one on model + explanations, one on the Streamlit UI and the presentation.

---

## 6. MVP definition of done

- [ ] Upload a ZIP or 5–10 `.py` files
- [ ] Comments and identifiers are normalized; syntax errors are reported clearly
- [ ] Starter-code subtraction works
- [ ] All unique pairs are generated and scored
- [ ] Random Forest trained on mutation-generated pairs, **calibrated**, saved with joblib
- [ ] Results table: probability, label, class percentile, likely disguises, main reason
- [ ] Evidence screen: highlighted matching blocks, SHAP reasons, original code
- [ ] Group graph above the threshold, with cluster types
- [ ] Evaluation tab: metrics, confusion matrix, baselines, per-mutation chart
- [ ] CSV export and instructor confirm/dismiss log
- [ ] `pytest` passes
- [ ] The words "not proof" and "instructor review" appear in the UI

---

## 7. Demo script (about 4 minutes)

1. **Problem (20s):** "Text diff tools fail once someone renames variables or reformats the code."
2. **Live upload (30s):** drop in `data/demo_submissions` (prepare 8–10 files: 2–3 genuine copies with different disguises, a couple of independent solutions to the *same* task, and unrelated files).
3. **Ranked queue (30s):** show the top pairs, probabilities and labels.
4. **Evidence (60s):** open the top pair: highlighted matching blocks, SHAP reasons, the "likely disguises" line, and the class percentile. Point out that identifier overlap is low, so it was not just the names.
5. **Fairness moment (30s):** open an independent same-task pair that scored *low* (a hard negative). Say: "Natural similarity is not accusation."
6. **Robustness chart (45s):** the per-mutation bar chart: CodeShield vs difflib vs TF-IDF at the same false-positive budget.
7. **Group graph (20s):** the ring of copiers.
8. **Close (15s):** "It learns which similarities are suspicious for this assignment, tells you which disguise was likely used, and shows the evidence. A human decides."

**Prepare answers for judges:**

| Question | Answer |
|---|---|
| Why ML instead of a threshold? | Baseline table at equal false-positive budget: RF wins on PR-AUC and per-disguise detection. |
| Isn't it just testing on your own mutations? | Yes for the main test, so we also report a hand-written real-disguise holdout separately. |
| Is 0.91 really 91%? | The forest is calibrated on a held-out split; treat it as an estimate, not a guarantee. |
| How do you avoid false accusations? | Hard negatives, class percentile, starter-code removal, evidence for every alert, human decision. |
| Does it run student code? | No. The behavioral fingerprint is static and labelled that way. |
| Privacy? | Pseudonymous IDs, in-memory ZIP reading, no retained uploads. |

---

## 8. Responsible-AI checklist

- Use "suspicious similarity" and "requires review", never "plagiarism detected".
- **Do not execute uploaded code.** If you ever add behavioral execution, use a locked-down container with time, memory, file and network limits.
- Show evidence for every alert; log confirm/dismiss decisions.
- Pseudonymize by default; do not persist uploads; keep the name mapping in memory only.
- Report false-positive rates, especially on short beginner assignments, and say so on the Evaluation tab.
- Never train on real student submissions without approval.
- Convergent clusters (many students with the same solution) may indicate a shared tutorial or AI-style answer, not copying between them. The graph labels them separately.

---

## 9. Extras, in priority order (only after Section 6 is done)

**Tier 1**
1. **Winnowing / Greedy String Tiling** for better matched-block highlighting (the current version uses line matching on normalized code).
2. **Per-pair HTML/PDF case report** with scores, highlights and SHAP reasons.
3. **Threshold tuning UI** using the instructor's confirm/dismiss log in `data/feedback.csv`; retrain a lightweight calibrator from it.

**Tier 2**
4. **Scalability prefilter:** TF-IDF nearest neighbors or MinHash/LSH to drop clearly unrelated pairs (500 students ≈ 125k pairs).
5. **Multi-label disguise model** trained on the mutation logs, replacing the heuristic `infer_disguises`.
6. **Grounded LLM summary:** give an LLM only the structured feature values and ask for two neutral sentences. It may narrate the numbers, never accuse.
7. **CodeBERT embedding cosine** as a 13th feature (pretrained inference only; retrain the forest afterwards).

**Tier 3 (future work slide)**
8. Optional submission timestamps as a labelled hint; sandboxed behavioral execution; tree-sitter for Java/C++.

---

## 10. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `ImportError: FrozenEstimator` | `pip install -U "scikit-learn>=1.6"` |
| Training says "Need at least 5 distinct family groups" | Add more solutions under `data/raw_submissions/<problem>/` |
| RF is barely better than TF-IDF | Too few problems or variants; inspect `data/generated_pairs/pair_features.csv`; check that the canonicalizer collapses your mutations (Checkpoint 3) |
| Everything scores ≥ 0.9 in the demo | Assignments are tiny and near-identical: add hard negatives, check the class percentile, use starter-code subtraction |
| Renamed copy scores low | Rename touched a keyword argument or an attribute; inspect `extract_features(...).normalized` of both files |
| SHAP shape error | The code handles list and 3-D outputs; if it still fails, the fallback runs automatically, and you should pin `shap>=0.46` |
| Streamlit `st.html` missing | Upgrade Streamlit (`>=1.40`) |
| Mutated code changes behavior | Expected occasionally (mutations are validated by parsing, not execution). Remove that mutation or tighten its guard |

---

## 11. Pitch line

> "Existing tools compare text or trees. CodeShield AI **learns** which similarities are suspicious for this assignment, tells you **which disguise was likely used**, and shows the **matching evidence**. The teacher decides."

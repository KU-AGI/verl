# import os, json

# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("/data/llm-reaction-reasoning/all_checkpoints/reflection_v4_fullft_all/best.ckpt/hf_model")



# smiles_map = {
#     "forward": "{reactants}.{reagents}",
#     "retro": "{products}",
#     "reagent": "{reactants}>>{products}",
# }

# tasks= ["forward", "retro", "reagent"]

# for task in tasks:
#     test_path = os.path.join("/data/llm-reaction-reasoning/data/orderly/excluded_test", f"excluded_{task}_test_v10_required.jsonl")
#     val_path = os.path.join("/data/llm-reaction-reasoning/data/orderly/balanced_val", f"balanced_{task}_val_v10_required.jsonl")
#     with open(test_path, "r") as f:
#         test_data = [json.loads(line) for line in f.readlines()]
#     with open(val_path, "r") as f:
#         val_data = [json.loads(line) for line in f.readlines()]

#     for test_entry in test_data:
#         reactants_str = ".".join(test_entry["reactants"])
#         reagents_str = ".".join(test_entry["reagents"])
#         products_str = ".".join(test_entry["products"])
#         input_smiles = smiles_map[task].format(
#             reactants=reactants_str,
#             reagents=reagents_str,
#             products=products_str,
#         )
#         input_ids = tokenizer.encode(input_smiles)



import os, json, math
from collections import Counter
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer

# 파일 상단 임포트 근처에 추가
try:
    from adjustText import adjust_text
    _HAS_ADJUSTTEXT = True
except Exception:
    _HAS_ADJUSTTEXT = False

# =========================
# 설정
# =========================
HF_MODEL_DIR = "/data/llm-reaction-reasoning/all_checkpoints/reflection_v4_fullft_all/best.ckpt/hf_model"
DATA_ROOT_TEST = "/data/llm-reaction-reasoning/data/orderly/excluded_test"
DATA_ROOT_VAL  = "/data/llm-reaction-reasoning/data/orderly/balanced_val"
OUT_DIR = "./Analysis/ngram_token_analysis"   # 결과 저장 디렉토리
N_LIST = list(range(1, 21))                   # 보고 싶은 n-gram
TOPK_PLOT = 20                       # 막대그래프에 표시할 상위 항목 수
SCATTER_LIMIT = 2000                 # 산점도에 표시할 최대 포인트 수(차이 큰 순)

os.makedirs(OUT_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_DIR)

smiles_map = {
    "forward": "{reactants}.{reagents}",
    "retro": "{products}",
    "reagent": "{reactants}>>{products}",
}
tasks = ["forward", "retro", "reagent"]

def iter_entries(path: str) -> List[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def build_input_smiles(entry: dict, task: str) -> str:
    reactants_str = ".".join(entry.get("reactants", []))
    reagents_str  = ".".join(entry.get("reagents", []))
    products_str  = ".".join(entry.get("products", []))
    return smiles_map[task].format(
        reactants=reactants_str,
        reagents=reagents_str,
        products=products_str,
    )

def encode_tokens(s: str) -> List[int]:
    return tokenizer.encode(s, add_special_tokens=False)

def ngrams(seq: List[int], n: int) -> List[Tuple[int, ...]]:
    if len(seq) < n:
        return []
    return [tuple(seq[i:i+n]) for i in range(len(seq) - n + 1)]

def decode_ngram(ng: Tuple[int, ...]) -> str:
    return tokenizer.decode(list(ng), clean_up_tokenization_spaces=False, skip_special_tokens=True)

def normalized_freq(counter: Counter) -> Dict[Tuple[int, ...], float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counter.items()}

def kl_divergence(p: Dict, q: Dict, eps=1e-12) -> float:
    keys = set(p) | set(q)
    kl = 0.0
    for k in keys:
        pk = p.get(k, 0.0) + eps
        qk = q.get(k, 0.0) + eps
        kl += pk * math.log(pk / qk)
    return kl

def plot_grouped_bars(labels, test_vals, val_vals, title, out_png):
    x = list(range(len(labels)))
    width = 0.45
    plt.figure(figsize=(max(8, len(labels)*0.5), 5))
    plt.bar([i - width/2 for i in x], test_vals, width=width, label="test")
    plt.bar([i + width/2 for i in x], val_vals, width=width, label="val")
    plt.xticks(x, labels, rotation=60, ha='right')
    plt.ylabel("Normalized frequency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def scatter_freq(all_labels, test_vals, val_vals, title, out_png):
    plt.figure(figsize=(6,6))
    plt.scatter(test_vals, val_vals, s=18, alpha=0.7)
    maxv = max(test_vals + val_vals + [0.05])
    plt.plot([0, maxv], [0, maxv])
    plt.xlabel("test freq")
    plt.ylabel("val freq")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_all_n_bar_subplots(task: str, n_to_topk_df: Dict[int, pd.DataFrame], out_png: str):
    rows = len(n_to_topk_df)
    fig, axes = plt.subplots(rows, 1, figsize=(max(8, TOPK_PLOT*0.5), 4.5*rows), squeeze=False)
    axes = axes.ravel()
    for idx, n in enumerate(sorted(n_to_topk_df.keys())):
        ax = axes[idx]
        df = n_to_topk_df[n]
        labels = df["ngram_text"].tolist()
        test_vals = df["test_freq"].tolist()
        val_vals  = df["val_freq"].tolist()

        x = list(range(len(labels)))
        width = 0.45
        ax.bar([i - width/2 for i in x], test_vals, width=width, label="test")
        ax.bar([i + width/2 for i in x], val_vals, width=width, label="val")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=9)
        ax.set_ylabel("Norm. freq")
        ax.set_title(f"{task} n={n} — Top {len(labels)} by |test−val|")
        if idx == 0:
            ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def plot_all_n_scatter_subplots(task, n_to_scatter_df, out_png: str, annotate_top=5):
    rows = len(n_to_scatter_df)
    fig, axes = plt.subplots(rows, 1, figsize=(6.5, 6*rows), squeeze=False)
    axes = axes.ravel()

    for idx, n in enumerate(sorted(n_to_scatter_df.keys())):
        ax = axes[idx]
        df = n_to_scatter_df[n]
        ax.scatter(df["test_freq"], df["val_freq"], s=18, alpha=0.7)

        # <-- 여기: 각 n의 데이터로 로컬 축 범위 계산
        if not df.empty:
            local_max = max(df["test_freq"].max(), df["val_freq"].max())
        else:
            local_max = 0.05
        pad = max(1e-4, local_max * 0.05)   # 살짝 여백
        hi = local_max + pad

        ax.plot([0, hi], [0, hi])
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)

        ax.set_xlabel("test freq")
        ax.set_ylabel("val freq")
        ax.set_title(f"{task} n={n} — test vs val freq (Top {len(df)} by |diff|)")

        if annotate_top > 0 and not df.empty:
            ann = df.sort_values("abs_diff", ascending=False).head(annotate_top)
            texts = []
            for _, r in ann.iterrows():
                t = ax.text(
                    r["test_freq"], r["val_freq"],
                    r["ngram_text"][:24],
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7)
                )
                texts.append(t)
            # 점과 텍스트를 연결하는 작은 화살표
            if _HAS_ADJUSTTEXT:
                adjust_text(
                    texts, ax=ax, expand_points=(1.1, 1.3),
                    arrowprops=dict(arrowstyle="-", lw=0.5, alpha=0.6)
                )
            else:
                # adjustText가 없으면 간단 스태거 폴백
                offsets = [(6,6), (6,-6), (-6,6), (-6,-6), (8,0), (0,8)]
                for i, t in enumerate(texts):
                    dx, dy = offsets[i % len(offsets)]
                    t.set_position((t.get_position()[0] + 0, t.get_position()[1] + 0))  # anchor 유지
                    t.set_transform(ax.transData)
                    # 화살표는 별도 주석(스태거 안 쓰고 텍스트만 위치 조정)
                # 필요하면 위 1) 방식처럼 annotate로 교체해도 OK
                pass

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


for task in tasks:
    test_path = os.path.join(DATA_ROOT_TEST, f"excluded_{task}_test_v10_required.jsonl")
    val_path  = os.path.join(DATA_ROOT_VAL,  f"balanced_{task}_val_v10_required.jsonl")

    test_data = iter_entries(test_path)
    val_data  = iter_entries(val_path)

    print(f"[{task}] #test={len(test_data)}, #val={len(val_data)}")

    n_to_topk_df = {}
    n_to_scatter_df = {}  # 산점도 서브플롯용 데이터 저장

    for n in N_LIST:
        test_counter: Counter = Counter()
        val_counter: Counter  = Counter()

        for entry in test_data:
            s = build_input_smiles(entry, task)
            ids = encode_tokens(s)
            test_counter.update(ngrams(ids, n))

        for entry in val_data:
            s = build_input_smiles(entry, task)
            ids = encode_tokens(s)
            val_counter.update(ngrams(ids, n))

        test_freq = normalized_freq(test_counter)
        val_freq  = normalized_freq(val_counter)

        kl_tv = kl_divergence(test_freq, val_freq)
        kl_vt = kl_divergence(val_freq, test_freq)
        print(f"  n={n}: KL(test||val)={kl_tv:.6f}, KL(val||test)={kl_vt:.6f}")

        keys = list(set(test_freq) | set(val_freq))
        records = []
        for k in keys:
            records.append({
                "ngram_ids": k,
                "ngram_text": decode_ngram(k),
                "test_freq": test_freq.get(k, 0.0),
                "val_freq": val_freq.get(k, 0.0),
                "abs_diff": abs(test_freq.get(k, 0.0) - val_freq.get(k, 0.0)),
                "test_count": test_counter.get(k, 0),
                "val_count": val_counter.get(k, 0),
            })
        df = pd.DataFrame.from_records(records).sort_values("abs_diff", ascending=False)

        # CSV 저장(개별)
        csv_path = os.path.join(OUT_DIR, f"{task}_n{n}_freqs.csv")
        # df.to_csv(csv_path, index=False)

        # 개별 막대 플롯
        topk = df.head(TOPK_PLOT)
        title = f"{task} n={n} — Top {TOPK_PLOT} n-grams by |test-val|"
        out_png = os.path.join(OUT_DIR, f"{task}_n{n}_top{TOPK_PLOT}_absdiff.png")
        # plot_grouped_bars(
        #     labels=topk["ngram_text"].tolist(),
        #     test_vals=topk["test_freq"].tolist(),
        #     val_vals=topk["val_freq"].tolist(),
        #     title=title,
        #     out_png=out_png
        # )

        # 개별 산점도 (차이 큰 순서 기준 상위 SCATTER_LIMIT개)
        df_sc = df.head(SCATTER_LIMIT).copy()
        scatter_png = os.path.join(OUT_DIR, f"{task}_n{n}_scatter_test_vs_val.png")
        # scatter_freq(
        #     all_labels=df_sc["ngram_text"].tolist(),
        #     test_vals=df_sc["test_freq"].tolist(),
        #     val_vals=df_sc["val_freq"].tolist(),
        #     title=f"{task} n={n} — test vs val freq (Top {len(df_sc)} by |diff|)",
        #     out_png=scatter_png
        # )

        # 서브플롯용 저장
        n_to_topk_df[n] = topk[["ngram_text","test_freq","val_freq","abs_diff"]].copy()
        n_to_scatter_df[n] = df_sc[["ngram_text","test_freq","val_freq","abs_diff"]].copy()

        # 콘솔에 상위 10개
        print(f"  {task} n={n} — most different n-grams (top 10):")
        for _, row in df.head(10).iterrows():
            print(f"    {row['ngram_text']!r} | test={row['test_freq']:.6f} "
                  f"val={row['val_freq']:.6f} diff={row['abs_diff']:.6f} "
                  f"(counts: {int(row['test_count'])}/{int(row['val_count'])})")

    # === 한 장짜리 subplot(막대) ===
    bar_subplot_out = os.path.join(OUT_DIR, f"{task}_all_n_top{TOPK_PLOT}_absdiff_SUBPLOTS.png")
    plot_all_n_bar_subplots(task, n_to_topk_df, bar_subplot_out)
    print(f"  -> saved bar subplots: {bar_subplot_out}")

    # === 한 장짜리 subplot(산점도) ===
    scatter_subplot_out = os.path.join(OUT_DIR, f"{task}_all_n_scatter_TOP{SCATTER_LIMIT}_SUBPLOTS.png")
    plot_all_n_scatter_subplots(task, n_to_scatter_df, scatter_subplot_out, annotate_top=5)
    print(f"  -> saved scatter subplots: {scatter_subplot_out}")

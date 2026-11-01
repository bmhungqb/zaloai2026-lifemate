#!/usr/bin/env python3
"""Profile the OFFICIAL DataMate data kit (downloaded by you from the portal).
Usage: python3 -I analyze_datamate.py <split_dir> [out.md]
  <split_dir> contains questions.jsonl, (answers.jsonl), rulebook.md, calendar.csv, data/<business_id>/...
Nothing is sent anywhere; the report contains aggregate stats only (no raw rows)."""
import sys, os, re, json, sqlite3, collections, unicodedata
import warnings; warnings.filterwarnings("ignore")
import pandas as pd

root = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else "datamate_profile.md"
L = []
def p(s=""): L.append(s)

def jl(path):
    if not os.path.exists(path): return []
    return [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]

Q = jl(os.path.join(root, "questions.jsonl")); A = {a["id"]: a for a in jl(os.path.join(root, "answers.jsonl"))}
p(f"# DataMate profile: {root}\n"); p(f"- questions: {len(Q)}; answers: {len(A)}")
p(f"- businesses in questions: {len(set(q['business_id'] for q in Q))}")
if A:
    t = collections.Counter(a["type"] for a in A.values()); p("\n## Answer types"); [p(f"- {k}: {v} ({v/len(A):.1%})") for k, v in t.most_common()]
    un = sum(1 for a in A.values() if a.get("answer") is None); p(f"- null answers (unanswerable): {un}")
perbiz = collections.Counter(q["business_id"] for q in Q); p("\n## Questions per business"); p(", ".join(f"{k}:{v}" for k, v in sorted(perbiz.items())))
asof = sorted(q["as_of_date"] for q in Q if q.get("as_of_date")); p(f"\n- as_of_date range: {asof[:1]} .. {asof[-1:]}")

# question lexicon: first informative words -> crude template families
def norm(s): return unicodedata.normalize("NFC", s.lower())
kw = ["doanh thu","lợi nhuận","giá vốn","công nợ","quá hạn","tồn kho","top","nhiều nhất","ít nhất","so với","cùng kỳ","tuần","tháng","quý","năm","tết","khách","nhà cung cấp","hủy","hoàn","giờ cao điểm","ca ","%","bao nhiêu","có bao nhiêu","danh sách","liệt kê","dự đoán","nên"]
c = collections.Counter(); 
for q in Q:
    s = norm(q["question"]); [c.update([k]) for k in kw if k in s]
p("\n## Keyword frequency in questions"); [p(f"- `{k}`: {v} ({v/len(Q):.1%})") for k, v in c.most_common()]
L2 = sorted(len(q["question"]) for q in Q); p(f"\n- question length chars: median {L2[len(L2)//2]}, max {L2[-1]}")

# data files
num_pat = {"k_suffix": r"^\s*[\d.,]+\s*k\s*$", "tr_infix": r"\d+\s*tr\s*\d*", "dong_suffix": r"[\d.,]+\s*(đ|₫|vnđ|vnd)\s*$", "space_thousands": r"^\d{1,3}(\s\d{3})+$", "comma_decimal": r"^\d+,\d+$"}
date_pat = {"iso": r"^\d{4}-\d{2}-\d{2}", "dmy_slash": r"^\d{1,2}/\d{1,2}/\d{4}", "dm_slash_noyear": r"^\d{1,2}/\d{1,2}$", "thg": r"thg\s*\d+", "excel_serial": r"^\d{5}(\.\d+)?$"}
stats = collections.Counter(); sheets = 0; rows_tot = 0; big = []
ddir = os.path.join(root, "data")
for biz in sorted(os.listdir(ddir)) if os.path.isdir(ddir) else []:
    for dp, _, fs in os.walk(os.path.join(ddir, biz)):
        for f in fs:
            fp = os.path.join(dp, f); ext = f.lower().rsplit(".", 1)[-1]
            frames = {}
            try:
                if ext == "csv": frames[f] = pd.read_csv(fp, dtype=str, header=None, encoding="utf-8", on_bad_lines="skip")
                elif ext in ("xlsx", "xls"):
                    for sn, df in pd.read_excel(fp, sheet_name=None, dtype=str, header=None).items(): frames[f"{f}:{sn}"] = df
                elif ext in ("sqlite", "db"):
                    con = sqlite3.connect(fp)
                    for (t,) in con.execute("select name from sqlite_master where type='table'"): frames[f"{f}:{t}"] = pd.read_sql(f'select * from "{t}"', con).astype(str)
                else: continue
            except Exception as e:
                stats["read_error"] += 1; continue
            for name, df in frames.items():
                sheets += 1; rows_tot += len(df); big.append((len(df), biz, name))
                first = df.head(8).astype(str).values.tolist()
                nn = [sum(1 for v in r if v not in ("nan", "None", "")) for r in first]
                if nn and nn[0] <= max(1, df.shape[1] // 3): stats["title_row_before_header"] += 1
                flat = df.astype(str).stack()
                s = flat.sample(min(len(flat), 4000), random_state=0) if len(flat) else flat
                for k, pat in num_pat.items(): stats["num_"+k] += int(s.str.contains(pat, flags=re.I, regex=True).sum() > 0)
                for k, pat in date_pat.items(): stats["date_"+k] += int(s.str.contains(pat, flags=re.I, regex=True).sum() > 0)
                if s.str.contains(r"^(tổng|total|ghi chú)", flags=re.I, regex=True).any(): stats["total_or_note_rows"] += 1
                hdr = df.iloc[0].astype(str).tolist() if len(df) else []
                if len(df) > 20 and (df.iloc[1:] .astype(str).apply(lambda r: r.tolist() == hdr, axis=1)).any(): stats["repeated_header"] += 1
p(f"\n## Data files\n- tables/sheets read: {sheets}; total rows: {rows_tot}")
p("- messiness counters (number of tables showing the pattern):"); [p(f"  - {k}: {v}") for k, v in sorted(stats.items())]
big.sort(reverse=True); p("- largest tables: " + "; ".join(f"{b}/{n} ({r} rows)" for r, b, n in big[:8]))
rb = os.path.join(root, "rulebook.md")
if os.path.exists(rb): p(f"\n## rulebook.md: {len(open(rb,encoding='utf-8').read())} chars (read it fully by hand!)")
open(out, "w", encoding="utf-8").write("\n".join(L)); print("\n".join(L)); print(f"\n-> {out}")

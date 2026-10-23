# live_link_features_v1 confirmation ids: admission read of the REPLACEMENT id 30297 (llf_conf_r1/cells_react) and the final 12-id list. Reads ONLY the queue_share field of each reactive summary.
# A topology is admitted iff all 8 of its reactive cells (4 windows x 2 rungs) have queue_share <= 0.80; unknown (missing, failed, no field) is not a pass.
import hashlib, json, os, re
U = "/share/nikola.lukic/uncut_groups"
NAME = re.compile(r"^cc40s(\d+)__(g\d)(moderate|heavy)__reactive_s0\.(summary|failed)\.json$")
def read(d):
    rows = {}
    for fn in sorted(os.listdir(d)):
        m = NAME.match(fn)
        if m: rows[(int(m.group(1)), m.group(2), m.group(3))] = json.load(open(os.path.join(d, fn))).get("queue_share") if m.group(4) == "summary" else None
    return rows
rows = {**read(U + "/llf_conf/cells_react"), **read(U + "/llf_conf_r1/cells_react")}
final = [t for t in range(30285, 30297) if t != 30288] + [30297]
bad = {}
for t in final:
    b = [(w, r, rows.get((t, w, r))) for w in ("g0", "g1", "g2", "g3") for r in ("moderate", "heavy") if rows.get((t, w, r)) is None or rows[(t, w, r)] > 0.80]
    if b: bad[t] = b
adm = {"final_ids": final, "sha256_comma_joined": hashlib.sha256(",".join(map(str, final)).encode()).hexdigest(), "failing": sorted(bad), "detail": bad,
       "max_queue_share_30297": max(v for k, v in rows.items() if k[0] == 30297 and v is not None), "max_queue_share_all": max(v for k, v in rows.items() if k[0] in final and v is not None),
       "replaced": {"30288": "g3heavy reactive killed at LIMIT 14400 and again at 28800 (unknown, not a pass); replaced by 30297 under the filed rule"}}
json.dump(adm, open(U + "/llf_conf_r1/admission_final.json", "w"), indent=1)
print(f"30297: cells {sum(1 for k in rows if k[0]==30297)} (expect 8); final 12 ids {final}; failing {sorted(bad)}; sha256 {adm['sha256_comma_joined']}; max queue_share 30297 {adm['max_queue_share_30297']}, all {adm['max_queue_share_all']}")
print("30297 cells:", sorted((k[1] + k[2], v) for k, v in rows.items() if k[0] == 30297))

"""Verify every DOI-bearing reference against Crossref (title, first author, year, venue).
Writes research/reference_verification.json. References without DOIs are checked separately."""
import json, time, pathlib, requests
DOIS = {
 "dunn2020matbench": "10.1038/s41524-020-00406-3",
 "debreuck2021modnet": "10.1038/s41524-021-00552-2",
 "debreuck2021bias": "10.1088/1361-648x/ac1280",
 "wang2021crabnet": "10.1038/s41524-021-00545-1",
 "goodall2020roost": "10.1038/s41467-020-19964-7",
 "ward2016magpie": "10.1038/npjcompumats.2016.28",
 "ward2018matminer": "10.1016/j.commatsci.2018.05.018",
 "ong2013pymatgen": "10.1016/j.commatsci.2012.10.028",
 "wolpert1992stacked": "10.1016/S0893-6080(05)80023-1",
 "breiman1996stacked": "10.1007/BF00117832",
 "vanderlaan2007super": "10.2202/1544-6115.1309",
 "hengl2007rk": "10.1016/j.cageo.2007.05.001",
 "kauwe2020bandgap": "10.1007/s40192-020-00178-0",
 "zhuo2018bandgap": "10.1021/acs.jpclett.8b00124",
 "miccio2025ee": "10.1016/j.commatsci.2025.114092",
 "chang2022moe": "10.1038/s41524-022-00929-x",
 "chen2016xgboost": "10.1145/2939672.2939785",
 "nadeau2003inference": "10.1023/A:1024068626366",
 "hollmann2025tabpfn": "10.1038/s41586-024-08328-6",
 "zou2025ecsg": "10.1038/s41467-024-55525-y",
 "shmalo2026kernel": "10.5281/zenodo.23329780",
 "geurts2006et": "10.1007/s10994-006-6226-1",
 "breiman2001rf": "10.1023/A:1010933404324",
 "meredig2014": "10.1103/PhysRevB.89.094104",
 "zhao2025hea": "10.3389/fmats.2025.1601874",
 "chernozhukov2018dml": "10.1111/ectj.12097",
 "ramakrishnan2015delta": "10.1021/acs.jctc.5b00099",
 "akiba2019optuna": "10.1145/3292500.3330701",
 "jain2013mp": "10.1063/1.4812323",
 "oliynyk2016": "10.1021/acs.chemmater.6b02724",
 "ward2018bmg": "10.1016/j.actamat.2018.08.002",
 "yang2012ss": "10.1016/j.matchemphys.2011.11.021",
 "laws2015ape": "10.1038/ncomms9123",
 "lee2016gap": "10.1103/PhysRevB.93.115104",
 "chen2021multifidelity": "10.1038/s43588-020-00002-x",
 "jha2019transfer": "10.1038/s41467-019-13297-w",
 "liu2022denoise": "10.1038/s41524-022-00925-1",
 "ko2024gap": "10.1016/j.mtcomm.2024.110717",
 "pilania2017mf": "10.1016/j.commatsci.2016.12.004",
}
out = {}
for k, d in DOIS.items():
    try:
        r = requests.get(f"https://api.crossref.org/works/{d}", timeout=30, headers={"User-Agent": "matbench-paper-refcheck (mailto:gurrambhaskar.ai@gmail.com)"})
        if r.status_code == 200:
            m = r.json()["message"]
            a = m.get("author") or []
            out[k] = {"doi": d, "ok": True, "title": (m.get("title") or [""])[0],
                      "first_author": (a[0].get("family", "") + ", " + a[0].get("given", "")) if a else "",
                      "n_authors": len(a), "authors": [(x.get("given", "") + " " + x.get("family", "")).strip() for x in a],
                      "year": (m.get("issued", {}).get("date-parts") or [[None]])[0][0],
                      "venue": (m.get("container-title") or [""])[0], "volume": m.get("volume"),
                      "page": m.get("page") or m.get("article-number"), "type": m.get("type")}
        else:
            out[k] = {"doi": d, "ok": False, "status": r.status_code}
    except Exception as e:
        out[k] = {"doi": d, "ok": False, "error": str(e)}
    time.sleep(0.5)
p = pathlib.Path(__file__).with_name("reference_verification.json")
p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
for k, v in out.items():
    print(k, v.get("ok"), v.get("year"), (v.get("first_author") or "")[:25], "|", (v.get("title") or str(v.get("status")))[:90])

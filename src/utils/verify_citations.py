"""Query Crossref (DOIs) and arXiv (IDs) for citation verification. Writes raw JSON metadata.

Usage: python -m src.utils.verify_citations  -> citations/_lookup_raw.json
"""
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

UA = "citation-check/0.1 (research project; mailto unset)"

DOIS = {
    # consensus report references
    "cr_LHe2025": "10.1080/21642583.2025.2518963",
    "cr_Ali2023": "10.1016/j.adhoc.2023.103320",
    "cr_Bovenzi2026": "10.1016/j.rineng.2026.111405",
    "cr_DACL2026": "10.1109/jiot.2026.3658904",
    "cr_SaezDeCamara2023": "10.1016/j.cose.2023.103299",
    "cr_FedDTCN2026": "10.3390/s26061918",
    "cr_LiLu2024": "10.3390/s24072122",
    "cr_LiYao2024": "10.1016/j.eswa.2024.124965",
    "cr_LopezMartin2021": "10.1016/j.inffus.2021.09.014",
    "cr_FeCoGraph2025": "10.1109/tifs.2025.3541890",
    "cr_Nguyen2026": "10.1186/s42400-025-00509-8",
    "cr_FeCoTDSC2025": "10.1109/tdsc.2025.3544106",
    "cr_TGDCLNet2025": "10.1109/jiot.2025.3586290",
    "cr_Yue2022": "10.1109/tnsm.2022.3218843",
    # papers/ PDFs with DOIs
    "feco_infocom2022": "10.1109/INFOCOM48880.2022.9796926",
    "clad_tnsm2026": "10.1109/TNSM.2026.3652529",
    "kitsune_2018": "10.14722/ndss.2018.23204",
    "nbaiot_2018": "10.1109/MPRV.2018.03367731",
    "moon_cvpr2021": "10.1109/CVPR46437.2021.01057",
    # datasets
    "ciciot2023": "10.3390/s23135941",
    "nslkdd_2009": "10.1109/CISDA.2009.5356528",
}
ARXIV = {
    "arp_dos_donts": "2010.09470", "clad_arxiv": "2601.09902", "conformal_tutorial": "2107.07511",
    "deep_sad": "1906.02694", "fedprox": "1812.06127", "moon": "2103.16257", "simclr": "2002.05709",
    "supcon": "2004.11362", "fedavg": "1602.05629", "rob_fcp": "2406.01960", "saez_de_camara": "2303.15986",
    "nbaiot": "1805.03409", "kitsune": "1802.09089",
}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def crossref(doi):
    try:
        m = json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"]
        return {
            "status": "found", "doi": m.get("DOI"), "title": (m.get("title") or [""])[0],
            "authors": [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in m.get("author", [])],
            "container": (m.get("container-title") or [""])[0], "volume": m.get("volume"),
            "issue": m.get("issue"), "page": m.get("page"),
            "year": (m.get("published") or m.get("issued") or {}).get("date-parts", [[None]])[0][0],
            "type": m.get("type"),
        }
    except Exception as e:  # noqa
        return {"status": "error", "error": repr(e)}


def arxiv(aid):
    try:
        x = get("http://export.arxiv.org/api/query?id_list=" + aid)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        e = ET.fromstring(x).find("a:entry", ns)
        return {
            "status": "found", "id": aid, "title": " ".join(e.find("a:title", ns).text.split()),
            "authors": [a.find("a:name", ns).text for a in e.findall("a:author", ns)],
            "published": e.find("a:published", ns).text,
            "journal_ref": (e.find("{http://arxiv.org/schemas/atom}journal_ref").text
                            if e.find("{http://arxiv.org/schemas/atom}journal_ref") is not None else None),
            "doi": (e.find("{http://arxiv.org/schemas/atom}doi").text
                    if e.find("{http://arxiv.org/schemas/atom}doi") is not None else None),
        }
    except Exception as ex:  # noqa
        return {"status": "error", "error": repr(ex)}


if __name__ == "__main__":
    out = {"crossref": {}, "arxiv": {}}
    for k, d in DOIS.items():
        out["crossref"][k] = crossref(d)
        time.sleep(0.3)
    for k, a in ARXIV.items():
        out["arxiv"][k] = arxiv(a)
        time.sleep(3.1)  # arXiv API etiquette
    with open("citations/_lookup_raw.json", "w") as f:
        json.dump(out, f, indent=1)
    for sec in out:
        for k, v in out[sec].items():
            print(sec, k, v.get("status"), "|", v.get("title", v.get("error", ""))[:90], "|", v.get("year", v.get("published", "")))

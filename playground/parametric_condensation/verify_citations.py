#!/usr/bin/env python3
"""Verify the cited DOIs against Crossref; print title/authors/year/venue.

Unverifiable entries are reported as such - never silently dropped.
"""
from __future__ import annotations

import json
import subprocess

DOIS = [
    ("Voormeeren & Rixen 2012", "10.1007/978-1-4614-2422-2_27"),
    ("Lee & Cho 2017", "10.1016/j.cma.2017.02.010"),
    ("Lee 2019", "10.1016/j.compstruc.2018.10.018"),
    ("Liu et al 2019", "10.1155/2019/6407437"),
    ("Cheon et al 2024", "10.1016/j.amc.2024.128778"),
    ("Kim & Lee 2015", "10.1002/nme.4880"),
    ("Bondsman Ahn Kim 2024", "10.1016/j.ymssp.2023.111074"),
    ("Huynh Knezevic Patera 2013", "10.1051/m2an/2012022"),
    ("Eftang & Patera 2013", "10.1002/nme.4543"),
    ("Smetana 2015", "10.1016/j.cma.2014.09.020"),
    ("Smetana & Patera 2016", "10.1137/15M1009603"),
    ("Bhouri & Patera 2021", "10.1016/j.cma.2021.114004"),
    ("Codecasa et al 2017", "10.1016/j.microrel.2017.06.059"),
    ("Codecasa et al 2018", "10.1016/j.microrel.2018.03.030"),
    ("Xie & Swaminathan 2014", "10.1109/TCPMT.2013.2286158"),
    ("Hernandez-Becerro 2020", "10.1007/s00466-020-01926-x"),
    ("Bruns & Benner 2015", "10.1080/13873954.2014.924534"),
    ("Amsallem & Farhat 2011", "10.1137/100813051"),
    ("Samuel et al 2013", "10.1109/TCPMT.2013.2248196"),
]


def fetch(doi: str):
    out = subprocess.run(
        ["curl.exe", "-s", "--max-time", "30", f"https://api.crossref.org/works/{doi}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if out.returncode != 0 or not out.stdout.strip():
        return None
    try:
        return json.loads(out.stdout)["message"]
    except Exception:
        return None


for label, doi in DOIS:
    message = fetch(doi)
    if message is None:
        print(f"UNVERIFIED  {label:<30} {doi}")
        continue
    title = (message.get("title") or ["<no title>"])[0]
    container = (message.get("container-title") or ["<no venue>"])[0]
    year = None
    for key in ("published-print", "published-online", "issued", "created"):
        part = message.get(key, {}).get("date-parts", [[None]])[0]
        if part and part[0]:
            year = part[0]
            break
    authors = message.get("author") or []
    family = ", ".join(a.get("family", "?") for a in authors[:6])
    print(f"OK          {label:<30} {doi}")
    print(f"    title  : {title}")
    print(f"    authors: {family}")
    print(f"    venue  : {container}  ({year})")

"""Sprawdza, czy pliki zainstalowanych pakietów są na miejscu (lista RECORD w dist-info).

uv uznaje pakiet za zainstalowany po samych metadanych, więc nie zauważy usuniętych lub
uciętych plików. Ten skrypt porównuje rozmiary plików z listą RECORD i wypisuje nazwy
uszkodzonych pakietów (po jednej w wierszu) - bootstrap.ps1 instaluje je ponownie.

RECORD jest czytany bezpośrednio: Distribution.files od Pythona 3.12 pomija brakujące pliki.

Uruchamiać Pythonem instalacji w trybie izolowanym:  python\\python.exe -I verify_packages.py
"""
import csv
import importlib.metadata as metadata
import os


def broken_distributions():
    broken = set()
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        record = dist.read_text("RECORD")
        if not name or not record:
            continue
        for row in csv.reader(record.splitlines()):
            # ścieżka,sha256=...,rozmiar; pomijamy wpisy bez rozmiaru (sam RECORD) i pliki spoza
            # site-packages (../../Scripts/*.exe, share/): program ich nie używa, a launchery
            # w Scripts i tak wskazują starą ścieżkę po przeniesieniu folderu.
            if len(row) < 3 or not row[2].isdigit() or row[0].replace("\\", "/").startswith("../"):
                continue
            try:
                intact = os.path.getsize(dist.locate_file(row[0])) == int(row[2])
            except OSError:
                intact = False
            if not intact:
                broken.add(name)
                break
    return sorted(broken, key=str.lower)


if __name__ == "__main__":
    for package in broken_distributions():
        print(package)

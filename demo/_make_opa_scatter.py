#!/usr/bin/env python
"""Draw the OPA scatter figure used by the demo.

Drawn here rather than lifted from a paper so it can be redistributed freely,
and so what is printed on it is known exactly — the reference reading committed
beside it is then correct by construction.

    python demo/_make_opa_scatter.py
"""
from __future__ import annotations

import math
import random
from pathlib import Path

try:
    import pymupdf
except ImportError:                            # pragma: no cover - optional dep
    import fitz as pymupdf

TOP, BOTTOM, LEFT = 62, 232, 84
DECADES = 4                                     # y axis 1 .. 10^4
GROUPS = (("PCV13", 150, 3.08), ("PCV20", 250, 2.74), ("Placebo", 350, 1.12))
BLUE = (0.16, 0.44, 0.71)


def y_for(titer: float) -> float:
    return BOTTOM - (math.log10(titer) / DECADES) * (BOTTOM - TOP)


def main() -> int:
    random.seed(613)
    document = pymupdf.open()
    page = document.new_page(width=430, height=290)

    page.insert_text((22, 24), "Figure 1. Serotype 6B OPA titers 28 days after "
                               "vaccination.", fontsize=8)
    page.draw_line((LEFT, TOP), (LEFT, BOTTOM))
    page.draw_line((LEFT, BOTTOM), (390, BOTTOM))
    for decade in range(DECADES + 1):
        y = y_for(10.0 ** decade)
        page.draw_line((LEFT - 5, y), (LEFT, y))
        page.insert_text((LEFT - 32, y + 2.5), f"{10 ** decade:g}", fontsize=7)
    page.insert_text((26, 190), "OPA titer (1/dilution)", fontsize=8, rotate=90)

    for name, x, mean in GROUPS:
        titers = [min(max(10 ** random.gauss(mean, 0.45), 8), 8000)
                  for _ in range(34)]
        for titer in titers:
            page.draw_circle((x + random.uniform(-11, 11), y_for(titer)), 2.1,
                             color=None, fill=BLUE)
        gmt = 10 ** (sum(math.log10(t) for t in titers) / len(titers))
        page.draw_line((x - 13, y_for(gmt)), (x + 13, y_for(gmt)))
        page.insert_text((x - 16, TOP - 8), f"GMT {gmt:.0f}", fontsize=7)
        page.insert_text((x - 14, BOTTOM + 14), name, fontsize=8)
        page.insert_text((x - 10, BOTTOM + 24), "n=34", fontsize=6.5)
        print(f"{name:8} GMT {gmt:7.1f}")

    page.get_pixmap(matrix=pymupdf.Matrix(2.4, 2.4), alpha=False).save(
        str(Path(__file__).resolve().parent / "opa-scatter.png"))
    document.close()
    print("wrote opa-scatter.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

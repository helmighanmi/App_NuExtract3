#!/usr/bin/env python3
"""Generate a synthetic invoice PDF + ground truth for demos and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pymupdf as fitz


def build_invoice(path: Path) -> dict:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4

    navy = (0.09, 0.15, 0.28)
    grey = (0.35, 0.38, 0.42)

    page.insert_text((50, 60), "ATLAS ROBOTICS GmbH", fontsize=20, color=navy)
    page.insert_text((50, 80), "Industriestrasse 12, 80339 Munich, Germany", fontsize=9, color=grey)
    page.insert_text(
        (50, 94), "billing@atlas-robotics.de  |  VAT DE123456789", fontsize=9, color=grey
    )

    page.insert_text((400, 60), "INVOICE", fontsize=24, color=navy)
    page.insert_text((400, 84), "INV-2026-0042", fontsize=11, color=navy)
    page.insert_text((400, 100), "Date: 2026-03-14", fontsize=9, color=grey)
    page.insert_text((400, 114), "Due: 2026-04-13", fontsize=9, color=grey)

    page.insert_text((50, 150), "BILL TO", fontsize=9, color=grey)
    page.insert_text((50, 168), "Nordwind Logistics AG", fontsize=12, color=navy)
    page.insert_text((50, 184), "Hafenstrasse 8, 20457 Hamburg, Germany", fontsize=9, color=grey)
    page.insert_text((50, 198), "VAT DE987654321", fontsize=9, color=grey)

    headers = ["Description", "Qty", "Unit price", "Total"]
    xs = [50, 330, 400, 490]
    y = 250
    page.draw_line((50, y - 12), (545, y - 12), color=grey, width=0.6)
    for x, header in zip(xs, headers, strict=True):
        page.insert_text((x, y), header, fontsize=9, color=grey)
    page.draw_line((50, y + 6), (545, y + 6), color=grey, width=0.6)

    items = [
        ("Industrial robot arm AR-6", 2, 1250.00, 2500.00),
        ("Gripper module GM-2", 5, 180.00, 900.00),
        ("On-site calibration service", 1, 1250.00, 1250.00),
    ]
    y = 280
    for description, quantity, unit_price, total in items:
        page.insert_text((xs[0], y), description, fontsize=10, color=navy)
        page.insert_text((xs[1], y), str(quantity), fontsize=10, color=navy)
        page.insert_text((xs[2], y), f"{unit_price:,.2f}", fontsize=10, color=navy)
        page.insert_text((xs[3], y), f"{total:,.2f}", fontsize=10, color=navy)
        y += 24

    subtotal, tax, total = 4650.00, 883.50, 5533.50
    page.draw_line((330, y + 6), (545, y + 6), color=grey, width=0.6)
    for label, value, offset in [
        ("Subtotal", subtotal, 24),
        ("VAT 19%", tax, 42),
        ("Total EUR", total, 66),
    ]:
        page.insert_text((380, y + offset), label, fontsize=10, color=grey)
        page.insert_text((490, y + offset), f"{value:,.2f}", fontsize=10, color=navy)

    page.insert_text((50, y + 130), "Payment terms: Net 30", fontsize=9, color=grey)
    page.insert_text(
        (50, y + 146),
        "Please transfer to IBAN DE44 5001 0517 5407 3249 31, BIC INGDDEFFXXX.",
        fontsize=9,
        color=grey,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()

    return {
        "invoice_number": "INV-2026-0042",
        "invoice_date": "2026-03-14",
        "due_date": "2026-04-13",
        "vendor": {
            "name": "Atlas Robotics GmbH",
            "address": "Industriestrasse 12, 80339 Munich, Germany",
            "tax_id": "DE123456789",
            "email": "billing@atlas-robotics.de",
        },
        "customer": {
            "name": "Nordwind Logistics AG",
            "address": "Hafenstrasse 8, 20457 Hamburg, Germany",
            "tax_id": "DE987654321",
        },
        "currency": "EUR",
        "subtotal": 4650.0,
        "tax": 883.5,
        "total": 5533.5,
        "payment_terms": "Net 30",
        "line_items": [
            {
                "description": "Industrial robot arm AR-6",
                "quantity": 2,
                "unit_price": 1250.0,
                "total": 2500.0,
            },
            {
                "description": "Gripper module GM-2",
                "quantity": 5,
                "unit_price": 180.0,
                "total": 900.0,
            },
            {
                "description": "On-site calibration service",
                "quantity": 1,
                "unit_price": 1250.0,
                "total": 1250.0,
            },
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/sample_invoice.pdf"))
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data/sample_invoice.gt.json"),
    )
    args = parser.parse_args()

    ground_truth = build_invoice(args.out)
    args.ground_truth.write_text(
        json.dumps(ground_truth, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.out}")
    print(f"Wrote {args.ground_truth}")


if __name__ == "__main__":
    main()

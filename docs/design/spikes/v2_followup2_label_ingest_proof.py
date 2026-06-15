#!/usr/bin/env python3
"""V2 follow-up #2 spike — QZSS OHI + NOAA GOES maneuver-label ingest proof (stdlib only).

Companion to ``v2_followup_label_ingest_proof.py`` (the Galileo NAGU proof). It demonstrates that the
two *new shippable operator* sources found in the second follow-up survey are machine-ingestible into
the project's normalised label record ``(norad_id, epoch, type, delta_v, source, ...)``,
deterministically, from a verbatim embedded sample of each — no network.

Sources (both redistribution-clean, so the embedded samples are reproduced under their terms):

- **QZSS OHI** — Operational History Information, Cabinet Office of Japan
  (https://qzss.go.jp/en/technical/qzssinfo/). Per-satellite ``ohi-qzsN.txt`` with a
  ``#+SATELLITE/MANEUVER`` block listing executed burns as ``start,end,duration,DVX,DVY,DVZ``. The
  only surveyed operator feed with an executed Δv. Reused under CC-BY-4.0 ("Source: Quasi-Zenith
  Satellite System website"). The Δv axis frame is undocumented, so only the magnitude is kept
  (``type = None``); clustered burns collapse into one event (Δv = sum of burn magnitudes).
- **NOAA GOES navsum** — NOAA OSPO navigation summary
  (https://www.ospo.noaa.gov/resources/cemscs/navsum.txt). A live-state file whose Comments footer
  names each GOES bird's last-maneuver day at ``yy/ddd`` granularity; US-Government public domain.
  A maneuver history is built by replaying its Internet-Archive snapshots (not shown here — the proof
  parses one snapshot into the latest-maneuver epoch per spacecraft).

Run: ``python docs/design/spikes/v2_followup2_label_ingest_proof.py`` — prints the normalised records.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone

# --- embedded verbatim samples ----------------------------------------------------------------------

# A real ohi-qzs2.txt excerpt (Cabinet Office of Japan, reused with attribution): two same-day burns
# (one campaign) then two more ~52 days later (a second campaign).
_QZS2_OHI = """\
#+SATELLITE/MASS
#DATE TIME START(UTC),END(UTC),MASS(kg)
2017-11-15 10:34:47,2018-01-06 06:55:39,2320
#-SATELLITE/MASS

#+SATELLITE/MANEUVER
#DATE TIME START(UTC),END(UTC),DURATION,DVX(m/s),DVY(m/s),DVZ(m/s)
2017-11-15 11:03:31,2017-11-15 11:05:50,00:02:19,-2.325,0.004,0.032
2017-11-15 18:33:48,2017-11-15 18:35:32,00:01:44,1.756,0.017,0.024
2018-01-06 07:23:19,2018-01-06 07:24:18,00:00:59,-0.916,0.001,0.012
2018-01-06 14:53:27,2018-01-06 14:54:10,00:00:43,0.692,-0.006,0.008
#-SATELLITE/MANEUVER
"""

# A real navsum.txt excerpt (NOAA OSPO, US-Government public domain), one spacecraft block.
_NAVSUM = (
    "=============================================================\n"
    "Spacecraft :                                  GOES-16\n"
    "Comments:\n"
    "Fuel and oxidizer remaining are estimates after the last maneuver on 26/159.\n"
)

_QZS_TO_NORAD = {"QZS-2": 42738}
_GOES_TO_NORAD = {"GOES-16": 41866}
_EVENT_GAP = timedelta(days=2)


def parse_qzss_ohi(text: str, qzs_label: str) -> list[dict]:
    """Parse the maneuver section of an OHI file into collapsed-event records (magnitude only)."""
    rows: list[tuple[datetime, datetime, float]] = []
    in_section = False
    for raw in text.splitlines():
        line = raw.strip()
        if line == "#+SATELLITE/MANEUVER":
            in_section = True
            continue
        if line == "#-SATELLITE/MANEUVER":
            break
        if not in_section or not line or line.startswith("#"):
            continue
        f = [c.strip() for c in line.split(",")]
        start = datetime.strptime(f[0], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        end = datetime.strptime(f[1], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        rows.append((start, end, math.sqrt(float(f[3]) ** 2 + float(f[4]) ** 2 + float(f[5]) ** 2)))
    rows.sort(key=lambda r: r[0])

    events: list[dict] = []
    es, ee, edv, last = rows[0][0], rows[0][1], rows[0][2], rows[0][1]
    for s, e, mag in rows[1:]:
        if s - last <= _EVENT_GAP:
            ee, edv, last = max(ee, e), edv + mag, max(last, e)
        else:
            events.append(_qzss_event(qzs_label, es, ee, edv))
            es, ee, edv, last = s, e, mag, e
    events.append(_qzss_event(qzs_label, es, ee, edv))
    return events


def _qzss_event(qzs_label: str, start: datetime, end: datetime, delta_v: float) -> dict:
    return {
        "norad_id": _QZS_TO_NORAD.get(qzs_label),
        "epoch": start,
        "window": (start, end),
        "type": None,  # the OHI Δv frame is undocumented -> magnitude only
        "delta_v": round(delta_v, 6),
        "source": "QZSS-OHI",
    }


def parse_navsum(text: str) -> list[dict]:
    """Parse one navsum snapshot into the latest-maneuver epoch per spacecraft."""
    records: list[dict] = []
    for block in re.split(r"^=+$", text, flags=re.MULTILINE):
        name = re.search(r"Spacecraft\s*:\s*(GOES-\d+)", block)
        epoch = re.search(r"last maneuver on\s+(\d{2})/(\d{1,3})", block)
        if not name or not epoch:
            continue
        day = datetime(2000 + int(epoch.group(1)), 1, 1, tzinfo=timezone.utc) + timedelta(
            days=int(epoch.group(2)) - 1
        )
        records.append(
            {
                "norad_id": _GOES_TO_NORAD.get(name.group(1).upper()),
                "epoch": day + timedelta(hours=12),
                "window": (day, day + timedelta(days=1)),
                "type": None,
                "delta_v": None,  # navsum is epoch-only
                "source": "NOAA-GOES",
            }
        )
    return records


if __name__ == "__main__":
    qzss = parse_qzss_ohi(_QZS2_OHI, "QZS-2")
    goes = parse_navsum(_NAVSUM)
    assert len(qzss) == 2, qzss  # two campaigns collapse from four burns
    assert qzss[0]["delta_v"] == round(
        math.hypot(-2.325, 0.004, 0.032) + math.hypot(1.756, 0.017, 0.024), 6
    )
    assert parse_qzss_ohi(_QZS2_OHI, "QZS-2") == qzss  # deterministic
    assert len(goes) == 1 and goes[0]["norad_id"] == 41866, goes
    for record in qzss + goes:
        print(record)
    print("OK: QZSS OHI + NOAA GOES navsum ingest into the normalised label record")

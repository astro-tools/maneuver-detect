# V7 follow-up — hidden-label competition via a never-committed forward holdout

**Status:** findings + recommendation, feeding **D16** (the hidden-label competition track). This
follow-up settles the design the **D12 amendment** deferred: the v0.2 leaderboard ships in
reproducibility mode because the answer key is committed, so the hidden-label firewall V7 designed is
unbuildable on it; a true competition needs a test set whose labels live in no public file. Three
open questions are settled here — **where the holdout's labels come from (and the GEO problem)**,
**whether it is a one-time cutoff or a rolling holdout**, and **when the held-back subset is
revealed** — and the V7 firewall is shown to hold on a forward holdout. The competition's
implementation (a second board on the existing Space, the holdout-fixture builder, the refresh
process) is the follow-up build this spike unblocks, not part of it.

## Question

The v0.2 board is a **reproducibility / convenience** board on the *public* test split: the committed
`dataset/v0.2/labels.json` + `splits.json` publish the full answer key, so the hidden-label firewall
(hidden labels + a held-back private subset deciding the final ranking) the V7 spike designed cannot
exist on it — the D12 amendment dropped it and kept the aggregate-only response and the rate limit as
courtesy guards. A true hidden-label competition was deferred to "a separate, never-committed
forward/rolling holdout." This follow-up answers four things before that gets built:

1. **Source & disjointness.** What is the holdout, concretely, and how is it guaranteed disjoint from
   the public dataset so no competition label can ever leak into `labels.json` / `splits.json`?
2. **The GEO problem.** Can a *forward* holdout be populated per class — especially GEO, the
   load-bearing risk, where the v0.2 labels were self-derived and the operator feeds were thin?
3. **Cutoff vs. rolling, and reveal cadence.** A one-time forward cutoff (simpler, ages out) or a
   sustained rolling holdout (ongoing burden); and when is the private subset scored / revealed?
4. **Does the firewall hold?** With a never-committed holdout, are all four D12.3 mechanisms restored
   as *integrity*, not courtesy — and does the public/private split actually protect the ranking?

---

## Part A — The forward holdout: definition and disjointness

The holdout is the set of maneuvers whose **epoch falls strictly after the public dataset's freeze**
— the release freeze that fixes `labels.json` / `splits.json` for a version — reconstructed from the
*same* operator feeds the public dataset uses, through the **same D2 recipe**, and **never added to
any committed file**. It is supplied to the competition board exactly as the v0.2 reproducibility
fixture is (a private HF Dataset the Space reads at runtime), with one difference that makes the
competition real: on the v0.2 board the fixture is private only to honour D2 (the *labels* are
already public); here the **labels themselves are private** — they appear in no committed file and no
response.

Disjointness is **temporal, not by satellite**, and is a single auditable comparison: every committed
public label has `epoch <= freeze`; every holdout label has `epoch > freeze`. The same object may
maneuver on both sides of the freeze (a satellite keeps station-keeping) — that is expected and fine,
because the partition is by epoch, not by NORAD id. This is the boundary D15 already reserved for
this track ("the hidden-label competition draws a forward holdout … disjoint from the historical
labels committed here, so nothing need be private and no competition label can leak"), now made
precise: the cut is `epoch > freeze`, and the proof asserts both that no committed label sits after
the freeze and that no holdout epoch appears in the committed set.

One D2 consequence carries over unchanged from the v0.2 board: the scorer's matching windows are real
elset epochs (derived Space-Track data the dataset does not redistribute), so the holdout fixture is
**built offline from a credentialed reconstruction and supplied as private deploy-time data**, never
committed — the same `build_*_fixture.py` pattern, scoped to the forward window.

## Part B — Per-class forward-holdout viability (the GEO problem)

The v0.2 caveat that motivated #95's "thin forward GEO" risk was real, but the v0.3 dataset growth
(**D15**) changed the source picture, so a forward holdout is viable per class — densely for LEO/MEO,
thinly-but-for-real for GEO/IGSO:

- **LEO — dense.** DORIS/IDS + ILRS publish operator maneuver files *with Δv* on the altimetry
  fleet at roughly monthly cadence per object (D3). A forward window of a release cycle yields a
  healthy LEO set — the competition's anchor class, and the one carrying a real Δv label.
- **MEO — moderate.** GPS NANU `FCSTDV` + Galileo NAGU `PLN_MANV` are live, no-auth,
  machine-ingestible notice feeds (D3/D13); both update going forward. GNSS satellites station-keep
  rarely, so the realized forward count is modest, but two independent operators contribute.
- **GEO / IGSO — thin, but no longer empty or self-derived.** This is the change that makes the
  competition honest. **QZSS OHI** files ship executed orbit-maintenance maneuvers *with a Δv vector*
  (IGSO for QZS-2/4/1R; GEO for the equatorial QZS-3/6), and **NOAA GOES** `navsum` epochs (replayed
  from Internet-Archive snapshots) give operator-announced GEO epochs — both **forward-collectable**
  (the feeds advance), breaking the v0.2 GEO self-label circularity (D15). **BeiDou NABU stays dead**
  (a JS-only SPA with no maneuver semantics and a shallow rolling CDN window — not headless-crawlable;
  see the BeiDou-NABU finding), so GEO does not get the densest operator feed, but it is real.

Expected forward yield is therefore **order-of-magnitude** (operator cadences, not a measured count):
over a release-cycle window, LEO populates comfortably, MEO modestly, and GEO/IGSO produce a handful
of labels each. The consequence for scoring is not to drop the thin classes but to **score them
per-class with their honest small-`N` uncertainty** — the benchmark already reports per-class Wilson
confidence intervals, so a thin GEO/IGSO class shows up as a wide interval, not a hidden weakness. The
competition is meaningful on LEO/MEO from launch; GEO/IGSO sharpen as forward labels accumulate across
reveal cycles.

## Part C — Cutoff vs. rolling, and reveal cadence

A **one-time forward cutoff** (freeze the holdout once, at a fixed date) is operationally simplest but
**ages out**: as soon as the next release extends the public dataset past the cutoff, those labels
become public and the holdout is consumed — a competition with a single, expiring deadline.

The recommendation is a **rolling holdout keyed to the release cadence.** The private holdout is
always "maneuvers after the current public freeze." At each release the matured window's labels are
**revealed** — folded into the new public dataset (becoming training data for the next version) — and
the fresh forward window automatically becomes the new private holdout. This is the standard Kaggle
private-leaderboard-at-deadline pattern, with the **deadline = each release**:

- **Live board** scores a **public subset** of the current holdout continuously (honest iteration).
- **Final ranking** for a version is recomputed on the **held-back private subset** and frozen into
  that release — the **reveal is per release**, when the window's labels are published anyway.

This reuses the existing release rhythm — **no new schedule to invent** — and bounds the operational
burden to *one offline credentialed fixture build per release*, scoped to the forward window, reusing
the same D2 reconstruction the dataset build already performs. There is no standing service to
operate; the refresh is a step in the release cut.

## Part D — The D12.3 firewall, restored

The only reason D12.3 was unbuildable on v0.2 was that the answer key was committed and git-history
makes that irretractable. On a never-committed forward holdout that obstruction is gone, so all four
mechanisms are restored — now as **integrity**, not courtesy:

1. **Hidden labels** — the holdout labels are never shipped; they live only on the Space (private HF
   Dataset / secret).
2. **Public / private split** — the live board scores the public subset; the final ranking is
   recomputed on the held-back private subset, revealed only at the release (Part C).
3. **Rate limit (now an integrity bound)** — five scored submissions per user per UTC day. On the
   v0.2 board this only slowed abusive volume; here it genuinely bounds label leakage, because probing
   now leaks *hidden* labels.
4. **Fixed-schema, aggregate-only round trip** — unchanged from D12.2: a submission can carry nothing
   but predictions, and a response returns nothing but aggregate metrics.

The V7 probing analysis carries over verbatim: a single-detection oracle (submit one detection at one
candidate gap, watch whether headline recall ticks up) recovers labels at **one submission per
candidate gap** — `ceil(G / R)` submission-days at `R` scored submissions/user/day, anomalous-volume
detectable — and **only ever touches the public subset**. The private subset, in a different fixture
the live board never scores against, decides the ranking and is never exposed.

---

## Proof — competition firewall dry-run (real service)

[`v7_followup_competition_proof.py`](v7_followup_competition_proof.py) (stdlib + the installed
package; no network, no GPU; deterministic; fictional catalogue ids `9000x` per the V1/D2
no-redistribution practice) builds a synthetic forward holdout — a committed-public history before a
freeze and a disjoint, never-committed holdout after it, split into public and private subsets across
a LEO object and the two operator-real classes (GEO, IGSO) — and runs the **real**
`LeaderboardService` + shipped scorer against it. It asserts the partition is a clean epoch cut, runs
the four integrity checks, confirms the rate limit binds on the real service, then runs the probing
oracle and an overfit-transfer test to show the public/private firewall holds. Verbatim output:

```
V7 follow-up — hidden-label competition firewall on a forward holdout (real service)
====================================================================================

[1] Forward-holdout partition (the clean timestamp cut):
{
  "public_freeze": "2026-09-01T00:00:00+00:00",
  "committed_public_labels": 4,
  "forward_holdout_labels": 10,
  "holdout_public_subset": 6,
  "holdout_private_subset": 4
}

[2] Honest submission → public result the competition Space would return:
{
  "operating_point_fa_per_sat_year": 1.0,
  "headline_recall_above_floor": {
    "LEO": 1.0,
    "MEO": null,
    "GEO": 0.0,
    "IGSO": 0.0,
    "HEO": null
  },
  "timing_only_floor_auc": {
    "LEO": 0.62,
    "GEO": 0.68
  }
}

[3] Integrity checks (all assert-backed, passed):
    - forward holdout disjoint from the committed public set (clean epoch cut)
    - held-out label epochs absent from the payload
    - response is aggregate-only (no per-label match table)
    - submission channel rejected 3/3 non-prediction payloads
    - scoring is byte-deterministic across runs (D8)
    - rate limit binds at 5 scored submissions / user / UTC day

[4] Public/private firewall under the single-detection probing oracle:
{
  "candidate_gaps": 123,
  "public_subset_above_floor_labels": 5,
  "public_labels_recovered_by_probing": 5,
  "private_subset_above_floor_labels": 4,
  "private_labels_recovered_by_probing": 0,
  "rate_per_user_per_day": 5,
  "submission_days_to_exfiltrate_public_subset": 25
}

[5] A prober who memorised the public subset cannot win the private ranking:
{
  "overfit_headline_recall_public_subset": 1.0,
  "overfit_headline_recall_private_subset": 0.0
}

    => probing recovers only the public subset (5/5 of its above-floor labels) and never a
       private-subset label (0 recovered); the held-back private subset, revealed once at the release,
       decides the final ranking. On the open v0.2 answer key this firewall was
       unbuildable; on a never-committed forward holdout it holds.
```

What the run shows:

- **The partition is a clean epoch cut.** All four committed labels sit on/before the freeze; all ten
  holdout labels sit after it; no holdout epoch appears in the committed set. So no competition label
  can leak into `labels.json` / `splits.json` — the property the open v0.2 dataset could not have.
- **The response carries no labels.** The honest submission gets aggregate-only metrics (per-class
  above-floor recall + the published timing floor); no held-out label epoch — public *or* private —
  appears in the payload, and the submission channel rejects all three non-prediction payloads.
- **The firewall holds under probing.** The oracle recovers all five public-subset above-floor labels
  (one probe per gap, 123 gaps = 25 days at 5/user/day, anomalous-volume detectable) and **zero** of
  the four private-subset labels — they are in a fixture the live board never touches. A competitor
  who memorised the entire public subset scores **1.0 on the public board and 0.0 on the private
  one**: the public board can be fully overfit and it still does not move the ranking the private
  subset decides. This is exactly the firewall the v0.2 open answer key made impossible.

## Recommendation (→ D16)

- **D16.1 — the holdout.** A **never-committed forward holdout**: maneuvers with `epoch > freeze`,
  reconstructed from the same operator feeds via the D2 recipe, disjoint from the committed dataset by
  a clean epoch cut, supplied to the board as private deploy-time data (labels *and* windows private).
- **D16.2 — rolling, keyed to the release cadence.** Reject the one-time cutoff (ages out). The
  private holdout is always "after the current freeze"; at each release the matured window is revealed
  (folded into the next public dataset) and a fresh forward window opens. **Reveal cadence = per
  release.**
- **D16.3 — the firewall, restored as integrity.** Re-instate all four D12.3 mechanisms — hidden
  labels + public/private subset split (private scored once at release) + the 5/user/UTC-day rate
  limit (now an integrity bound) + the aggregate-only/fixed-schema round trip (D12.2, unchanged).
  Proven against the real service.
- **D16.4 — thin classes scored, not dropped.** LEO (Δv-labelled) and MEO carry the competition from
  launch; GEO/IGSO are populated for real by the D15 operator feeds (QZSS OHI, NOAA GOES) but thin, so
  they are scored per-class with their honest small-`N` Wilson intervals, sharpening across reveal
  cycles. BeiDou remains unavailable.
- **Out of scope (the follow-up build).** The second "competition" board on the Space + service, the
  holdout-fixture builder, the public/private subset split, and the per-release refresh step are the
  implementation this spike unblocks — a follow-up to #95, gated on the first post-freeze window
  existing (i.e. after a v0.3 release).

## Reproducibility

`v7_followup_competition_proof.py` is stdlib + the installed package, deterministic (no RNG, fixed
synthetic data), and reuses the shipped `maneuver_detect.leaderboard` service and `benchmark` scorer —
so the integrity properties are properties of the code the competition Space would run, not of a mock.
Run it with `python docs/design/spikes/v7_followup_competition_proof.py` (or `uv run`); two
invocations produce byte-identical output. The per-class forward-yield assessment (Part B) is an
estimate from public operator cadences — reproducible as reasoning, not as a benchmark; the realized
counts land when the first forward window is reconstructed at the follow-up build.

## Caveats / open items

- **The proof is synthetic by necessity.** Real elset epochs are derived Space-Track data the project
  does not redistribute (D2), so the dry-run uses fictional catalogue ids — exactly as the V7 proof
  did. The firewall it demonstrates is a property of the partition and the scorer, not of the
  particular labels.
- **GEO/IGSO depth is the residual risk, not GEO existence.** The operator feeds make a forward
  GEO/IGSO holdout real, but thin; a release-cycle window may yield only a few labels each, so early
  GEO/IGSO ranks carry wide intervals. This is surfaced honestly (Part B), not hidden.
- **Sockpuppet accounts** weaken the per-user rate limit (the known public-leaderboard limitation);
  the public/private split, not the rate limit, is what actually protects the ranking — and it holds
  regardless of how many accounts probe the public subset (proof §[4]–[5]).
- **The freeze must be a real release freeze.** The first forward window cannot be reconstructed until
  a public dataset version is frozen to define `epoch > freeze`; the implementation is therefore gated
  on a release, as noted in D16.
- Ratify D16 when the competition board is implemented against it (the V7 → D12 discipline).

## References

- [`v7-leaderboard-integrity-and-compute-budget.md`](v7-leaderboard-integrity-and-compute-budget.md)
  — the hidden-label firewall and probing bound this follow-up re-applies (its analysis was retained
  for exactly this track).
- [`../decisions.md`](../decisions.md) — **D12** (leaderboard integrity) + its amendment (why v0.2 is
  reproducibility-mode), **D13/D15** (the label sources, including the GEO/IGSO operator feeds), and
  **D16** (this track); D2 (recipe-first, no redistribution).
- [`../benchmark-protocol.md`](../benchmark-protocol.md) §8 — the submission / held-out-label /
  rate-limiting contract this track extends.
- [`v2-followup2-heo-igso-sources.md`](v2-followup2-heo-igso-sources.md) — the QZSS OHI / NOAA GOES
  GEO/IGSO sources (D15) that de-risk the forward GEO holdout, and the BeiDou-NABU dead end.
- `maneuver_detect.leaderboard.service` / `.fixture`, `benchmark.scoring` — the real service and
  deterministic scorer the proof reuses unchanged.

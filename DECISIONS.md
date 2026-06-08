# DECISIONS — quick reference

One-glance digest of every choice made. Full rationale, evidence and journal live in
`Strategy.md`; this file is the scannable index so nothing gets lost. Each row points to the
`DEC-N` entry in `Strategy.md §5`.

## Pipeline at a glance

```
image
 └─ DETECTION (Hough circles)  ──────────────────────────────┐
     ├─ easy coins: clean Hough (minDist≈95, param2≈44)        │ the circle is also the
     └─ weak-rim coins: overshoot (CLAHE+median, 85/42)        │ required brief annotation
        → GHT-confidence gate (drop margin < 0.08)             │
 └─ CLASSIFICATION (partial, geometry/number-first)            │
     ├─ number: fair GHT on in-circle edges → the digit        │
     ├─ cent-vs-euro: chroma-ring bi-metal flag (+ size)       │
     └─ size: intra-image radius ratios (needs a scale anchor) │
 └─ OUTPUT: Coin N {value}, Partial Amount, Total Amount ──────┘
```

> **Note (DEC-15, 2026-05-30):** the *classification* row above is the earlier **GHT-digit** line; the
> **operating pipeline is now §11.3 bis** — whole-disc shape match-% + bi-metal colour favour + relative
> size + per-coin localized reference blur. Detection and output are unchanged; the GHT-digit is parked as
> an alternative line.

## Best results so far (and which decisions they depend on)

| # | Result | Best number (reproducible) | Depends on | Validation |
|---|--------|----------------------------|------------|------------|
| 1 | **Detection — easy coins** | exact count on the 6-image showcase (4/3/3/4/1/1), **0 background FP** | DEC-4 (Gaussian pre-smooth), **DEC-5** (Hough, mid `minDist≈95`) | showcase + manual |
| 2 | **Detection — full target set** | 142 imgs, **228 coins**, ~1.6/img; **32/142 (23 %) zero** = weak-rim | DEC-5 | `all_smoke.ipynb` |
| 3 | **Weak-rim recovery** | **~21/28 (75 %) at ~3 FP** (overshoot → GHT-confidence gate); shape gate ~18–20/28 · 3–4 FP | DEC-6 (no global levers), **DEC-13** / DEC-8 | 8-weak + 6-easy harness |
| 4 | **FP self-identification** | real coins GHT-margin **0.10–0.14**, FP/duplicates **0.01–0.03** (flat profile) | **DEC-13** | image_85 + harness |
| 5 | **Number / digit (classification)** | **4/8 denomination, 5/8 digit-level, top-2 in 6/8** (fair GHT) — *8-coin GT, small* | DEC-10/11 (context), **DEC-12** | hand-read GT (8) |
| 6 | **Bi-metal flag (cent↔euro)** | refs 1€ +0.53 / 2€ +0.65 vs all mono ≤ +0.11; flags **both euros in image_85** | DEC-3/DEC-7 (cast), **DEC-14** | 8 refs + image_85 |

**The two strongest, defensible results are #1 (detection) and #6 (bi-metal flag).** #3/#4 are solid
partial wins; #5 is real but on a tiny GT (treat as promising, not proven).

## How we got here (the path)

The method throughout: **falsify approaches empirically on real data, keep only what survives.** Many
strong-looking ideas were rejected; two of our own beliefs were corrected mid-way.

1. **M0/M1 inspection** killed the two "obvious" cues up front: absolute radius is not metric (DEC-1)
   and colour carries a non-removable blue cast (DEC-3).
2. **Detection** first leaned LoG/DoG (it found a low-contrast coin Hough missed). The **full smoke
   test reversed this**: LoG misses bright coins and floods dark backgrounds → switched to **Hough**
   (DEC-5), with Gaussian (not bilateral) pre-smoothing (DEC-4).
3. The smoke test also surfaced the **weak-rim failure** (23 % zero). Every *global* sensitivity lever
   couples recall and FP (DEC-6), so we moved to **overshoot + per-candidate gate**: shape-based gate
   first (DEC-8), then the **GHT-confidence gate** (DEC-13).
4. **Classification looked dead.** Colour collapsed under the cast; SIFT star-model failed (DEC-9); a
   whole family of matchers (ZNCC, shape, edge-NCC, chamfer) scored ≈ random, so we closed appearance
   classification (DEC-10/11).
5. **Two corrections reopened it.** (a) The "national-side cap" was a **misread of rotated digits** —
   the number is visible on ~all coins. (b) "Why always 2cent?" exposed that the GHT test was **broken**
   (uneven template sizes + a sparsity-biased score); a **fair GHT** (uniform-density templates,
   size-normalized) recovers the digit cue (DEC-12).
6. **Colour came back, in a targeted way.** Absolute hue stays dead, but the **chroma-gradient ring**
   at ρ≈0.70 survives the cast and gives a clean **bi-metal flag** (DEC-14) — the cent-vs-euro
   discriminator the digit needs.
7. **Honesty check.** A headline "23/28 · 2 FP" gate turned out to be a fragile preprocessing corner;
   a controlled sweep showed the FP swings 2–7 and that gate vs classifier want opposite template
   sharpness. The recorded numbers are the reproducible ones (DEC-13).

## Decisions

| ID | Choice | One-line reason |
|----|--------|-----------------|
| DEC-1 | Use **intra-image relative radius** + per-image scale anchor; drop absolute radius | abs. radius not metric (camera distance varies ~2×) |
| DEC-2 | **No background** scale anchor | plain backgrounds, no ruler/hand/A4 |
| DEC-3 | **Colour eliminated** as a cross-set classification cue | strong blue specular cast, not WB-removable |
| DEC-4 | Pre-Canny smoothing = **Gaussian σ≈2** (not bilateral) | bilateral keeps bg noise → floods Hough |
| DEC-5 | Detector = **Hough circles** (mid `minDist≈95`); LoG/DoG dropped | rim is polarity-agnostic; LoG misses bright coins |
| DEC-6 | **Reject global sensitivity levers** for weak-rim | recall and FP are coupled globally |
| DEC-7 | Bi-metal colour **sign discarded** (cast inverts it); use \|b* step\| magnitude | 24-coin scan gave an impossible 20×2€ / 4×1€ |
| DEC-8 | *(alternative to DEC-13)* weak-rim = overshoot → **shape-based gate** (§6.2) | 18–20/28 · 3–4 FP, beat ZNCC/rim-score |
| DEC-9 | **Reject the §6.5 SIFT star model** as detector/recall-booster | A=0/28, B=1/13; cast starves SIFT |
| DEC-10 | **Reject appearance classification** (texture + edge), go geometry-first | SIFT 1/13, edge-shape ≈0.02 margin |
| DEC-11 | *(partly superseded by DEC-12)* close appearance classification | whole matcher family ≈ random |
| DEC-12 | **Reopen the number cue: a *fair* GHT works** | uniform-density templates + size-norm score → 4/8 (random 1/8) |
| DEC-13 | Weak-rim = overshoot (85/42) → **GHT-confidence gate** (`margin ≥ 0.08`, sharp templates) | ~21/28 · 3 FP (≈ shape gate; param-sensitive 2–7 FP); reuses classifier, FP have flat GHT profile |
| DEC-14 | Bi-metal flag = **chroma-gradient ring** at ρ≈0.70 (not grayscale) | chroma discontinuity survives the cast; refs 0.53/0.65 vs mono ≤0.11 |
| DEC-15 | **Operating pipeline = §11.3 bis** (`scale_playground`): Hough → whole-disc **shape match-%** + material/bi-metal **colour favour** + relative **size** (order `W_ORDER11=0.25` / same `W_SAME11=0.15`) + per-coin **localized reference blur** (`BLUR3b`) | most-parametrized self-contained framework; fuses the 3 surviving cues with per-coin reference control. Sets aside the GHT-digit (DEC-12, different line). Limits (§11/§12): shape match-% low/close (~40–50%, cf. DEC-10), scale circular, colour bi-metal-only (§12 **falsified** relative bronzo/oro), blur breaks ref/tgt symmetry |

## What works / what doesn't (on the cast targets)

**Works**
- **Detection** via Hough (easy coins exact; weak-rim via overshoot + confidence gate, ~21/28 · 3 FP, on par with the shape gate).
- **Number/digit** via a *fair* GHT (uniform-density edge templates, size-normalized, rotation-searched): ~4/8, top-2 in 6/8.
- **Bi-metal flag** via the chroma-gradient ring → reliably isolates 1€/2€ (cent-vs-euro discriminator).
- **FP filter** = GHT confidence (margin ≈ 0 on spurious/duplicate circles).

**Doesn't work**
- Absolute colour / colour-family (cast collapses it, even relatively).
- Holistic appearance matchers: SIFT (raw 1/13, enh 0/13), ZNCC-enh (margin ~0.05), edge-NCC / chamfer (0/8).
- Rim-score as a weak-rim gate (kills the weak coins it should keep).
- Absolute radius as a cross-image size cue (distance varies ~2×).
- The "national-side cap" — **was a misread of rotated digits**; the number is visible on ~all coins (confirmed by the user: no coin without a number).
- **Negative (255−img) and R↔B swap** — cosmetic relabelings; **no effect on detection** (confirmed visually, playground v17). Useful for the eye, not the algorithm.

## Root causes (why classification is hard)
1. **Blue specular cast** (DEC-3/7) — kills colour, corrupts intensity.
2. **Dilution** — the discriminative digit is ~10–20 % of a common-side design that is *identical across denominations* (map + stars); holistic matchers drown in the shared part.
3. **Wear + low resolution** (r ≈ 50–110 px) — the relief is subtle; the digit cannot be cleanly segmented.
4. Reading the digit needs human/learned gestalt OCR — **forbidden** by the brief; hand-crafted CV recovers it only partially (fair GHT).

## Committed direction
- **Detection + counting:** solid (Hough + confidence gate). Each coin gets a circle (brief annotation).
- **Classification — operating pipeline = §11.3 bis (DEC-15):** Hough → whole-disc **shape match-%** + **bi-metal colour** favour + **relative size** (order/same) + per-coin **localized reference blur**. The decision is carried by the **fusion** — whole-disc shape alone is weak (~40–50%, scores close).
- **Set aside / documented limits:** the **GHT-digit** (DEC-12) is parked as an *alternative line* (it segments the number; §11.3 bis scores the whole disc instead); **bronzo/oro colour does not transfer** (§12, falsified) — colour helps **only** as the bi-metal gate; the **absolute scale anchor** (`s=median(r/DIAM[pred0])` is seed-circular) is the real residual bottleneck. Per the brief, a justified partial > blind-complete.

## Open items
- **Per-image scale anchor** (§4.2/§4.4): bi-metal known diameter + constellation fit — needed for the size half. *Not yet built.*
- **Larger hand-read GT** to validate GHT-digit beyond 8 coins.
- **Over-detection** on dense/touching clusters (dedup / `minDist`).
- Fold the validated detection + partial classifier into `solution.ipynb` (still on the old LoG demo).

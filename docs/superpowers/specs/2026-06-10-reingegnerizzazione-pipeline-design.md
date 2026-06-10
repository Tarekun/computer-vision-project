# Re-ingegnerizzazione pipeline monete — Design

**Data:** 2026-06-10 · **Stato:** approvato dall'utente (sezioni 1–9 validate in brainstorming)
**Target:** coin-level accuracy ≥ 0.85 (checkpoint di realismo a fine Fase 0)
**Campione da battere:** e12_size_constraints — coin-acc 0.63, exact 0.40, MAE 0.243, count 0.92

## 0. Contesto e motivazione

L'approccio incrementale (16 esperimenti `fable/`) è stato dichiarato inefficiente: detection allucinata (miss, FP, centri imprecisi) e classification appena sufficiente (0.63). I cue misurati come più robusti (colore relativo, taglia relativa) sono oggi *correttivi*, mentre il cue primario (shape matching su 8 riferimenti puliti contro target degradati) è quello che si rompe sempre (5 esperimenti falliti sul muro del rumore: e7, e7b, e8, e9, e10).

**Premessa di design:** TUTTI i cue sono deboli e possono allucinare (famiglia-colore 0.75, segno bimetallo prec 0.68, scala a cascata). Quindi: nessun cue decide da solo, astensione obbligatoria, fusione per-scena, riduzione del degrado a monte.

## 1. Architettura (approccio B "constraint-first" + fase 0 di A "measurement-first")

Pipeline runtime per immagine:

1. **Prefilter** (§3) — trasformazione scelta misurando su GT; rami separati ammessi per edges (detection) e croma (misure colore), dichiarati.
2. **Detection a due stadi** (§4, §6.3) — Hough *demansionato* a generatore di candidati (soglie permissive, multi-pass) + **validatore geometrico** per candidato (riuso e4): bordi radiali selezionati per-raggio, edge-support, fit Kasa per ricentrare. Accetta/rigetta su edge-support.
3. **Rettifica prospettica** (§2) — fit ellisse (cv2.fitEllipse) sui punti di bordo del validatore; se eccentricità sopra soglia fisica → unwarp anisotropo a cerchio PRIMA di ogni misura. Raggio per la taglia = semiasse maggiore. Scene multi-moneta: omografia di scena unica votata da tutte le ellissi; fit per-moneta come fallback per singole. Soglia minima di punti per attivare il fit (fallback: disco com'è).
4. **Misure per-moneta** (nessuna decisione): tono relativo ancorato al background (catena e13), step+segno anello/nucleo, raggio rettificato, σ blur, shape match.
5. **Decisione per-scena a vincoli (DP)** — estensione di e12 a famiglia×taglia, voti soft + vincoli fisici duri.
6. **Output** formato assignment (valore per ogni moneta, Partial, Total) + confidenza per-moneta.

**Conformità alla traccia (vincolo emerso in brainstorming):** il reference_set "serve to build and calibrate your classifier" → OGNI cue è calibrato dai ref, non solo lo shape: prototipi colore dai ref, segno bimetallico dai 2 ref bimetallici, griglia diametri dai ref se condividono scala (da verificare, Q1) altrimenti costanti fisiche euro dichiarate, template shape dagli 8 ref. Lo shape resta sempre attivo come termine di conferma (soft-add stile e6) e decisore pieno sui nodi ambigui — mai spento.

## 2. Fase 0 — GT posizionale + error budget

- **GT posizionale** etichettato da Claude guardando crop + immagine intera; per ogni moneta vera `(cx, cy, r, label, sure)`; `label ∈ {8 token, "unknown"}`. `unknown` = illeggibile anche per un umano: esce dal denominatore coin-acc, resta nel count (politica dichiarata nel notebook).
- **Revisione utente sui pixel, non sul JSON**: `v2/gt_overlay.py` (base: `fable/make_overlays.py`) renderizza tutte le 142 immagini con cerchi GT etichettati (verde=sure, giallo=dubbio, rosso=unknown) + **contact sheet dei soli dubbi** (griglia di crop). L'utente corregge guardando gli overlay; GT congelato solo dopo validazione visiva.
- **Error budget su e12**: ogni errore attribuito al primo stadio che lo causa: detection (miss/FP/centro) → famiglia → taglia → shape/tie-break. Output: tabella "dove vivono i 37 punti mancanti" (100 − 63 di coin-acc del campione e12; per lo 0.85 ne vanno recuperati almeno 22).
- **Affidabilità per-cue misurata** (famiglia-colore, segno bimetallico, purezza cluster same-size, shape top-1 entro famiglia) — serve a GIUSTIFICARE l'ordinamento dei pesi, NON a fittarli.

**Anti-overfitting procedurale (non è ML):** il GT serve SOLO a misurare. Calibrazione = reference_set + fisica. Pesi della fusione: pochi, semplici, motivati qualitativamente, con **analisi di sensibilità** nel notebook (robustezza in un intorno, stile plateau W_DIGIT e6). Split fisso del GT: **metà sviluppo (pari) / metà sigillata (dispari)** — la sigillata si apre solo a fine progetto per il numero finale.

**Checkpoint realismo:** se illeggibili+detection costano da soli >10 punti, lo 0.85 si rinegozia con i numeri in mano.

## 3. Fase 1 — Degrado, detection, rettifica

- **Prefilter misurato**: candidate da `fable/legibility` (raw, NLM, bilaterale, mediana, ricetta e11 = stretch percentile 2–98 + NLM adattivo) valutate su metriche di detection + stabilità misure colore.
- **Detection generatore+validatore** come da §1 punto 2. I tre difetti noti (FP, miss, centri) attaccati insieme: FP cadono per edge-support basso, centri corretti da Kasa, weak-rim veri sopravvivono alla soglia permissiva.
- **Rettifica** come da §1 punto 3.
- **Criterio d'uscita:** miss+FP dimezzati vs detector attuale, errore mediano centro ≤ 2px (metà sviluppo). FP sopravvissuti devono comunque convincere famiglia+taglia nel DP (doppio sbarramento indipendente).

## 4. Fase 2 — Classificatore a fusione di vincoli

Misure per-moneta su disco rettificato e filtrato; ogni cue → distribuzione soft sugli 8 tagli (o famiglie) + astensione (= distribuzione uniforme: tace, non vota contro):

| cue | misura | calibrato da | si astiene se |
|---|---|---|---|
| famiglia | dev. tono (a*,b*) disco interno vs anello background, dopo stretch L* (e13) | prototipi rame/oro dai ref | cast forte (indice e5) |
| bimetallo | ampiezza + segno step croma nucleo→anello | 2 ref bimetallici | step < rumore scena (MAD) |
| taglia | raggio rettificato (semiasse maggiore, Kasa) | griglia diametri (ref o costanti euro) | edge-support basso (e4) |
| shape | match-% orientazione gradienti whole-disc + digit soft-add (e6, W≈0.3) | 8 ref | mai (sempre attivo, peso moderato) |

**DP per-scena** (estensione e12): massimizza somma voti soft, vincoli DURI solo fisici: ordine diametri reali inviolabile; cluster same-size (tol 0.030 su log-raggio, giustificata dal rumore misurato |Δr| p50≈1px) → stessa etichetta. Scala: gerarchia flag-bimetallo-confermato > picco Hough (e3, LOG_BIN=0.01) > fallback circolare.

**Cambio strutturale vs m2/e12: il bimetallo NON è più gate duro ma voto forte nel DP.** Corregge il difetto noto (falso flag image_85 coin3 incondannabile dal gate). Rischio dichiarato: perdita della garanzia MAE; ripiego pronto se l'eval lo mostra: gate solo sull'ancora di scala, mai sull'argmax.

**Scene singole (punto debole dichiarato):** bimetallo → segno; mono → famiglia colore + shape/digit dentro la terna. Niente trucchi nuovi; attenuanti: rettifica+prefilter migliorano il matching; l'error budget dirà se le mono singole dominano il gap → in tal caso si apre il dossier template-extra dal dataset (PRIMA verifica con traccia/docenti — Q6).

**Output:** sempre argmax (l'assignment esige un valore per moneta); astensione solo interna; confidenza per-moneta riportata.

## 5. Validazione, robustezza, integrazione

- **Protocollo:** eval estesa al GT posizionale (matching pred↔GT per IoU cerchi > 0.5): detection (miss/FP/centro/raggio) + classification (coin-acc/exact/MAE) + attribuzione per-stadio. Confronto sempre vs e12. Smoke set solo per iterare, mai per congelare (lezione e7: margini smoke ×6, held-out −13 punti).
- **Fallback runtime dichiarati:** validatore rigetta tutto → migliori candidati Hough flaggati low-conf; fit ellisse instabile → niente rettifica; zero monete → `0 coin(s) found`, Partial 0. Mai crash, mai eccezioni silenziose.
- **Notebook: `solution_2.ipynb` NUOVO, affiancato a `solution.ipynb`** (confronto diretto vecchia/nuova pipeline; il vecchio resta intatto). Struttura a specchio della pipeline: per stadio (1) giustificazione ancorata al capitolo Summary, (2) numero prima/dopo su GT sviluppo, (3) visualizzazione su 2-3 immagini. Più tabella sensibilità pesi e sezione *honest assessment*. Prosa/commenti in inglese, stile lab.

## 6. Componenti e layout

```
Project/
├── v2/
│   ├── prefilter.py   # img → img filtrata (rami edges / colore)
│   ├── detect.py      # img → [candidati (cx,cy,r)] Hough permissivo
│   ├── validate.py    # candidato → (accept, cx', cy', r', edge_support, ellisse)
│   ├── rectify.py     # disco + ellisse [+ omografia scena] → disco rettificato
│   ├── cues.py        # disco → {famiglia, bimetallo(segno), taglia, shape} + astensioni
│   ├── scene.py       # [misure] → assegnazione DP + confidenze
│   ├── refs.py        # reference_set → prototipi (tono, segno, griglia, template)
│   ├── evalpos.py     # eval GT posizionale (detection + classification + budget)
│   ├── gt_overlay.py  # gt_pos.json → overlays + contact sheet dubbi
│   └── gt_pos.json    # GT posizionale (Fase 0)
├── solution_2.ipynb   # nuovo notebook (importa v2/, prosa + viz)
└── fable/             # invariato (storico + e12 di confronto)
```

Ogni modulo gira standalone (`python -m` su singola immagine) per debug visivo. Il notebook racconta/chiama/visualizza, non duplica. Travaso eventuale in celle: meccanico, in Fase 4 (dipende da Q5). Python: `"/home/gianmarco/envs/IP&CV/bin/python"`. Solo cv2/numpy/matplotlib — **niente skimage**.

## 7. Formati dati

GT posizionale:
```json
{"image_N.jpg": {"coins": [{"cx":424,"cy":477,"r":85,"label":"5cent","sure":true}],
                 "tilt":"none|mild|strong", "note":""}}
```
Output esteso (retrocompatibile con fable/eval.py):
```json
{"image_N.jpg": [{"i":0,"cx":424,"cy":477,"r":85,"pred":"5cent","value":0.05,
                  "conf":0.71,"cues":{"fam":"copper","fam_abst":false,"bimetal":-1}}]}
```

## 8. Milestone

| # | milestone | dipende da | criterio di uscita |
|---|---|---|---|
| M0a | GT posizionale 142 img + overlays + contact sheet dubbi | — | lista dubbi pronta per revisione utente |
| M0b | Error budget e12 + affidabilità per-cue | M0a | tabella errori per stadio; checkpoint 0.85 |
| M1a | Prefilter misurato | M0a | scelta motivata da numeri |
| M1b | Detection generatore+validatore | M1a | miss+FP dimezzati, centro ≤2px (sviluppo) |
| M1c | Rettifica prospettica | M1b | errore raggio % ridotto su tilt≠none, no regressioni su piatte |
| M2 | Cues + DP famiglia×taglia | M1c | batte e12 su coin-acc E MAE (sviluppo) |
| M3 | Tie-break mono singole + sensibilità pesi | M2 | tabella sensibilità; decisione dossier template |
| M4 | solution_2.ipynb + apertura metà sigillata | M3 | numero finale; confronto con solution.ipynb |

Ogni milestone = cartella `fable/`-style con `run.py` + `NOTES.md`. Niente codice M2 prima della pubblicazione di M0b (l'error budget può ribaltare le priorità).

## 9. Questioni aperte (si chiudono in Fase 0)

1. I ref condividono la scala? → misura raggi 8 ref vs griglia reale (10 min). Se sì: griglia dai ref.
2. Verso del segno anello/nucleo → misurato sui 2 ref bimetallici (chiude BCE vs memoria utente).
3. Frequenza del tilt → classi `tilt` dal GT; dimensiona M1c.
4. Politica `unknown` → se >10% delle monete, decidere prima del checkpoint (proposta: fuori denominatore, dichiarato).
5. Cosa si consegna (solo notebook?) → determina il travaso in Fase 4.
6. Template extra dal dataset → congelato finché M0b non mostra che le mono singole dominano; poi prima domanda ai docenti.

## Vincoli permanenti

Solo CV tradizionale (vincolo traccia, ribadito due volte in rosso). Solo cv2/numpy/matplotlib, NO skimage fino a nuovo ordine. Ogni scelta ancorata a Summary §2–§6. Trade-off accuratezza/spiegabilità a favore della spiegabilità (criteri d'esame). GT fatto a mano = solo misura, mai calibrazione.

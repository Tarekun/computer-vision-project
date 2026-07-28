caric# Report tecnico — Le tre aggiunte vincenti dell'ablation

## Contesto

Baseline *from-scratch* (ResNet-like, stem a 3 conv 3x3, blocchi `[2,2,2,2]`, average
pooling, testa lineare singola) su FGVC-Aircraft: **20.31% val**. Su 100 varianti di
velivoli con sole 3334 immagini di training il problema e' *fine-grained*: le classi
differiscono per dettagli locali minuscoli (numero di motori, forma del muso, winglet,
livrea) su una silhouette quasi identica. Le tre aggiunte che hanno funzionato attaccano
**tre stadi diversi e ortogonali** della pipeline, ed e' per questo che si sommano invece
di sovrapporsi.

---

## 1. `wider` — canali piu' larghi (32-64-128-256 -> 48-96-192-384)

**Effetto isolato:** 20.31% -> **25.68%** (+5.37) · 2.8M -> 6.4M param

**Meccanismo.** Aumenta la capacita' rappresentativa del *backbone*: piu' filtri per stage
= piu' pattern locali codificabili in parallelo. La scelta *allargare invece di
approfondire* e' supportata dai dati: `wider` +5.4 contro `deeper` (+1 blocco per stage)
appena +1.0.

**Paper di riferimento — Wide Residual Networks** (Zagoruyko & Komodakis, BMVC 2016,
arXiv:1605.07146). Tesi centrale: aumentare la **larghezza** anziche' la profondita' da'
miglior accuratezza-per-parametro ed e' piu' facile da addestrare, *specialmente su
dataset piccoli*, dove le reti thin-deep tendono a sotto-addestrarsi o overfittare. Il
nostro risultato (width >> depth con 3334 immagini) e' una replica pulita di questa tesi.

---

## 2. `gem` — Generalized Mean Pooling (il miglior rapporto impatto/costo)

**Effetto isolato:** 20.31% -> **26.25%** (+5.94) · **con un solo parametro in piu'**
(2.839.300 -> 2.839.301)

**Meccanismo.** Sostituisce il global average pooling con la **media generalizzata**:

    f_k = ( (1/|X_k|) * sum_{x in X_k} x^p )^(1/p)

Il parametro `p` e' **apprendibile** (nel codice: init 3.0, clampato in [1, 8]). I due
estremi sono casi noti: **p = 1 -> average pooling**, **p -> inf -> max pooling**. Con
p > 1 il pooling *enfatizza le attivazioni piu' forti* (le regioni salienti: il motore, la
coda) senza scartare del tutto il contesto come farebbe il max puro. E' un "focus morbido"
sulle parti discriminanti, esattamente cio' che serve nel fine-grained, dove l'average
diluisce il segnale utile nella massa di pixel di sfondo/fusoliera comune a tutte le classi.

**Paper di riferimento — "Fine-tuning CNN Image Retrieval with No Human Annotation"**
(Radenovic, Tolias, Chum, IEEE TPAMI 2019, arXiv:1711.02512), che introduce il GeM pooling
per l'image retrieval fine-grained. La lezione trasferita qui: aggregare le feature map con
una media generalizzata apprendibile cattura meglio i dettagli localizzati rispetto
all'average pooling. (Radici teoriche del pooling anche in Boureau et al., *A theoretical
analysis of feature pooling in visual recognition*, ICML 2010.)

Che dia +5.9 punti al costo di **un singolo scalare** e' il risultato piu' elegante dello
studio.

---

## 3. `mlp_head` — testa di classificazione non lineare (il vincitore singolo)

**Effetto isolato:** 20.31% -> **31.74%** (+11.43) · 2.84M -> 3.0M param

**Meccanismo.** Sostituisce la testa `Dropout -> Linear(384, 100)` con una testa a collo di
bottiglia non lineare:

    Linear(384 -> 512) -> BatchNorm1d(512) -> ReLU -> Dropout(0.4) -> Linear(512 -> 100)

Una testa lineare singola impone un confine di decisione **lineare** nello spazio delle
feature: su dati fine-grained le classi sono vicine e non linearmente separabili, quindi e'
un collo di bottiglia. Il layer nascosto + ReLU permette confini **non lineari**; la
BatchNorm stabilizza e accelera la convergenza della testa; il Dropout(0.4) regolarizza la
capacita' aggiunta contro l'overfitting sulle 3334 immagini.

**Paper di riferimento.**
- **BatchNorm** — Ioffe & Szegedy, *Batch Normalization*, ICML 2015 (arXiv:1502.03167):
  normalizzare le attivazioni del layer nascosto stabilizza il training e agisce da
  regolarizzatore.
- **Dropout** — Srivastava, Hinton et al., *Dropout: A Simple Way to Prevent Neural
  Networks from Overfitting*, JMLR 2014.
- **Testa MLP non lineare** — l'evidenza piu' diretta viene da **SimCLR** (Chen et al.,
  *A Simple Framework for Contrastive Learning of Visual Representations*, ICML 2020,
  arXiv:2002.05709): una *projection head* MLP non lineare supera nettamente una testa
  lineare. Sebbene il contesto sia il representation learning, il principio e' lo stesso:
  una testa non lineare aggiunge capacita' decisionale e lascia alle feature del penultimo
  layer una rappresentazione piu' ricca da sfruttare.

---

## L'interazione: perche' 53% e non 32%

| Configurazione | Val acc |
|---|---|
| baseline | 20.31% |
| miglior componente singolo (mlp_head) | 31.74% |
| **`11_combo_core` = wider + gem + mlp_head** | **52.96%** |

La combinazione da' **+21 punti sul miglior singolo**: un effetto **super-additivo** (la
somma ingenua dei tre delta isolati sarebbe ~+23 sul baseline ~= 43%; il risultato reale e'
~53%, quindi si rinforzano oltre l'additivita').

**Il motivo e' l'ortogonalita'.** Ognuno migliora uno stadio diverso e complementare:

    [ backbone piu' ricco ] -> [ aggregazione migliore ] -> [ classificatore migliore ]
           wider                       gem                        mlp_head
       piu' feature discri-      le concentra sulle          traccia confini non
       minanti disponibili      regioni salienti            lineari tra classi vicine

Feature migliori sono inutili se il pooling le diluisce; un pooling migliore e' sprecato se
la testa lineare non sa separarle. Insieme, la catena non ha piu' anelli deboli. E' l'esatto
opposto di `10_combined` (25.80%), che diluiva i vincitori con i **perdenti** (`se` -3.0,
`multiscale` -3.7) e per giunta *ometteva* l'mlp_head: aggiungere componenti che
interferiscono peggiora, aggiungere componenti ortogonali compone.

---

## Tabella ablation completa (validation accuracy)

| # | Variante | Val acc | Delta vs baseline | Param | Verdetto |
|---|---|---|---|---|---|
| 11 | **combo_core (wider+gem+mlp_head)** | **52.96%** | **+32.65** | 6.58M | vincitore |
| 08 | mlp_head | 31.74% | +11.43 | 3.0M | vincitore singolo |
| 07 | gem | 26.25% | +5.94 | 2.84M | forte (+1 param) |
| 10 | combined (tutto) | 25.80% | +5.49 | 7.9M | deludente |
| 03 | wider | 25.68% | +5.37 | 6.4M | forte |
| 04 | deeper | 21.33% | +1.02 | 3.5M | marginale |
| 06 | dilated_stage4 | 21.33% | +1.02 | 2.84M | marginale |
| 00 | baseline | 20.31% | — | 2.84M | riferimento |
| 01 | preact | 20.07% | -0.24 | 2.84M | neutro |
| 02 | se | 17.34% | -2.97 | 2.86M | peggiora |
| 09 | multiscale | 16.62% | -3.69 | 2.85M | peggiora |
| 05 | delayed_downsampling | 13.74% | -6.57 | 2.84M | peggiora molto |

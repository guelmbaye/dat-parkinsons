# DaT Parkinson's Challenge — pipeline complet

Classification normal / anormal d'examens DaTscan (SPECT au transporteur de la
dopamine), challenge SFMN hebergé par DrivenData. Metrique de classement :
**log loss**. Soumission par **execution de code** (`submission.zip` avec
`main.py` a la racine, A100 80 Go, 24 vCPU, 3 h, **sans reseau**).

---

## 1. Ce que fait le pipeline

```
NIfTI heterogenes ──► pretraitement ──► ┬─► 110 descripteurs ─► LightGBM
  (tailles et                           │                     └─► logistique
   resolutions                          └─► volume 64x64x40 ──► ResNet3D
   variables)                                                 └─► Proj2D (ImageNet)
                                                                     │
                                          fusion logit + Platt + ecretage
                                                                     ▼
                                                            submission.csv
```

### Pretraitement (`src/datscan/imaging.py`)

C'est la piece la plus importante : elle absorbe l'heterogeneite multicentrique
qui est explicitement le sujet du challenge.

1. **Orientation** — `as_closest_canonical` ramene tout en RAS+. Les centres
   n'exportent pas tous dans la meme convention ; sans cette etape, un modele
   appris sur une orientation ne transfere pas.
2. **Resolution** — reechantillonnage a 2 mm isotrope depuis le spacing du
   header (2,46 mm et 3,895 mm coexistent dans le jeu).
3. **Intensite** — division par le **fond non specifique**, estime par moyenne
   tronquee p25-p85 des voxels intra-tete. Le volume sort en unites de type
   SBR : `6.0` = captation six fois le fond. C'est l'equivalent sans atlas de
   la normalisation occipitale clinique, et cela supprime l'effet dose, duree
   d'acquisition et sensibilite du scanner.
4. **Geometrie** — plan striatal par coupe axiale de captation maximale, lacet
   par axe principal de l'ellipse cranienne, **ligne mediane par maximisation
   de symetrie** (et non par barycentre : en denervation unilaterale severe le
   barycentre derive vers le cote sain). Rotation + translation + recadrage
   128x128x80 mm en **une seule interpolation affine**.

Environ 0,7 s/examen sur un vCPU faible, donc quelques minutes sur 24 cœurs.

### Descripteurs (`src/datscan/features.py`)

110 grandeurs semi-quantitatives : SBR au pic, VOI iso-contour 50 %, rapport
putamen/caude, pente antero-posterieure, etendues, asymetries, profil de forme.
Ce sont les criteres de lecture clinique.

**Toutes les grandeurs bilaterales sont reduites en min / max / moyenne /
asymetrie.** Le cote atteint est arbitraire d'un patient a l'autre : les
descripteurs sont donc invariants par echange gauche-droite, et le modele n'a
aucune raison d'apprendre qu'« un hemisphere est special ».

### Validation (`src/datscan/cv.py`)

Validation croisee **groupee par pseudo-centre** (signature matrice +
resolution). Une CV aleatoire place deux examens du meme scanner de part et
d'autre du pli et surestime largement la performance. Le score groupe repond a
la vraie question : que vaut ce modele sur un scanner jamais vu.

La colonne `pseudo_center` sert **uniquement au decoupage**. Elle n'est jamais
donnee au modele : ce serait le meilleur moyen d'apprendre la prevalence par
centre, qui ne se transporte pas au test prive.

### Calibration (`src/datscan/blend.py`)

La log loss recompense autant la calibration que la discrimination.

1. fusion lineaire des logits, poids contraints positifs, optimisee en log loss ;
2. Platt (temperature + biais) — la fusion a poids normalises n'a pas de degre
   de liberte en echelle, le Platt le fournit ;
3. **ecretage prudent** : plancher lie a l'effectif (~1/4n) et, a performance
   quasi egale, on retient l'ecretage le plus large. Une prediction a 1e-6
   fausse coute 14 nats ; sur 1000 examens, cela ajoute 0,014 a la moyenne.

Tout est ajuste **hors-pli**, et `04_blend.py` affiche en plus une
**estimation croisee de la calibration** — c'est ce chiffre qu'il faut suivre
d'une iteration a l'autre, pas la log loss brute qui est optimiste.

---

## 2. Commandes

```bash
# 0. auto-test du code reseau (30 s, a faire AVANT tout entrainement long)
python tools/selftest_cnn.py

# 1. pretraitement (une fois ; le cache est relu par tous les entrainements)
python scripts/01_preprocess.py --data data --out cache --workers 24

# 2. modeles sur descripteurs : quelques secondes, reference forte et calibree
python scripts/02_train_tab.py --cache cache --out models/tab

# 3. reseaux (regler --epochs d'apres curve.csv)
python scripts/03_train_cnn.py --cache cache --out models/resnet3d \
    --model resnet3d --epochs 60 --batch-size 16
python scripts/03_train_cnn.py --cache cache --out models/proj2d \
    --model proj2d --backbone convnext_tiny --epochs 30 --batch-size 32 --lr 1e-4

# 4. fusion + calibration
python scripts/04_blend.py --out models/blend --manifest cache/manifest.csv \
    --oof gbm=models/tab/oof_gbm.csv \
    --oof linear=models/tab/oof_linear.csv \
    --oof resnet3d=models/resnet3d/oof.csv \
    --oof proj2d=models/proj2d/oof.csv

# 5. archive de soumission
python scripts/05_pack.py --tab models/tab --blend models/blend \
    --cnn resnet3d=models/resnet3d --cnn proj2d=models/proj2d --out dist

# 6. test local dans le conteneur officiel
cp dist/submission.zip <runtime-repo>/submission/submission.zip
cd <runtime-repo> && just test-submission
```

### Colab

`notebooks/colab_dat_parkinsons.ipynb` reprend tout le parcours, adapte aux
contraintes de la plateforme : versions epinglees sur celles du `uv.lock`
officiel, cache et poids sur Drive, reprise pli par pli apres deconnexion,
precision mixte choisie selon le GPU (bf16 sur A100/L4, fp16 sur T4), et
repetition de l'inference avec mesure de debit.

Sans donnees reelles sous la main, tout se rejoue sur des fantomes :

```bash
python tools/make_phantom.py --out data_phantom --n 400
python scripts/01_preprocess.py --data data_phantom --out cache --workers 8
```

---

## 3. Robustesse de l'inference (`submission_src/main.py`)

- **Une soumission sort toujours.** Chaque etage est protege : si le CNN tombe,
  les modeles sur descripteurs prennent le relais et les poids de la recette
  sont renormalises ; si tout tombe, la prevalence d'entrainement est ecrite.
  Un job en erreur a trois jours de la cloture coute bien plus qu'un modele
  moyen.
- **Budget temps surveille.** Un etage couteux est abandonne *avant* d'engager
  du temps qu'on n'a pas, pas apres.
- **Aucune exception ne remonte du pretraitement** : un examen illisible sort
  en volume nul avec `qc.ok = False`.
- **Meme module de pretraitement qu'a l'entrainement** — pas de
  reimplementation, donc pas de divergence train/test.

Verifie par simulation complete du conteneur (unzip, `DATA_DIR`, execution,
`submission.csv` a l'ordre et au format exacts).

---

## 4. Poids pre-entraines

Le conteneur n'a **pas de reseau**. Pour `proj2d`, les poids ImageNet doivent
etre telecharges a l'entrainement (`--pretrained 1`) et voyagent ensuite dans
les `fold*.pt` du zip. Verifier la licence du backbone : `convnext_tiny` et les
`resnet` timm sont sous licence permissive, ce qui satisfait la clause
« external data with rights » du reglement, et la clause MIT imposee aux
solutions gagnantes.

---

## 5. Etat de validation

| Composant | Statut |
|---|---|
| `imaging.py` | execute sur 400 fantomes multicentriques ; signe de rotation verifie par test de symetrie |
| `features.py` | 110 descripteurs, 0 NaN / 0 inf, separation nette sur fantomes |
| `models_tab.py` | LightGBM + logistique, hors-pli honnete (nombre d'arbres fige avant scoring) |
| `blend.py` | fusion + Platt + ecretage, avec estimation croisee |
| `main.py` | simulation complete du conteneur, sortie au format exact |
| `models_cnn.py` | **ecrit mais non execute** — lancer `tools/selftest_cnn.py` en premier |
| notebook Colab | JSON et cellules Python valides ; non execute sur Colab |

Les scores obtenus sur fantomes (log loss ~0,04) ne disent **rien** de la
performance reelle : les fantomes sont separables par construction. Ils
valident le code, pas le modele.

---

## 6. Plan sur la semaine restante

| Jour | Action |
|---|---|
| 1 | `selftest_cnn.py`, puis pretraitement complet. Inspecter `manifest.csv` : pseudo-centres, `ok=False`, `peak_ratio` aberrants. |
| 1 | `02_train_tab.py` + `04_blend.py` + `05_pack.py` → **premiere soumission le jour 1**. Elle fixe la reference et valide le format. |
| 2 | `03_train_cnn.py --model resnet3d`. Regler `--epochs` sur `curve.csv`. Soumettre la fusion. |
| 3 | `proj2d` avec backbone pre-entraine. Comparer les gains par pseudo-centre. |
| 4 | Diversite : deuxieme graine, fenetre de recadrage plus large, backbone alternatif. C'est ce qui paie le plus en log loss. |
| 5 | Regler `--shrink` (0,05-0,10) et l'ecretage sur l'estimation croisee. |
| 6 | Smoke test, puis soumission complete. Garder une marge : le budget est de 3 h et la file est partagee. |
| 7 | Selection finale. Preferer la soumission la plus robuste par pseudo-centre, pas la meilleure au tableau public. |

Deux erreurs a ne pas commettre : soumettre pour la premiere fois le dernier
jour, et choisir la soumission finale sur le tableau public alors que le
classement se joue sur un test prive de composition inconnue.

---

## 7. Licence

MIT — conforme a la clause « Open source license » imposee aux solutions
gagnantes.

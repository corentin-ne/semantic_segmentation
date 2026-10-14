# Semantic Segmentation — Pascal VOC 2012

Projet ENSC de **Corentin Nivelet** et **Thomas Renaud** (nov.–déc. 2023), refondu en v2 :
package Python propre, modèles plus précis *et* plus légers, pipeline d'entraînement optimisé,
dashboard interactif pour suivre les trainings et explorer les résultats.

> Le code de 2023 (`DeconvMobileNet`, `DeconvMobileUNet`, notebook Colab) est conservé tel quel
> dans [`legacy/`](legacy/). `DeconvMobileNet` existe aussi, nettoyé, dans le nouveau registre
> (`deconv_mobilenet`) pour servir de point de comparaison.

## Démarrage rapide

```bash
pip install -r requirements.txt

# démo hors-ligne (dataset synthétique, CPU, quelques minutes)
python -m semseg train -c configs/synthetic_demo.yaml

# Pascal VOC (GPU recommandé) — VOC est téléchargé dans ./data
python -m semseg train -c configs/deeplabv3plus_mobilenet.yaml

# dashboard
streamlit run dashboard/app.py -- --runs runs

# dashboard sur les runs de démo fournis (sans rien entraîner)
streamlit run dashboard/app.py -- --runs docs/demo_runs
```

Sur Colab : [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb).

## Modèles

| Nom | Architecture | Rôle |
|---|---|---|
| `deeplabv3plus_mobilenet` | DeepLabV3+ (OS16), MobileNetV3-Large, ASPP séparable | **recommandé** : meilleur compromis précision / coût |
| `mobile_unet` | U-Net MobileNetV3-Large, décodeur séparable, gate de contexte global | successeur de `DeconvMobileUNet` |
| `lraspp_mobilenet` | LR-ASPP MobileNetV3-Large (torchvision) | le plus rapide (temps réel / embarqué) |
| `deeplabv3plus_resnet50` | DeepLabV3+ ResNet-50 | référence précision |
| `deeplabv3_mobilenet`, `deeplabv3_resnet50` | torchvision | baselines |
| `deconv_mobilenet` | baseline 2023 (MobileNetV3-Small + déconvolutions, sans skip) | comparaison |

`python -m semseg models` liste les modèles, et `python -m semseg benchmark` mesure les paramètres,
les GFLOPs et la latence (résultats affichés dans l'onglet *Comparaison* du dashboard).

Les modèles torchvision acceptent `model.pretrained: coco_voc` : poids COCO restreints aux 20
classes VOC, le meilleur point de départ pour un fine-tuning court
(voir `configs/lraspp_mobilenet_coco.yaml`).

## Mesures

Benchmark CPU (4 threads, image 512×512, batch 1, PyTorch 2.14), `python -m semseg benchmark` :

| Modèle | Params | GFLOPs | Latence CPU |
|---|---:|---:|---:|
| `lraspp_mobilenet` | 3.2 M | 3.9 | 47 ms |
| `mobile_unet` | 3.3 M | 3.7 | 64 ms |
| `deeplabv3_mobilenet` | 11.0 M | 19.7 | 68 ms |
| `deeplabv3plus_mobilenet` | 3.7 M | 6.4 | 77 ms |
| `deconv_mobilenet` (2023) | 12.6 M | 24.6 | 85 ms |
| `deeplabv3plus_resnet50` | 26.7 M | 75.4 | 291 ms |
| `deeplabv3_resnet50` | 39.6 M | 327.2 | 804 ms |

Entraînement de démo sur le dataset synthétique (`configs/synthetic_demo.yaml`) : CPU, sans poids
pré-entraînés, 12 époques, 1 200 images. Les runs sont dans [`docs/demo_runs/`](docs/demo_runs/).

| Modèle | mIoU val | Pixel acc |
|---|---:|---:|
| `lraspp_mobilenet` | 72.5 % | 96.9 % |
| `mobile_unet` | 46.2 % | 93.1 % |
| `deeplabv3plus_mobilenet` | 42.1 % | 92.8 % |
| `deconv_mobilenet` (2023) | 23.2 % | 90.2 % |

Ces chiffres valident le pipeline, ce ne sont pas des scores VOC. Les scores VOC s'obtiennent
avec les configs `configs/*.yaml` sur GPU, avec les poids ImageNet ou COCO.

## Ce qui a changé par rapport à 2023

**Justesse**
- Les bords d'objets VOC (255) étaient comptés comme *background*. Ils sont maintenant ignorés
  (`ignore_index`) dans la loss et les métriques.
- L'early stopping s'arrêtait sur la *baisse* du mIoU (`mIoU < best`). Il porte maintenant sur sa hausse.
- La validation se fait à la résolution native, avec du padding ignoré au lieu d'un resize en
  224×224, et en batch (avant, la taille de batch de 1 était imposée).
- Le mIoU est calculé sur une matrice de confusion (un seul `bincount` par batch au lieu d'une
  boucle Python sur les classes). On obtient aussi pixel acc, mean acc, fwIoU, mDice et l'IoU par classe.
- Les BatchNorm des MobileNetV3 de torchvision (momentum 0.01) sont ramenées à 0.1, sinon les
  prédictions en mode `eval` s'effondrent en fine-tuning court.

**Précision**
- Architectures modernes : DeepLabV3+ avec ASPP et features bas niveau, U-Net à skips multi-échelles,
  upsampling bilinéaire (fini les masques en escalier).
- Augmentations façon DeepLab : échelle aléatoire 0.5–2×, crop 512, flip, color jitter.
- Split `trainaug` (VOC + SBD, 10 582 images au lieu de 1 464).
- Loss OHEM ou CE + Dice, AdamW avec LR différencié encodeur / décodeur, warmup + poly, EMA des poids.
- TTA (flip + multi-échelle) à l'inférence.

**Optimisation**
- Mixed precision (bf16 / fp16), channels-last, `torch.compile` optionnel, AdamW fused, cudnn benchmark.
- Convolutions séparables en profondeur dans les décodeurs : environ 8× moins de FLOPs qu'une 3×3 dense.
- DataLoader avec workers persistants, prefetch, pin memory et seeds reproductibles.
- Export ONNX avec batch et résolution dynamiques.

**Outillage**
- Configs YAML surchargeables en CLI (`-o optim.lr=1e-3 train.epochs=10`).
- Checkpoints `best` / `last` et reprise d'entraînement (`train.resume=…/last.pt`).
- Run directory en JSON/PNG, lisible par le dashboard pendant l'entraînement ; W&B en option.
- Tests unitaires et de bout en bout (`pytest`), dataset synthétique pour tourner sans VOC.

## CLI

```bash
python -m semseg train     -c configs/mobile_unet.yaml -o train.epochs=30 data.train_split=train
python -m semseg evaluate  --run runs/<run>
python -m semseg predict   --run runs/<run> --image photo.jpg --tta
python -m semseg export    --run runs/<run>            # -> best.onnx
python -m semseg benchmark --device cuda --output runs/benchmark_gpu.json
```

## Dashboard

`streamlit run dashboard/app.py -- --runs runs`

- **Vue d'ensemble** : tous les runs (statut live, meilleur mIoU, coût).
- **Entraînement** : courbes par époque et loss par itération, rafraîchies automatiquement pendant le training.
- **Résultats** : IoU par classe, matrice de confusion, prédictions de validation époque par époque.
- **Comparaison** : précision vs GFLOPs, benchmarks de latence.
- **Démo** : segmente ta propre image avec n'importe quel checkpoint (overlay, confiance, TTA).

## Structure

```
semseg/
  config.py            dataclasses + YAML + overrides CLI
  data/                VOC (+SBD), dataset synthétique, transforms image/masque
  models/              registre, DeepLabV3+, MobileUNet, wrappers torchvision, baseline 2023
  engine/              trainer, évaluation, EMA, optimiseur/scheduler, run logger
  losses.py metrics.py inference.py benchmark.py export.py visualize.py cli.py
configs/               recettes prêtes à l'emploi
dashboard/app.py       interface Streamlit
tests/                 pytest
legacy/                code d'origine (2023)
```

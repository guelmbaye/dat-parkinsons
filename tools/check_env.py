"""Verifie que l'environnement local peut produire des artefacts rechargeables
dans le conteneur d'evaluation.

    python tools/check_env.py

Un `requirements.txt` dit ce qu'il FAUT installer ; ce script dit ce qui EST
installe et si l'ecart est grave. La distinction compte, parce qu'un
`pip install` qui reussit peut quand meme laisser une version divergente
(paquet deja present, resolution de conflit, image Colab preinstallee).

La gravite depend de ce que le paquet fait traverser la frontiere :

* CRITIQUE — un artefact serialise par ce paquet doit se recharger dans le
  conteneur. Une divergence peut faire echouer la soumission a l'execution,
  c'est-a-dire au pire moment.
* IMPORTANT — influence le resultat numerique sans casser le chargement.
* INFO — aucun artefact ne transite, la version importe peu.
"""

from __future__ import annotations

import importlib.metadata as md
import sys

# (paquet, version du conteneur, gravite, raison)
PINS = [
    ("scikit-learn", "1.8.0", "CRITIQUE",
     "le pipeline logistique voyage en pickle"),
    ("timm", "1.0.27", "CRITIQUE",
     "l'architecture proj2d doit etre reconstructible"),
    ("torch", "2.12.1", "CRITIQUE",
     "chargement des state_dict des CNN"),
    ("lightgbm", "4.6.0", "IMPORTANT",
     "modele serialise en texte, donc portable"),
    ("nibabel", "5.4.2", "IMPORTANT",
     "lecture des headers NIfTI"),
    ("scipy", "1.17.1", "IMPORTANT",
     "interpolation du pretraitement"),
    ("scikit-image", "0.26.0", "INFO", ""),
    ("numpy", "2.2.6", "INFO", ""),
    ("pandas", "3.0.3", "INFO", ""),
]
REQUIRED = {"numpy", "scipy", "pandas", "scikit-learn", "lightgbm", "nibabel"}


def base(v: str) -> str:
    """Ignore le suffixe local : 2.12.1+cpu et 2.12.1+cu129 sont compatibles."""
    return v.split("+")[0]


def main() -> None:
    print(f"python {sys.version.split()[0]}"
          f"{'  <- identique au conteneur' if sys.version_info[:2] == (3, 12) else '  <- le conteneur utilise 3.12'}\n")
    print(f"{'paquet':15} {'installe':16} {'conteneur':12} {'etat'}")
    print("-" * 68)

    blocking, warnings = [], []
    for name, want, severity, why in PINS:
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            state = "ABSENT" + ("  <- requis" if name in REQUIRED else "  (optionnel)")
            if name in REQUIRED:
                blocking.append(f"{name} absent")
            print(f"{name:15} {'-':16} {want:12} {state}")
            continue

        if base(have) == want:
            state = "ok"
        elif severity == "CRITIQUE":
            state = f"DIVERGENT ({severity})"
            blocking.append(f"{name} {have} != {want} — {why}")
        elif severity == "IMPORTANT":
            state = f"divergent ({severity.lower()})"
            warnings.append(f"{name} {have} != {want} — {why}")
        else:
            state = "divergent (sans effet)"
        print(f"{name:15} {have:16} {want:12} {state}")

    print()
    if blocking:
        print("A CORRIGER avant de produire une soumission :")
        for b in blocking:
            print(f"  - {b}")
        print("\n  pip install -r requirements.txt")
        print("  # puis, si les CNN sont utilises :")
        print("  pip install -r requirements-cnn-cpu.txt   # machine sans GPU")
        print("  pip install -r requirements-colab.txt     # Colab, sans torch")
    if warnings:
        print("\nEcarts a surveiller (sans blocage) :")
        for w in warnings:
            print(f"  - {w}")
    if not blocking and not warnings:
        print("Environnement aligne sur le conteneur d'evaluation.")

    print("\nRappel : une divergence CRITIQUE ne se voit pas a l'entrainement. "
          "Elle apparait\na l'execution dans le conteneur, quand il est trop "
          "tard pour corriger.")
    sys.exit(1 if blocking else 0)


if __name__ == "__main__":
    main()

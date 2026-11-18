"""Command-line helpers.   python manage.py <command>

  init            create tables, train the first model, load demo data (safe to run again)
  demo            load demo patients      clear-demo   remove them
  train FILE.csv  train a new model version from a labelled CSV (not activated automatically)
  create-key NAME create an API key (printed once)
"""
import sys

import pandas as pd

import core
import ml
import seed
from app import create_app, init_data


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    app = create_app()
    cmd = argv[0]
    if cmd == "init":
        init_data(app)
    elif cmd in ("demo", "clear-demo"):
        with app.app_context():
            core.ensure_model()
            print(f"{seed.load_demo()} assessments created" if cmd == "demo" else (seed.clear_demo() or "demo data removed"))
    elif cmd == "train" and len(argv) > 1:
        with app.app_context():
            df = ml.clean_frame(pd.read_csv(argv[1], na_values=["?", ""]).rename(columns=str.lower))
            mv = core.train_and_register(df, f"CLI: {argv[1]}")
            print(f"trained {mv.name} (not active - activate it in Admin > Models)")
    elif cmd == "create-key" and len(argv) > 1:
        import api
        with app.app_context():
            print("API key (shown once):", api.create_key(" ".join(argv[1:])))
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

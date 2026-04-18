"""
Exports all trained artifacts as static files into docs/ (GitHub Pages root).
Run after train_all.py completes.

Copies:
  - webapp/templates/index.html      → docs/index.html  (with API calls rewritten to static JSON)
  - webapp/static/                   → docs/static/
  - saved_models/metrics.json        → docs/data/metrics.json
  - data/processed/shap_data.json    → docs/data/shap_data.json
  - data/processed/fraud_rings.json  → docs/data/fraud_rings.json

The static index.html fetches from /data/*.json instead of /api/*.
"""

import json
import shutil
import yaml
from pathlib import Path

ROOT   = Path(__file__).parent
DOCS   = ROOT / "docs"
DATA   = DOCS / "data"


def load_config():
    with open(ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

    # Clean & recreate docs/
    if DOCS.exists():
        shutil.rmtree(DOCS)
    DOCS.mkdir()
    DATA.mkdir()

    # Copy static assets
    shutil.copytree(ROOT / "webapp" / "static", DOCS / "static")

    # Copy data JSON files
    files = {
        cfg["model"]["metrics_path"]:                      DATA / "metrics.json",
        cfg["data"]["processed_dir"] + "/shap_data.json":  DATA / "shap_data.json",
        cfg["data"]["fraud_rings_file"]:                   DATA / "fraud_rings.json",
    }
    for src, dst in files.items():
        src_path = Path(src)
        if src_path.exists():
            shutil.copy(src_path, dst)
            print(f"Copied {src_path.name} → {dst.relative_to(ROOT)}")
        else:
            print(f"WARNING: {src_path} not found — run train_all.py first")

    # Rewrite index.html: replace /api/* with /data/*.json
    template_candidates = [
        ROOT / "webapp" / "templates" / "index.html",
        ROOT / "index.html",
    ]
    html_path = next((path for path in template_candidates if path.exists()), None)
    if html_path is None:
        candidates = "\n".join(f"  - {path}" for path in template_candidates)
        raise FileNotFoundError(
            "Static export could not find an HTML template. Checked:\n"
            f"{candidates}\n"
            "Commit webapp/templates/index.html to the repository so CI can export the site."
        )

    html = html_path.read_text()
    html = html.replace("src='/static/", "src='/FraudSense/static/")
    html = html.replace('src="/static/', 'src="/FraudSense/static/')
    html = html.replace("href='/static/", "href='/FraudSense/static/")
    html = html.replace('href="/static/', 'href="/FraudSense/static/')
    (DOCS / "index.html").write_text(html)

    # Write static-mode JS (replaces /api/* calls with /data/*.json)
    static_js_path = DOCS / "static" / "js" / "config.js"
    static_js_path.write_text("""
// Static mode — fetch from pre-generated JSON files instead of FastAPI
window.FRAUDSENSE_STATIC = true;
window.FRAUDSENSE_BASE   = '/FraudSense';
""")

    print(f"\nStatic site exported to docs/")
    print("Push to GitHub → Settings → Pages → Source: main /docs")


if __name__ == "__main__":
    main()

# Deploy & Run from GitHub

You have three easy options:

## 1) GitHub Codespaces (one click, no setup)
- Push this folder to a GitHub repo.
- Click **Code → Codespaces → Create codespace**.
- Wait for the container to build (uses `.devcontainer/devcontainer.json`).
- In the Codespace terminal run:
  ```bash
  streamlit run kmz_altitude_app.py
  ```
- A forwarded port (8501) will open in your browser.

## 2) Streamlit Community Cloud (zero infra)
- Push to GitHub.
- Go to https://share.streamlit.io → **New app** → pick your repo and `kmz_altitude_app.py` as the entry.
- (Optional) If `rasterio` fails on Streamlit Cloud, rely on the default OpenTopoData API path,
  or switch to the Docker deployment below.

## 3) Docker + GitHub Container Registry (portable)
- With the included `Dockerfile`, you can build locally or via GitHub Actions:

### Build locally
```bash
docker build -t kmz-altitude:latest .
docker run --rm -p 8501:8501 kmz-altitude:latest
```

### Build via GitHub Actions to GHCR
- Enable **Actions** in your repo.
- The workflow `.github/workflows/build-and-publish.yml` publishes `ghcr.io/<OWNER>/kmz-altitude:latest` on push to `main`.
- Run it, then deploy the image on Render/Railway/Fly.io/etc.

## Notes
- If you only need the **OpenTopoData API** (no local GeoTIFF uploads), everything should work out of the box.
- For heavier **rasterio** usage, Docker is the most reliable path.

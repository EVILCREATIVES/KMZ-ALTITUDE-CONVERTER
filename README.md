# KMZ Altitude Injector (Streamlit)

A simple web app to add altitudes to KML/KMZ geometries using either:
- OpenTopoData SRTM30m API (default)
- A local GeoTIFF DEM you upload (via rasterio)

## Features
- Supports KML inside KMZ (doc.kml) or plain KML uploads
- Updates altitudes for Points, LineStrings, LinearRings (Polygons), and gx:Track
- Sets `<altitudeMode>` to `absolute` (default) or `relativeToGround`
- Optional offset in meters
- Repackages into a new KMZ, keeping original non-KML assets (icons, overlays)

## Install
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
streamlit run kmz_altitude_app.py
```
Open the local URL Streamlit prints (usually http://localhost:8501).

## Notes
- OpenTopoData is rate-limited and may be slow for very large files—consider using a local GeoTIFF DEM for speed.
- Altitudes are written in meters.
- `absolute` means Google Earth will use your numbers directly. If your goal is “height above ground,” use `relativeToGround` and set **Offset** to your desired height.

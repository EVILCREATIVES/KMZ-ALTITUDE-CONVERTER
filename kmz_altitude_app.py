\
import io
import math
import os
import re
import tempfile
import zipfile
from typing import List, Tuple, Iterable

import requests
import streamlit as st
from lxml import etree

# -----------------------------
# Helpers
# -----------------------------

KML_NS = "http://www.opengis.net/kml/2.2"
GX_NS = "http://www.google.com/kml/ext/2.2"
NSMAP = {"kml": KML_NS, "gx": GX_NS}

def unzip_kmz_to_kml_bytes(kmz_bytes: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(kmz_bytes)) as zf:
        # Find the first .kml in the archive (often doc.kml)
        for name in zf.namelist():
            if name.lower().endswith(".kml"):
                return zf.read(name)
    raise ValueError("KMZ archive does not contain a .kml file.")

def rezip_kml_to_kmz_bytes(kml_bytes: bytes, original_kmz: bytes = None) -> bytes:
    # Rebuild a minimal KMZ (doc.kml at root). If original_kmz was provided, we keep other assets (icons, overlays).
    out_buf = io.BytesIO()
    with zipfile.ZipFile(out_buf, "w", compression=zipfile.ZIP_DEFLATED) as z_out:
        if original_kmz:
            with zipfile.ZipFile(io.BytesIO(original_kmz)) as z_in:
                # Copy everything EXCEPT any existing .kml; we'll write our updated one as doc.kml
                for name in z_in.namelist():
                    if not name.lower().endswith(".kml"):
                        z_out.writestr(name, z_in.read(name))
        # Write our updated KML
        z_out.writestr("doc.kml", kml_bytes)
    return out_buf.getvalue()

def parse_coords_text(text: str) -> List[Tuple[float, float, float | None]]:
    """
    Parse KML coordinate string: "lon,lat[,alt] lon,lat[,alt] ..."
    Returns list of (lon, lat, alt or None)
    """
    coords = []
    if not text:
        return coords
    # Coordinates may be separated by spaces, newlines, or tabs
    for token in re.split(r"\s+", text.strip()):
        if not token:
            continue
        parts = token.split(",")
        if len(parts) < 2:
            continue
        lon = float(parts[0])
        lat = float(parts[1])
        alt = float(parts[2]) if len(parts) > 2 and parts[2] != "" else None
        coords.append((lon, lat, alt))
    return coords

def format_coords_text(coords: List[Tuple[float, float, float | None]]) -> str:
    toks = []
    for lon, lat, alt in coords:
        if alt is None:
            toks.append(f"{lon:.8f},{lat:.8f}")
        else:
            toks.append(f"{lon:.8f},{lat:.8f},{alt:.3f}")
    # KML usually allows spaces/newlines; we keep it compact
    return " ".join(toks)

# -----------------------------
# Elevation samplers
# -----------------------------

def sample_elevations_opentopodata(points: List[Tuple[float, float]]) -> List[float | None]:
    """
    Query OpenTopoData (SRTM 30m by default). Batch in chunks of up to ~100 points.
    API: https://api.opentopodata.org/v1/srtm30m?locations=lat,lon|lat,lon
    Returns list of elevations in meters (WGS84 geoid-ish) or None if failed.
    """
    base_url = "https://api.opentopodata.org/v1/srtm30m"
    out: List[float | None] = []
    chunk_size = 90  # conservative
    for i in range(0, len(points), chunk_size):
        chunk = points[i:i+chunk_size]
        locs = "|".join([f"{lat},{lon}" for lon, lat in chunk])
        try:
            r = requests.get(base_url, params={"locations": locs}, timeout=30)
            r.raise_for_status()
            data = r.json()
            if "results" in data:
                for rec in data["results"]:
                    elev = rec.get("elevation", None)
                    out.append(float(elev) if elev is not None else None)
            else:
                out.extend([None]*len(chunk))
        except Exception:
            out.extend([None]*len(chunk))
    return out

def sample_elevations_rasterio(points: List[Tuple[float, float]], tif_bytes: bytes) -> List[float | None]:
    """
    Sample elevations from an uploaded GeoTIFF using rasterio.
    """
    import rasterio
    with tempfile.NamedTemporaryFile(suffix=".tif") as tf:
        tf.write(tif_bytes)
        tf.flush()
        with rasterio.open(tf.name) as src:
            # rasterio.sample expects iterable of (x, y) = (lon, lat)
            samples = list(src.sample([(lon, lat) for lon, lat in points]))
            # Each sample is an array with one band; handle nodata
            out = []
            for arr in samples:
                val = float(arr[0])
                if src.nodata is not None and math.isclose(val, src.nodata):
                    out.append(None)
                else:
                    out.append(val)
            return out

# -----------------------------
# KML elevation injection
# -----------------------------

def set_altitude_mode(elem: etree._Element, mode: str):
    """
    Ensure <altitudeMode> exists and equals one of: 'absolute', 'relativeToGround', 'clampToGround'
    """
    # Remove gx:altitudeMode if present (to avoid conflicts) and set standard altitudeMode
    for gx_mode in elem.xpath(".//gx:altitudeMode", namespaces=NSMAP):
        gx_parent = gx_mode.getparent()
        if gx_parent is not None:
            gx_parent.remove(gx_mode)
    # Find or create altitudeMode under the geometry parent (Point/LineString/LinearRing/Polygon)
    am_elems = elem.xpath(".//kml:altitudeMode", namespaces=NSMAP)
    if am_elems:
        am = am_elems[0]
        am.text = mode
    else:
        # Best-effort: append to first geometry child
        geom = None
        for tag in ("Point", "LineString", "LinearRing", "Polygon"):
            found = elem.find(f".//{{{KML_NS}}}{tag}")
            if found is not None:
                geom = found
                break
        if geom is None:
            # If no geometry, nothing to do
            return
        am = etree.Element(f"{{{KML_NS}}}altitudeMode")
        am.text = mode
        geom.append(am)

def inject_altitudes_into_coords_text(text: str, sampler, offset_m: float = 0.0) -> str:
    coords = parse_coords_text(text)
    if not coords:
        return text
    pts_lonlat = [(lon, lat) for lon, lat, _ in coords]
    elevs = sampler(pts_lonlat)
    out = []
    for (lon, lat, _old_alt), elev in zip(coords, elevs):
        if elev is None:
            out.append((lon, lat, None))  # leave as is
        else:
            out.append((lon, lat, float(elev) + float(offset_m)))
    return format_coords_text(out)

def process_kml(kml_bytes: bytes, mode: str, offset_m: float, sampler) -> bytes:
    """
    Walk the KML and update altitudes for:
      - kml:Point/kml:coordinates
      - kml:LineString/kml:coordinates
      - kml:LinearRing (Polygon boundaries)/kml:coordinates
      - gx:Track/gx:coord
    Set altitudeMode accordingly.
    """
    parser = etree.XMLParser(remove_blank_text=False, ns_clean=True, recover=True, encoding="utf-8")
    root = etree.fromstring(kml_bytes, parser)

    # Points, LineStrings, LinearRings coordinates
    coord_nodes = root.xpath(".//kml:Point/kml:coordinates | .//kml:LineString/kml:coordinates | .//kml:LinearRing/kml:coordinates", namespaces=NSMAP)
    for node in coord_nodes:
        parent_geom = node.getparent().getparent() if node.getparent() is not None else None
        # Update coordinates
        node.text = inject_altitudes_into_coords_text(node.text or "", sampler, offset_m=offset_m)
        # Set altitudeMode on the Placemark or parent geometry
        if parent_geom is not None:
            set_altitude_mode(parent_geom, mode)

    # gx:Track coords are in <gx:coord> elements with "lon lat alt" triplets (space-separated)
    gx_coord_nodes = root.xpath(".//gx:Track/gx:coord", namespaces=NSMAP)
    if gx_coord_nodes:
        # Collect all coords, sample in one go for efficiency per Track
        # But simpler: process per node
        for gx in gx_coord_nodes:
            raw = gx.text.strip() if gx.text else ""
            if not raw:
                continue
            parts = raw.split()
            if len(parts) >= 2:
                lon = float(parts[0]); lat = float(parts[1])
                elev = sampler([(lon, lat)])[0]
                if elev is not None:
                    alt = float(elev) + float(offset_m)
                    gx.text = f"{lon:.8f} {lat:.8f} {alt:.3f}"
        # Set altitudeMode for each Track parent
        for track in root.xpath(".//gx:Track", namespaces=NSMAP):
            set_altitude_mode(track, mode)

    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", pretty_print=True)

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="KMZ Altitude Injector", page_icon="🗺️", layout="centered")

st.title("🗺️ KMZ Altitude Injector")
st.caption("Upload a KMZ or KML. We'll add terrain elevations and return a KMZ with altitudes.")

with st.expander("How it works / options", expanded=False):
    st.markdown("""
- **Altitude Mode**:  
  - `absolute` (default) writes real elevations (meters) and sets `<altitudeMode>absolute</altitudeMode>` so Google Earth uses them.  
  - `relativeToGround` writes numbers interpreted as *offset above terrain*.
- **DEM Source**:  
  - *OpenTopoData (SRTM30m)* — free web API (rate-limited). Good for quick runs.  
  - *Local GeoTIFF* — upload a DEM for offline, faster, or higher-res sampling.
- **Offset**: Add/subtract meters to raise or lower the geometry.
""")

col1, col2 = st.columns(2)
mode = col1.selectbox("Altitude mode", ["absolute", "relativeToGround"])
offset_m = col2.number_input("Offset (meters)", value=0.0, step=1.0)

dem_source = st.radio("DEM source", ["OpenTopoData (SRTM30m API)", "Local GeoTIFF (.tif)"], index=0, horizontal=True)
tif_bytes = None
if dem_source == "Local GeoTIFF (.tif)":
    tif_file = st.file_uploader("Upload GeoTIFF DEM", type=["tif", "tiff"], accept_multiple_files=False)
    if tif_file is not None:
        tif_bytes = tif_file.read()

uploaded = st.file_uploader("Upload KMZ or KML", type=["kmz", "kml"], accept_multiple_files=False)

run = st.button("Convert", type="primary", disabled=(uploaded is None or (dem_source.startswith("Local") and tif_bytes is None)))

if run and uploaded is not None:
    try:
        in_name = uploaded.name
        in_bytes = uploaded.read()

        # Determine KML bytes
        if in_name.lower().endswith(".kmz"):
            kml_bytes = unzip_kmz_to_kml_bytes(in_bytes)
            original_kmz_bytes = in_bytes
        else:
            kml_bytes = in_bytes
            original_kmz_bytes = None

        # Choose sampler
        if dem_source.startswith("OpenTopoData"):
            sampler = sample_elevations_opentopodata
        else:
            if tif_bytes is None:
                st.error("Please upload a GeoTIFF DEM.")
                st.stop()
            sampler = lambda pts: sample_elevations_rasterio(pts, tif_bytes)

        st.info("Processing… This can take a bit for large files.")

        out_kml = process_kml(kml_bytes, mode=mode, offset_m=offset_m, sampler=sampler)
        out_kmz = rezip_kml_to_kmz_bytes(out_kml, original_kmz=original_kmz_bytes)

        st.success("Done! Download your file below.")
        st.download_button(
            "⬇️ Download converted KMZ",
            data=out_kmz,
            file_name=(os.path.splitext(in_name)[0] + "_altitudes.kmz"),
            mime="application/vnd.google-earth.kmz",
        )

        with st.expander("View updated KML (preview)", expanded=False):
            st.code(out_kml.decode("utf-8")[:20000], language="xml")

    except Exception as e:
        st.exception(e)
        st.error("Conversion failed. Check the error above and try a smaller file or a different DEM source.")

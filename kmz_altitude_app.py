import io
import math
import os
import re
import tempfile
import zipfile
from typing import List, Tuple, Iterable, Optional

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

def extract_depth_from_name(name: str) -> Optional[float]:
    """
    Extract depth information from placemark names like:
    - "B1c/ 4130-4540'" -> returns average depth (4335.0)
    - "V1-2800'" -> returns 2800.0
    - "Au-3500-3800'" -> returns average (3650.0)
    Returns depth in feet, or None if no depth found
    """
    if not name:
        return None
    
    # Pattern 1: Range format like "4130-4540'" or "3500-3800'"
    range_match = re.search(r'(\d+)-(\d+)[\'"]?\s*$', name)
    if range_match:
        start = float(range_match.group(1))
        end = float(range_match.group(2))
        return (start + end) / 2  # Return average depth
    
    # Pattern 2: Single depth like "2800'" or "V1-2800'"
    single_match = re.search(r'-?(\d+)[\'"]?\s*$', name)
    if single_match:
        return float(single_match.group(1))
    
    # Pattern 3: Depth anywhere in name like "depth:4500ft" or "4500m"
    depth_match = re.search(r'(\d+)(?:ft|m|\'|")', name, re.IGNORECASE)
    if depth_match:
        return float(depth_match.group(1))
    
    return None

def feet_to_meters(feet: float) -> float:
    """Convert feet to meters"""
    return feet * 0.3048

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
    return " ".join(toks)

# -----------------------------
# Elevation samplers
# -----------------------------

def sample_elevations_opentopodata(points: List[Tuple[float, float]]) -> List[float | None]:
    """
    Query OpenTopoData (SRTM 30m by default). Batch in chunks of up to ~100 points.
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
            samples = list(src.sample([(lon, lat) for lon, lat in points]))
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
    
    # Find or create altitudeMode under the geometry parent
    am_elems = elem.xpath(".//kml:altitudeMode", namespaces=NSMAP)
    if am_elems:
        am = am_elems[0]
        am.text = mode
    else:
        # Find geometry to attach altitude mode to
        geom = None
        for tag in ("Point", "LineString", "LinearRing", "Polygon"):
            found = elem.find(f".//{{{KML_NS}}}{tag}")
            if found is not None:
                geom = found
                break
        if geom is not None:
            am = etree.Element(f"{{{KML_NS}}}altitudeMode")
            am.text = mode
            geom.append(am)

def inject_altitudes_into_coords_text(text: str, sampler, offset_m: float = 0.0, depth_override: float = None) -> str:
    """
    Enhanced coordinate injection that can use depth override from name parsing
    """
    coords = parse_coords_text(text)
    if not coords:
        return text
    
    if depth_override is not None:
        # Use depth from name parsing instead of terrain sampling
        out = []
        for lon, lat, _old_alt in coords:
            # Convert depth (positive underground) to negative altitude (below surface)
            new_alt = -abs(depth_override) + offset_m
            out.append((lon, lat, new_alt))
        return format_coords_text(out)
    else:
        # Use original terrain sampling logic
        pts_lonlat = [(lon, lat) for lon, lat, _ in coords]
        elevs = sampler(pts_lonlat)
        out = []
        for (lon, lat, _old_alt), elev in zip(coords, elevs):
            if elev is None:
                out.append((lon, lat, None))
            else:
                out.append((lon, lat, float(elev) + float(offset_m)))
        return format_coords_text(out)

def process_kml(kml_bytes: bytes, mode: str, offset_m: float, sampler, use_depth_from_names: bool = False) -> bytes:
    """
    Enhanced KML processing that can extract depths from placemark names
    """
    parser = etree.XMLParser(remove_blank_text=False, ns_clean=True, recover=True, encoding="utf-8")
    root = etree.fromstring(kml_bytes, parser)

    # Process all placemarks to handle name-based depths
    placemarks = root.xpath(".//kml:Placemark", namespaces=NSMAP)
    
    for placemark in placemarks:
        # Extract depth from name if requested
        depth_override = None
        if use_depth_from_names:
            name_elem = placemark.find(f".//{{{KML_NS}}}name")
            if name_elem is not None and name_elem.text:
                depth_feet = extract_depth_from_name(name_elem.text)
                if depth_feet is not None:
                    depth_override = feet_to_meters(depth_feet)
        
        # Process coordinates within this placemark
        coord_nodes = placemark.xpath(".//kml:Point/kml:coordinates | .//kml:LineString/kml:coordinates | .//kml:LinearRing/kml:coordinates", namespaces=NSMAP)
        
        for node in coord_nodes:
            parent_geom = node.getparent().getparent() if node.getparent() is not None else None
            # Update coordinates with depth override if available
            node.text = inject_altitudes_into_coords_text(
                node.text or "", 
                sampler, 
                offset_m=offset_m,
                depth_override=depth_override
            )
            # Set altitudeMode
            if parent_geom is not None:
                set_altitude_mode(parent_geom, mode)

        # Handle gx:Track coords within this placemark
        gx_coord_nodes = placemark.xpath(".//gx:Track/gx:coord", namespaces=NSMAP)
        for gx in gx_coord_nodes:
            raw = gx.text.strip() if gx.text else ""
            if not raw:
                continue
            parts = raw.split()
            if len(parts) >= 2:
                lon = float(parts[0])
                lat = float(parts[1])
                
                if depth_override is not None:
                    # Use depth from name
                    alt = -abs(depth_override) + offset_m
                    gx.text = f"{lon:.8f} {lat:.8f} {alt:.3f}"
                else:
                    # Use terrain sampling
                    elev = sampler([(lon, lat)])[0]
                    if elev is not None:
                        alt = float(elev) + float(offset_m)
                        gx.text = f"{lon:.8f} {lat:.8f} {alt:.3f}"
        
        # Set altitudeMode for tracks
        for track in placemark.xpath(".//gx:Track", namespaces=NSMAP):
            set_altitude_mode(track, mode)

    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", pretty_print=True)

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Enhanced KMZ Altitude Injector", page_icon="🗺️", layout="centered")

st.title("🗺️ Enhanced KMZ Altitude Injector")
st.caption("Upload a KMZ or KML. Extract depths from names or add terrain elevations.")

with st.expander("How it works / options", expanded=False):
    st.markdown("""
- **Altitude Mode**:  
  - `absolute` writes real elevations and sets `<altitudeMode>absolute</altitudeMode>`
  - `relativeToGround` writes numbers as offset above terrain
- **Depth from Names**: Extract depth info from placemark names like "B1c/ 4130-4540'" or "V1-2800'"
- **DEM Source**: OpenTopoData API or local GeoTIFF for terrain elevation
- **Offset**: Add/subtract meters to adjust final altitude
""")

col1, col2 = st.columns(2)
mode = col1.selectbox("Altitude mode", ["absolute", "relativeToGround"])
offset_m = col2.number_input("Offset (meters)", value=0.0, step=1.0)

# New option for depth extraction
use_depth_from_names = st.checkbox(
    "Extract depths from placemark names", 
    value=True,
    help="Extract depth values from names like 'B1c/ 4130-4540'' and use as underground altitudes"
)

dem_source = st.radio("DEM source", ["OpenTopoData (SRTM30m API)", "Local GeoTIFF (.tif)"], index=0, horizontal=True)
tif_bytes = None
if dem_source == "Local GeoTIFF (.tif)":
    tif_file = st.file_uploader("Upload GeoTIFF DEM", type=["tif", "tiff"], accept_multiple_files=False)
    if tif_file is not None:
        tif_bytes = tif_file.read()

uploaded = st.file_uploader("Upload KMZ or KML", type=["kmz", "kml"], accept_multiple_files=False)

# Show sample of depth extraction if enabled
if uploaded is not None and use_depth_from_names:
    try:
        in_bytes = uploaded.read()
        if uploaded.name.lower().endswith(".kmz"):
            kml_bytes = unzip_kmz_to_kml_bytes(in_bytes)
        else:
            kml_bytes = in_bytes
        
        # Parse and show sample of name extraction
        parser = etree.XMLParser(remove_blank_text=False, ns_clean=True, recover=True, encoding="utf-8")
        root = etree.fromstring(kml_bytes, parser)
        placemarks = root.xpath(".//kml:Placemark", namespaces=NSMAP)
        
        depth_examples = []
        for pm in placemarks[:5]:  # Show first 5 examples
            name_elem = pm.find(f".//{{{KML_NS}}}name")
            if name_elem is not None and name_elem.text:
                depth_feet = extract_depth_from_name(name_elem.text)
                if depth_feet is not None:
                    depth_m = feet_to_meters(depth_feet)
                    depth_examples.append(f"'{name_elem.text}' → {depth_feet:.0f}ft ({depth_m:.1f}m depth)")
        
        if depth_examples:
            st.info("Preview of depth extraction:\n" + "\n".join(depth_examples))
        
        # Reset file position
        uploaded.seek(0)
    except Exception as e:
        st.warning(f"Could not preview depth extraction: {e}")

run_disabled = (uploaded is None or 
                (dem_source.startswith("Local") and tif_bytes is None and not use_depth_from_names))

run = st.button("Convert", type="primary", disabled=run_disabled)

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

        # Choose sampler (only needed if not using depth from names exclusively)
        if dem_source.startswith("OpenTopoData"):
            sampler = sample_elevations_opentopodata
        else:
            if tif_bytes is None and not use_depth_from_names:
                st.error("Please upload a GeoTIFF DEM or enable depth extraction from names.")
                st.stop()
            sampler = lambda pts: sample_elevations_rasterio(pts, tif_bytes) if tif_bytes else lambda pts: [None] * len(pts)

        st.info("Processing… Extracting depths and updating coordinates.")

        out_kml = process_kml(kml_bytes, mode=mode, offset_m=offset_m, sampler=sampler, 
                            use_depth_from_names=use_depth_from_names)
        out_kmz = rezip_kml_to_kmz_bytes(out_kml, original_kmz=original_kmz_bytes)

        st.success("Done! Download your file below.")
        st.download_button(
            "⬇️ Download converted KMZ",
            data=out_kmz,
            file_name=(os.path.splitext(in_name)[0] + "_with_depths.kmz"),
            mime="application/vnd.google-earth.kmz",
        )

        with st.expander("View updated KML (preview)", expanded=False):
            st.code(out_kml.decode("utf-8")[:20000], language="xml")

    except Exception as e:
        st.exception(e)
        st.error("Conversion failed. Check the error above and try again.")

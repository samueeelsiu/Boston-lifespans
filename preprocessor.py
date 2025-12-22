import pandas as pd
import geopandas as gpd
import pyogrio
import numpy as np
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def process_demolition_data(
        gpkg_path='ma_structures_FINAL_with_YR_SOURCE.gpkg',
        zoning_path='Boston_Zoning_Subdistricts.geojson'
):
    """
    Main processing function.
    1. Loads Demolition Data (GPKG).
    2. Loads Zoning Data (GeoJSON).
    3. Performs Spatial Join to link demolitions to zoning districts.
    4. Generates complex JSON structure required by the dashboard.
    """

    print("=" * 50)
    print("STARTING PROCESSING")
    print(f"1. Loading demolition data from {gpkg_path}...")

    # 1. Load Demolition Data (Using Pyogrio for speed)
    # ---------------------------------------------------------
    try:
        gdf = pyogrio.read_dataframe(gpkg_path, use_arrow=True)

        # If geometry is Polygon (building footprints), convert to Centroid (Points)
        if not gdf.empty and gdf.geometry.iloc[0].geom_type != 'Point':
            print("   Converting building polygons to centroids...")
            gdf['geometry'] = gdf.geometry.centroid

    except Exception as e:
        print(f"Error loading GPKG: {e}")
        return None

    # 2. Ensure Coordinate System is Lat/Lon (EPSG:4326)
    # ---------------------------------------------------------
    if gdf.crs and gdf.crs.to_string() != 'EPSG:4326':
        print("   Reprojecting demolition data to EPSG:4326...")
        gdf = gdf.to_crs(epsg=4326)

    # 3. Spatial Join with Zoning Data
    # ---------------------------------------------------------
    print(f"2. Loading zoning data from {zoning_path}...")
    try:
        zoning_gdf = gpd.read_file(zoning_path)

        if zoning_gdf.crs and zoning_gdf.crs.to_string() != 'EPSG:4326':
            zoning_gdf = zoning_gdf.to_crs(epsg=4326)

        print("   Performing spatial join (matching points to districts)...")
        gdf = gpd.sjoin(gdf, zoning_gdf[['geometry', 'Zoning_District', 'Zoning_Subdistrict']], how='left',
                        predicate='within')
        gdf = gdf[~gdf.index.duplicated(keep='first')]
    except Exception as e:
        print(f"   Warning: Could not process zoning data ({e}). Zoning features will be skipped.")
        gdf['Zoning_District'] = None
        gdf['Zoning_Subdistrict'] = None

    # 4. Extract Coordinates and Convert to Standard DataFrame
    # ---------------------------------------------------------
    gdf['LONGITUDE'] = gdf.geometry.x
    gdf['LATITUDE'] = gdf.geometry.y

    # Drop geometry to save memory
    df = pd.DataFrame(gdf.drop(columns='geometry'))
    initial_ma_count = len(df)

    # 5. Data Cleaning & Calculation
    # ---------------------------------------------------------
    print("3. Cleaning and filtering data...")

    # Ensure numeric year_built
    df['year_built'] = pd.to_numeric(df['year_built'], errors='coerce')

    # Handle Dates for Demolition Calculation
    df['DEMOLITION_DATE'] = pd.to_datetime(df['DEMOLITION_DATE'], errors='coerce')
    df['demolition_year'] = df['DEMOLITION_DATE'].dt.year
    # Calculate Lifespan
    df['lifespan'] = df['demolition_year'] - df['year_built']

    # Calculate Current Age for ALL buildings (Current Year - Year Built)
    CURRENT_YEAR = 2025
    df['current_age'] = CURRENT_YEAR - df['year_built']

    # --- KEY CHANGE: CREATE A COPY OF ALL BUILDINGS HERE ---
    print("   Creating snapshot of all buildings (for zoning density stats)...")
    all_buildings_df = df.copy()

    print("   Refining 'Current Buildings' inventory (Subtracting Positive-Lifespan RAZE)...")
    exclude_mask = (all_buildings_df['DEMOLITION_TYPE'] == 'RAZE') & (all_buildings_df['lifespan'] > 0)
    all_buildings_df = all_buildings_df[~exclude_mask]
    print(f"   -> Removed {exclude_mask.sum()} demolished buildings from Current Inventory.")

    # --- FILTERING STEP FOR DEMOLITION DATASET ---
    # This step removes buildings that have NO demolition date or invalid lifespan
    # df now becomes "Demolished Buildings Only"
    df = df[df['lifespan'] < 500]

    # Handle Material and Foundation columns
    if 'material_type_desc' not in df.columns: df['material_type_desc'] = 'Unknown'
    if 'foundation_type' not in df.columns: df['foundation_type'] = 'Unknown'

    # Also handle missing material for the all_buildings_df for accurate filtering later if needed
    if 'material_type_desc' not in all_buildings_df.columns: all_buildings_df['material_type_desc'] = 'Unknown'
    if 'Est GFA sqmeters' not in df.columns:
        print("   Warning: 'Est GFA sqmeters' column not found! Defaulting to 0.")
        df['Est GFA sqmeters'] = 0
        all_buildings_df['Est GFA sqmeters'] = 0

    df['Est GFA sqmeters'] = pd.to_numeric(df['Est GFA sqmeters'], errors='coerce').fillna(0)
    all_buildings_df['Est GFA sqmeters'] = pd.to_numeric(all_buildings_df['Est GFA sqmeters'], errors='coerce').fillna(
        0)

    df['material_group'] = df['material_type_desc'].fillna('Unknown').str.strip()
    all_buildings_df['material_group'] = all_buildings_df['material_type_desc'].fillna('Unknown').str.strip()

    foundation_map = {
        'C': 'Crawl Space',
        'B': 'Basement',
        'S': 'Slab',
        'P': 'Pier',
        'I': 'Pile',
        'F': 'Fill',
        'W': 'Solid Wall'
    }

    df['foundation_group'] = df['foundation_type'].map(foundation_map).fillna(df['foundation_type']).fillna('Unknown')
    all_buildings_df['foundation_group'] = all_buildings_df['foundation_type'].map(foundation_map).fillna(
        all_buildings_df['foundation_type']).fillna('Unknown')

    if 'OCC_CLS' not in df.columns: df['OCC_CLS'] = 'Unknown'
    if 'OCC_CLS' not in all_buildings_df.columns: all_buildings_df['OCC_CLS'] = 'Unknown'

    df['occupancy_group'] = df['OCC_CLS'].fillna('Unknown').str.strip()
    all_buildings_df['occupancy_group'] = all_buildings_df['OCC_CLS'].fillna('Unknown').str.strip()

    print("   Mapping DEMOLITION_STATUS to Open/Close...")
    if 'DEMOLITION_STATUS' in df.columns:
        def simple_status_check(val):
            if pd.isna(val): return 'Open'
            s = str(val).strip().upper()
            if s in ['CLOSED', 'CLOSE']: return 'Close'
            return 'Open'

        df['status_norm'] = df['DEMOLITION_STATUS'].apply(simple_status_check)
    else:
        df['status_norm'] = 'Close'

    # ==========================================
    # GENERATE JSON STRUCTURE
    # ==========================================
    result = {}

    demo_avg = {}
    for dtype in ['RAZE', 'EXTDEM', 'INTDEM']:
        sub = df[df['DEMOLITION_TYPE'] == dtype]
        sub_pos = sub[sub['lifespan'] > 0]
        demo_avg[dtype] = float(sub_pos['lifespan'].mean()) if not sub_pos.empty else 0.0

    result['material_lifespan_demo_avg'] = demo_avg

    # --- A. Summary Stats ---
    boston_df = df[df['Zoning_District'].notna()]
    raze_df = boston_df[boston_df['DEMOLITION_TYPE'] == 'RAZE']
    r_pos = raze_df[raze_df['lifespan'] > 0]
    r_zero = raze_df[raze_df['lifespan'] == 0]
    r_neg = raze_df[raze_df['lifespan'] < 0]

    def get_counts(d):
        return {
            'open': int((d['status_norm'] == 'Open').sum()),
            'close': int((d['status_norm'] == 'Close').sum())
        }

    sb_positive = get_counts(r_pos)
    sb_zero = get_counts(r_zero)
    sb_negative = get_counts(r_neg)
    sb_total = get_counts(raze_df)

    pos_lifespan_df = boston_df[boston_df['lifespan'] > 0]

    # Calculate avg age for cleaned current buildings (same logic as current_age_distribution)
    current_buildings_for_avg = all_buildings_df[all_buildings_df['Zoning_District'].notna()]
    current_buildings_for_avg = current_buildings_for_avg[
        ~((current_buildings_for_avg['DEMOLITION_TYPE'] == 'RAZE') & (current_buildings_for_avg['lifespan'] > 0))]

    result['summary_stats'] = {
        'total_demolitions': int(len(boston_df)),
        'average_lifespan': float(r_pos['lifespan'].mean()) if not r_pos.empty else 0, 
        'raze_count': int(len(r_pos)), \
        'extdem_count': int((boston_df['DEMOLITION_TYPE'] == 'EXTDEM').sum()),
        'intdem_count': int((boston_df['DEMOLITION_TYPE'] == 'INTDEM').sum()),
        'negative_raze_count': sb_negative['close'],
        'zero_raze_count': sb_zero['close'],
        'avg_current_building_age': float(
            current_buildings_for_avg['current_age'].mean()) if not current_buildings_for_avg.empty else 0,
        'raze_status_by_lifespan': {
            'positive': sb_positive, 'zero': sb_zero, 'negative': sb_negative, 'total': sb_total
        }
    }
    result['summary_stats_closed'] = result['summary_stats']

    # --- B. Map Points ---
    print("4. Generating Map Points...")
    map_data = []
    for _, row in df.iterrows():

        if pd.notna(row['LATITUDE']) and row['lifespan'] > 0:
            map_data.append({
                'lat': float(row['LATITUDE']),
                'lng': float(row['LONGITUDE']),
                'type': row['DEMOLITION_TYPE'],
                'lifespan': int(row['lifespan']),
                'status': row['status_norm'],
                'year_built': int(row['year_built']),
                'material': str(row['material_group']),
                'foundation': str(row.get('foundation_type', 'N/A'))
            })
    result['map_points'] = map_data

    # --- C. Zoning District Stats (CORRECTED LOGIC) ---
    print("5. Aggregating Zoning District Stats (Using Full Building Inventory)...")
    zoning_stats = {}

    # We use districts present in ALL buildings, not just demolished ones
    all_districts = all_buildings_df['Zoning_District'].dropna().unique()

    def make_hist(series, width=10):
        if series.empty: return []
        # Filter reasonable age range
        series = series[(series >= 0) & (series < 500)]
        if series.empty: return []
        bins = range(0, int(series.max()) + width, width)
        return [{'range': f"{b}-{b + width}", 'count': int(((series >= b) & (series < b + width)).sum())} for b in bins]

    def make_district_heatmap(d_df, bin_size=20):
        heatmap_counts = {}
        heatmap_gfa = {}
        if d_df.empty: return {'count': {}, 'gfa': {}}

        top_materials = d_df['material_group'].value_counts().head(15).index.tolist()

        for mat in top_materials:
            mat_df = d_df[d_df['material_group'] == mat]
            if len(mat_df) == 0: continue
            bins_c = {}
            bins_g = {}
            for i in range(0, 200, bin_size):
                label = f"{i}-{i + bin_size}"
                mask = (mat_df['lifespan'] >= i) & (mat_df['lifespan'] < i + bin_size)
                subset = mat_df[mask]
                count = len(subset)
                if count > 0:
                    bins_c[label] = int(count)
                    bins_g[label] = int(subset['Est GFA sqmeters'].sum())
            if bins_c:
                heatmap_counts[mat] = bins_c
                heatmap_gfa[mat] = bins_g
        return {'count': heatmap_counts, 'gfa': heatmap_gfa}

    def make_district_foundation_heatmap(d_df, bin_size=20):
        heatmap_counts = {}
        heatmap_gfa = {}
        if d_df.empty: return {'count': {}, 'gfa': {}}

        top_foundations = d_df['foundation_group'].value_counts().head(15).index.tolist()

        for fnd in top_foundations:
            fnd_df = d_df[d_df['foundation_group'] == fnd]
            if len(fnd_df) == 0: continue
            bins_c = {}
            bins_g = {}
            for i in range(0, 200, bin_size):
                label = f"{i}-{i + bin_size}"
                mask = (fnd_df['lifespan'] >= i) & (fnd_df['lifespan'] < i + bin_size)
                subset = fnd_df[mask]
                count = len(subset)
                if count > 0:
                    bins_c[label] = int(count)
                    bins_g[label] = int(subset['Est GFA sqmeters'].sum())
            if bins_c:
                heatmap_counts[fnd] = bins_c
                heatmap_gfa[fnd] = bins_g
        return {'count': heatmap_counts, 'gfa': heatmap_gfa}

    def make_district_occupancy_heatmap(d_df, bin_size=20):
        heatmap_counts = {}
        heatmap_gfa = {}
        if d_df.empty: return {'count': {}, 'gfa': {}}

        top_occs = d_df['occupancy_group'].value_counts().head(15).index.tolist()

        for occ in top_occs:
            occ_df = d_df[d_df['occupancy_group'] == occ]
            if len(occ_df) == 0: continue
            bins_c = {}
            bins_g = {}
            for i in range(0, 200, bin_size):
                label = f"{i}-{i + bin_size}"
                mask = (occ_df['lifespan'] >= i) & (occ_df['lifespan'] < i + bin_size)
                subset = occ_df[mask]
                count = len(subset)
                if count > 0:
                    bins_c[label] = int(count)
                    bins_g[label] = int(subset['Est GFA sqmeters'].sum())
            if bins_c:
                heatmap_counts[occ] = bins_c
                heatmap_gfa[occ] = bins_g
        return {'count': heatmap_counts, 'gfa': heatmap_gfa}

    def make_current_heatmap(d_df, bin_size=20, group_col='material_group'):
        heatmap_counts = {}
        heatmap_gfa = {}
        if d_df.empty: return {'count': {}, 'gfa': {}}

        top_groups = d_df[group_col].value_counts().head(15).index.tolist()

        for grp in top_groups:
            grp_df = d_df[d_df[group_col] == grp]
            if len(grp_df) == 0: continue
            bins_c = {}
            bins_g = {}
            for i in range(0, 200, bin_size):
                label = f"{i}-{i + bin_size}"
                mask = (grp_df['current_age'] >= i) & (grp_df['current_age'] < i + bin_size)
                subset = grp_df[mask]
                count = len(subset)
                if count > 0:
                    bins_c[label] = int(count)
                    bins_g[label] = int(subset['Est GFA sqmeters'].sum())
            if bins_c:
                heatmap_counts[grp] = bins_c
                heatmap_gfa[grp] = bins_g
        return {'count': heatmap_counts, 'gfa': heatmap_gfa}

    def make_boxplot_data(d_df, group_col='material_group', value_col='lifespan'):
        boxplot_data = {}
        if d_df.empty: return boxplot_data

        top_groups = d_df[group_col].value_counts().head(20).index.tolist()

        for grp in top_groups:
            grp_df = d_df[(d_df[group_col] == grp) & (d_df[value_col] > 0)]
            if len(grp_df) > 0:
                boxplot_data[grp] = grp_df[value_col].tolist()
        return boxplot_data

    def make_stats_data(d_df, group_col='material_group', is_demolished=True):
        stats_list = []
        if d_df.empty: return stats_list

        for grp in d_df[group_col].unique():
            grp_df = d_df[d_df[group_col] == grp]

            if is_demolished:
                raze_pos_df = grp_df[(grp_df['DEMOLITION_TYPE'] == 'RAZE') & (grp_df['lifespan'] > 0)]
                avg_lifespan = float(raze_pos_df['lifespan'].mean()) if not raze_pos_df.empty else 0.0
                if pd.isna(avg_lifespan):
                    avg_lifespan = 0.0
                stats_list.append({
                    'name': grp,
                    'count': int(len(grp_df)),
                    'avg_lifespan': avg_lifespan,
                    'demolition_breakdown': {
                        'RAZE': int((grp_df['DEMOLITION_TYPE'] == 'RAZE').sum()),
                        'EXTDEM': int((grp_df['DEMOLITION_TYPE'] == 'EXTDEM').sum()),
                        'INTDEM': int((grp_df['DEMOLITION_TYPE'] == 'INTDEM').sum())
                    }
                })
            else:
                avg_age = float(grp_df['current_age'].mean()) if not grp_df.empty else 0.0
                if pd.isna(avg_age):
                    avg_age = 0.0
                stats_list.append({
                    'name': grp,
                    'count': int(len(grp_df)),
                    'avg_age': avg_age
                })

        stats_list.sort(key=lambda x: x['count'], reverse=True)
        return stats_list

    for dist in all_districts:
        # 1. Get Demolished data for this district (from df)
        d_demo_df = df[df['Zoning_District'] == dist]
        r_df = d_demo_df[d_demo_df['DEMOLITION_TYPE'] == 'RAZE']
        r_pos = r_df[r_df['lifespan'] > 0]

        # 2. Get ALL Building data for this district (from all_buildings_df)
        d_all_df = all_buildings_df[all_buildings_df['Zoning_District'] == dist]

        # Points for Map (Only RAZE)
        points = []
        for _, r in r_pos.iterrows():
            if pd.notna(r['LATITUDE']):
                points.append({
                    'lat': r['LATITUDE'],
                    'lng': r['LONGITUDE'],
                    'lifespan': int(r['lifespan']),
                    'status': r['status_norm'],
                    'year_built': int(r['year_built']),
                    'material': str(r['material_group']),
                    'foundation': str(r.get('foundation_type', 'N/A'))
                })

        zoning_stats[str(dist)] = {

            'count_raze': int(len(r_pos)),
            'avg_raze_lifespan': float(r_pos['lifespan'].mean()) if len(r_pos) > 0 else 0,
            'demolished_age_distribution_10yr': make_hist(r_pos['lifespan']),
            'heatmap_data': make_district_heatmap(r_pos, bin_size=20),
            'heatmap_data_foundation': make_district_foundation_heatmap(r_pos, bin_size=20),
            'heatmap_data_occupancy': make_district_occupancy_heatmap(r_pos, bin_size=20),
            'positive_raze_points': points,


            'count_total': int(len(d_all_df)),
            'avg_current_age': float(d_all_df['current_age'].mean()) if len(d_all_df) > 0 else 0,
            'current_age_distribution_10yr': make_hist(d_all_df['current_age']),

            'current_heatmap_material': make_current_heatmap(d_all_df, 20, 'material_group'),
            'current_heatmap_foundation': make_current_heatmap(d_all_df, 20, 'foundation_group'),
            'current_heatmap_occupancy': make_current_heatmap(d_all_df, 20, 'occupancy_group'),

            'boxplot_material': make_boxplot_data(r_pos, 'material_group', 'lifespan'),
            'boxplot_foundation': make_boxplot_data(r_pos, 'foundation_group', 'lifespan'),
            'boxplot_occupancy': make_boxplot_data(r_pos, 'occupancy_group', 'lifespan'),

            'current_boxplot_material': make_boxplot_data(d_all_df, 'material_group', 'current_age'),
            'current_boxplot_foundation': make_boxplot_data(d_all_df, 'foundation_group', 'current_age'),
            'current_boxplot_occupancy': make_boxplot_data(d_all_df, 'occupancy_group', 'current_age'),

            'stats_material': make_stats_data(d_demo_df, 'material_group', True),
            'stats_foundation': make_stats_data(d_demo_df, 'foundation_group', True),
            'stats_occupancy': make_stats_data(d_demo_df, 'occupancy_group', True),

            'current_stats_material': make_stats_data(d_all_df, 'material_group', False),
            'current_stats_foundation': make_stats_data(d_all_df, 'foundation_group', False),
            'current_stats_occupancy': make_stats_data(d_all_df, 'occupancy_group', False),
        }

    all_raze_pos = df[(df['DEMOLITION_TYPE'] == 'RAZE') & (df['lifespan'] > 0) & (df['Zoning_District'].notna())]
    all_current = all_buildings_df[all_buildings_df['Zoning_District'].notna()].copy()
    all_current = all_current[~((all_current['DEMOLITION_TYPE'] == 'RAZE') & (all_current['lifespan'] > 0))]

    all_points = []
    for _, r in all_raze_pos.iterrows():
        if pd.notna(r['LATITUDE']):
            all_points.append({
                'lat': r['LATITUDE'], 'lng': r['LONGITUDE'],
                'lifespan': int(r['lifespan']), 'status': r['status_norm'],
                'year_built': int(r['year_built']), 'material': str(r['material_group']),
                'foundation': str(r.get('foundation_type', 'N/A'))
            })

    zoning_stats['All Boston'] = {
        'count_raze': int(len(all_raze_pos)),
        'avg_raze_lifespan': float(all_raze_pos['lifespan'].mean()) if len(all_raze_pos) > 0 else 0,
        'demolished_age_distribution_10yr': make_hist(all_raze_pos['lifespan']),
        'positive_raze_points': all_points,
        'heatmap_data': make_district_heatmap(all_raze_pos, 20),
        'heatmap_data_foundation': make_district_foundation_heatmap(all_raze_pos, 20),
        'heatmap_data_occupancy': make_district_occupancy_heatmap(all_raze_pos, 20),
        'current_heatmap_material': make_current_heatmap(all_current, 20, 'material_group'),
        'current_heatmap_foundation': make_current_heatmap(all_current, 20, 'foundation_group'),
        'current_heatmap_occupancy': make_current_heatmap(all_current, 20, 'occupancy_group'),
        'boxplot_material': make_boxplot_data(all_raze_pos, 'material_group', 'lifespan'),
        'boxplot_foundation': make_boxplot_data(all_raze_pos, 'foundation_group', 'lifespan'),
        'boxplot_occupancy': make_boxplot_data(all_raze_pos, 'occupancy_group', 'lifespan'),
        'current_boxplot_material': make_boxplot_data(all_current, 'material_group', 'current_age'),
        'current_boxplot_foundation': make_boxplot_data(all_current, 'foundation_group', 'current_age'),
        'current_boxplot_occupancy': make_boxplot_data(all_current, 'occupancy_group', 'current_age'),
        'stats_material': make_stats_data(df, 'material_group', True),
        'stats_foundation': make_stats_data(df, 'foundation_group', True),
        'stats_occupancy': make_stats_data(df, 'occupancy_group', True),
        'current_stats_material': make_stats_data(all_current, 'material_group', False),
        'current_stats_foundation': make_stats_data(all_current, 'foundation_group', False),
        'current_stats_occupancy': make_stats_data(all_current, 'occupancy_group', False),
        'count_total': int(len(all_current)),
        'avg_current_age': float(all_current['current_age'].mean()) if len(all_current) > 0 else 0,
        'current_age_distribution_10yr': make_hist(all_current['current_age']),
    }

    result['zoning_district_stats'] = zoning_stats
    result['zoning_district_names'] = ['All Boston'] + sorted([str(x) for x in all_districts])

    # --- D. Zoning Subdistrict Stats ---
    sub_stats = {}
    for sub in df['Zoning_Subdistrict'].dropna().unique():
        s_df = df[(df['Zoning_Subdistrict'] == sub) & (df['DEMOLITION_TYPE'] == 'RAZE')]
        if len(s_df) > 0:
            sub_stats[str(sub)] = {'avg_raze_lifespan': float(s_df['lifespan'].mean())}
    result['zoning_subdistrict_stats'] = sub_stats

    # --- F. Material Heatmap Data (No changes needed, uses Demolition data) ---
    print("7. Generating Material Heatmap data...")
    material_lifespan_demo = {}
    material_lifespan_demo_gfa = {}
    bin_sizes = [10, 20, 25, 30, 50]

    for demo_type in ['RAZE', 'EXTDEM', 'INTDEM', 'all']:
        material_lifespan_demo[demo_type] = {}
        material_lifespan_demo_gfa[demo_type] = {}
        if demo_type == 'all':
            type_df = df
        else:
            type_df = df[df['DEMOLITION_TYPE'] == demo_type]

        for bin_size in bin_sizes:
            bin_key = f'bin_{bin_size}'
            material_lifespan_demo[demo_type][bin_key] = {}
            material_lifespan_demo_gfa[demo_type][bin_key] = {}
            for mat in type_df['material_group'].unique():
                mat_df = type_df[type_df['material_group'] == mat]
                if len(mat_df) == 0: continue
                bins_c = {}
                bins_g = {}
                for i in range(0, 200, bin_size):
                    label = f"{i}-{i + bin_size} years" if i < 150 else f"{i}+ years"
                    mask = (mat_df['lifespan'] >= i) & (mat_df['lifespan'] < i + bin_size)
                    subset = mat_df[mask]
                    count = len(subset)
                    if count > 0:
                        bins_c[label] = int(count)
                        bins_g[label] = int(subset['Est GFA sqmeters'].sum())
                if bins_c:
                    material_lifespan_demo[demo_type][bin_key][mat] = bins_c
                    material_lifespan_demo_gfa[demo_type][bin_key][mat] = bins_g

    result['material_lifespan_demo'] = material_lifespan_demo
    result['material_lifespan_demo_gfa'] = material_lifespan_demo_gfa

    print("8. Generating Foundation Heatmap data...")
    foundation_lifespan_demo = {}
    foundation_lifespan_demo_gfa = {}

    for demo_type in ['RAZE', 'EXTDEM', 'INTDEM', 'all']:
        foundation_lifespan_demo[demo_type] = {}
        foundation_lifespan_demo_gfa[demo_type] = {}

        if demo_type == 'all':
            type_df = df
        else:
            type_df = df[df['DEMOLITION_TYPE'] == demo_type]

        for bin_size in bin_sizes:
            bin_key = f'bin_{bin_size}'
            foundation_lifespan_demo[demo_type][bin_key] = {}
            foundation_lifespan_demo_gfa[demo_type][bin_key] = {}

            for fnd in type_df['foundation_group'].unique():
                fnd_df = type_df[type_df['foundation_group'] == fnd]
                if len(fnd_df) == 0: continue

                bins_c = {}
                bins_g = {}

                for i in range(0, 200, bin_size):
                    label = f"{i}-{i + bin_size} years" if i < 150 else f"{i}+ years"
                    mask = (fnd_df['lifespan'] >= i) & (fnd_df['lifespan'] < i + bin_size)
                    subset = fnd_df[mask]
                    count = len(subset)
                    if count > 0:
                        bins_c[label] = int(count)
                        bins_g[label] = int(subset['Est GFA sqmeters'].sum())

                if bins_c:
                    foundation_lifespan_demo[demo_type][bin_key][fnd] = bins_c
                    foundation_lifespan_demo_gfa[demo_type][bin_key][fnd] = bins_g

    result['foundation_lifespan_demo'] = foundation_lifespan_demo
    result['foundation_lifespan_demo_gfa'] = foundation_lifespan_demo_gfa

    print("10. Generating Occupancy Heatmap data...")
    occupancy_lifespan_demo = {}
    occupancy_lifespan_demo_gfa = {}

    for demo_type in ['RAZE', 'EXTDEM', 'INTDEM', 'all']:
        occupancy_lifespan_demo[demo_type] = {}
        occupancy_lifespan_demo_gfa[demo_type] = {}

        if demo_type == 'all':
            type_df = df
        else:
            type_df = df[df['DEMOLITION_TYPE'] == demo_type]

        for bin_size in bin_sizes:
            bin_key = f'bin_{bin_size}'
            occupancy_lifespan_demo[demo_type][bin_key] = {}
            occupancy_lifespan_demo_gfa[demo_type][bin_key] = {}

            # 使用 occupancy_group
            for occ in type_df['occupancy_group'].unique():
                occ_df = type_df[type_df['occupancy_group'] == occ]
                if len(occ_df) == 0: continue

                bins_c = {}
                bins_g = {}

                for i in range(0, 200, bin_size):
                    label = f"{i}-{i + bin_size} years" if i < 150 else f"{i}+ years"
                    mask = (occ_df['lifespan'] >= i) & (occ_df['lifespan'] < i + bin_size)
                    subset = occ_df[mask]
                    count = len(subset)
                    if count > 0:
                        bins_c[label] = int(count)
                        bins_g[label] = int(subset['Est GFA sqmeters'].sum())

                if bins_c:
                    occupancy_lifespan_demo[demo_type][bin_key][occ] = bins_c
                    occupancy_lifespan_demo_gfa[demo_type][bin_key][occ] = bins_g

    result['occupancy_lifespan_demo'] = occupancy_lifespan_demo
    result['occupancy_lifespan_demo_gfa'] = occupancy_lifespan_demo_gfa

    print("9. Generating Advanced Chart Data (Current Age, Eras, Strip Plot)...")

    def make_global_hist(series, width=10):
        if series.empty: return []
        series = series[(series >= 0) & (series < 500)]  # Filter reasonable range
        if series.empty: return []
        max_val = int(series.max())
        bins = range(0, max_val + width, width)
        return [{'range': f"{b}-{b + width}", 'count': int(((series >= b) & (series < b + width)).sum())} for b in bins]

    # Current Age Distribution - All buildings with valid DEMOLITION_TYPE
    # Rule: Remove RAZE with positive lifespan (buildings that no longer exist)
    current_buildings = all_buildings_df[all_buildings_df['DEMOLITION_TYPE'].notna()]
    current_buildings = current_buildings[
        ~((current_buildings['DEMOLITION_TYPE'] == 'RAZE') & (current_buildings['lifespan'] > 0))]

    result['current_age_distribution_10yr'] = make_global_hist(current_buildings['current_age'], 10)

    # 2. Yearly Data for Advanced Stacked Charts

    yearly_age_dist = {}
    yearly_era_dist = {}

    age_bins_def = [
        (0, 5, '0-5 years'), (5, 10, '5-10 years'), (10, 20, '10-20 years'),
        (20, 30, '20-30 years'), (30, 50, '30-50 years'), (50, 75, '50-75 years'),
        (75, 100, '75-100 years'), (100, 150, '100-150 years'), (150, 9999, '150+ years')
    ]

    eras_def = [
        (0, 1900, 'Pre-1900'), (1900, 1920, '1900-1920'), (1920, 1940, '1920-1940'),
        (1940, 1960, '1940-1960'), (1960, 1980, '1960-1980'), (1980, 2000, '1980-2000'),
        (2000, 2010, '2000-2010'), (2010, 2020, '2010-2020'), (2020, 9999, '2020+')
    ]

    all_years = sorted(df['demolition_year'].dropna().unique().astype(int))

    for year in all_years:
        y_str = str(year)
        yearly_age_dist[y_str] = {'All': {}}
        yearly_era_dist[y_str] = {'All': {}}

        year_df = df[df['demolition_year'] == year]

        # === A. Age Distribution Logic ===
        for demo_type in ['RAZE', 'EXTDEM', 'INTDEM', 'All']:
            if demo_type == 'All':
                sub_df = year_df[year_df['lifespan'] > 0]
            else:
                sub_df = year_df[(year_df['DEMOLITION_TYPE'] == demo_type) & (year_df['lifespan'] > 0)]

            counts = {label: 0 for _, _, label in age_bins_def}

            for start, end, label in age_bins_def:
                c = len(sub_df[(sub_df['lifespan'] >= start) & (sub_df['lifespan'] < end)])
                counts[label] = c

            yearly_age_dist[y_str][demo_type] = counts

            # === B. Construction Era Logic ===
            for demo_type in ['RAZE', 'EXTDEM', 'INTDEM', 'All']:
                if demo_type == 'All':
                    sub_df = year_df[year_df['lifespan'] > 0]
                else:
                    sub_df = year_df[(year_df['DEMOLITION_TYPE'] == demo_type) & (year_df['lifespan'] > 0)]


                counts = {label: 0 for _, _, label in eras_def}

                for start, end, label in eras_def:
                    c = len(sub_df[(sub_df['year_built'] >= start) & (sub_df['year_built'] < end)])
                    counts[label] = c

                yearly_era_dist[y_str][demo_type] = counts

    result['yearly_age_distribution'] = yearly_age_dist
    result['yearly_construction_era'] = yearly_era_dist

    # 3. Strip Plot Data (Lifespan by Year Boxplot)

    strip_plot_data = {'All': []}
    for t in ['RAZE', 'EXTDEM', 'INTDEM']:
        strip_plot_data[t] = []

    for year in all_years:
        year_df = df[df['demolition_year'] == year]

        # For 'All'
        lifespans_all = year_df['lifespan'].dropna().tolist()
        strip_plot_data['All'].append({
            'year': int(year),
            'lifespans': [int(x) for x in lifespans_all if x > 0]  # Filter positive only
        })

        # For individual types
        for dtype in ['RAZE', 'EXTDEM', 'INTDEM']:
            sub = year_df[year_df['DEMOLITION_TYPE'] == dtype]
            lifespans = sub['lifespan'].dropna().tolist()
            strip_plot_data[dtype].append({
                'year': int(year),
                'lifespans': [int(x) for x in lifespans if x > 0]
            })

    result['lifespan_by_year_boxplot'] = strip_plot_data

    # --- G, H, I Legacy Features ---
    # Yearly Stacked (Uses Demolition Data)
    yearly_data = []
    years = sorted(df['demolition_year'].unique())
    for year in years:
        y_df = df[df['demolition_year'] == year]

        raze_all = y_df[y_df['DEMOLITION_TYPE'] == 'RAZE']
        raze_pos_count = int((raze_all['lifespan'] > 0).sum())
        raze_neg_count = int((raze_all['lifespan'] <= 0).sum())

        row = {
            'year': int(year),
            'RAZE': raze_pos_count,
            'EXTDEM': int((y_df['DEMOLITION_TYPE'] == 'EXTDEM').sum()),
            'INTDEM': int((y_df['DEMOLITION_TYPE'] == 'INTDEM').sum()),
            'demolished_and_replaced': raze_neg_count
        }

        row.update({f"{k}_closed": v for k, v in row.items() if k != 'year'})
        yearly_data.append(row)

    result['yearly_stacked'] = yearly_data
    result['yearly_stacked_closed'] = yearly_data

    # Lifespan Dist
    dist_10yr = make_hist(df['lifespan'], 10)
    final_dist = []
    for item in dist_10yr:
        rng = item['range']
        s, e = map(int, rng.split('-'))
        sub_df = df[(df['lifespan'] >= s) & (df['lifespan'] < e)]
        final_dist.append({
            'range': rng,
            'RAZE': int((sub_df['DEMOLITION_TYPE'] == 'RAZE').sum()),
            'EXTDEM': int((sub_df['DEMOLITION_TYPE'] == 'EXTDEM').sum()),
            'INTDEM': int((sub_df['DEMOLITION_TYPE'] == 'INTDEM').sum()),
            'RAZE_closed': int((sub_df['DEMOLITION_TYPE'] == 'RAZE').sum()),
            'EXTDEM_closed': int((sub_df['DEMOLITION_TYPE'] == 'EXTDEM').sum()),
            'INTDEM_closed': int((sub_df['DEMOLITION_TYPE'] == 'INTDEM').sum()),
        })
    result['lifespan_distribution'] = final_dist
    result['lifespan_distribution_closed'] = final_dist

    # Metadata (Updated to reflect full building scope)
    result['metadata'] = {
        'year_range': f"{int(df['demolition_year'].min())}-{int(df['demolition_year'].max())}",
        'generated_date': datetime.now().isoformat(),
        'total_ma_buildings': int(initial_ma_count),
        'total_boston_buildings': int(len(all_buildings_df)),  # New metadata field
        'total_boston_demolitions': int(len(df)),
        'year_built_range': f"{int(all_buildings_df['year_built'].min())}-{int(all_buildings_df['year_built'].max())}"
    }

    # City Stats
    city_stats = []
    for city in df['PROP_CITY'].value_counts().head(10).index:
        city_df = df[df['PROP_CITY'] == city]
        city_stats.append({
            'city': city,
            'count': int(len(city_df)),
            'avg_lifespan': float(city_df['lifespan'].mean())
        })
    result['city_stats'] = city_stats

    # Material Stats
    mat_stats = []
    for mat in df['material_group'].unique():
        m_df = df[df['material_group'] == mat]


        raze_pos_df = m_df[(m_df['DEMOLITION_TYPE'] == 'RAZE') & (m_df['lifespan'] > 0)]
        avg_raze_lifespan = float(raze_pos_df['lifespan'].mean()) if not raze_pos_df.empty else 0.0

        mat_stats.append({
            'material': mat,
            'count': int(len(m_df)),
            'avg_lifespan': avg_raze_lifespan,
            'demolition_breakdown': {
                'RAZE': int((m_df['DEMOLITION_TYPE'] == 'RAZE').sum()),
                'EXTDEM': int((m_df['DEMOLITION_TYPE'] == 'EXTDEM').sum()),
                'INTDEM': int((m_df['DEMOLITION_TYPE'] == 'INTDEM').sum())
            }
        })
    mat_stats.sort(key=lambda x: x['count'], reverse=True)
    result['material_stats'] = mat_stats

    # Boxplot
    raw_boxplot = {}
    for demo in ['RAZE', 'EXTDEM', 'INTDEM']:
        raw_boxplot[demo] = {}
        for mat in mat_stats[:20]:
            m_df = df[
                (df['DEMOLITION_TYPE'] == demo) & (df['material_group'] == mat['material']) & (df['lifespan'] > 0)]
            if len(m_df) > 0:
                raw_boxplot[demo][mat['material']] = m_df['lifespan'].tolist()
    result['material_lifespan_raw_by_demo'] = raw_boxplot

    return result


def save_json(data, filename='boston_demolition_data.json'):
    print(f"Saving data to {filename}...")
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    data = process_demolition_data()
    if data:
        save_json(data)
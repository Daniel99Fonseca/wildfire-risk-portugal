import argparse
import time
from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GRID_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "grid_5km.parquet"
)

VEGETATION_FRACTION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "grid_vegetation_fraction.parquet"
)

SENTINEL_PARTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sentinel_weekly_parts"
)

SENTINEL_PARTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------------------
# Global project objects
# ---------------------------------------------------------------------

GRID = None
VEGETATION_FRACTION_LOOKUP = None

SENTINEL2 = None
VEGETATION_MASK = None

FINAL_RULE = (
    (7, 95),
    (14, 90),
    (30, 85)
)

VEGETATION_CLC_CLASSES = [
    211, 212, 213,
    221, 222, 223,
    231,
    241, 242, 243, 244,
    311, 312, 313,
    321, 322, 323, 324
]


# ---------------------------------------------------------------------
# Earth Engine
# ---------------------------------------------------------------------

def initialize_earth_engine():

    global SENTINEL2
    global VEGETATION_MASK

    ee.Initialize(
        project="wildfireproject-506722"
    )

    SENTINEL2 = ee.ImageCollection(
        "COPERNICUS/S2_SR_HARMONIZED"
    )

    clc2018 = (
        ee.Image(
            "COPERNICUS/CORINE/V20/100m/2018"
        )
        .select("landcover")
    )

    VEGETATION_MASK = (
        clc2018
        .remap(
            VEGETATION_CLC_CLASSES,
            [1] * len(VEGETATION_CLC_CLASSES),
            0
        )
        .rename("vegetation_support")
    )


# ---------------------------------------------------------------------
# Temporal helpers
# ---------------------------------------------------------------------

def make_reference_dates(year):

    return pd.date_range(
        start=f"{year}-05-01",
        end=f"{year}-10-25",
        freq="W-MON"
    )


# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------

def get_cell_ee_geometry(cell_id):

    row = GRID.loc[
        GRID["cell_id"] == cell_id
    ]

    if row.empty:
        raise ValueError(
            f"Unknown cell_id: {cell_id}"
        )

    geometry = (
        row
        .to_crs("EPSG:4326")
        .geometry
        .iloc[0]
    )

    return ee.Geometry(
        geometry.__geo_interface__
    )


# ---------------------------------------------------------------------
# Sentinel-2 quality
# ---------------------------------------------------------------------

def scl_to_valid_mask(image):

    scl = image.select("SCL")

    valid = (
        scl.neq(0)
        .And(scl.neq(1))
        .And(scl.neq(3))
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
        .rename("valid")
    )

    return valid.unmask(0)


def get_acquisition_days(collection):

    timestamps = (
        collection
        .aggregate_array("system:time_start")
        .getInfo()
    )

    if not timestamps:
        return pd.Series(
            dtype="datetime64[ns]"
        )

    dates = pd.to_datetime(
        timestamps,
        unit="ms"
    )

    return (
        pd.Series(dates)
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )


def build_daily_valid_mask(
    collection,
    acquisition_day
):

    start_date = ee.Date(
        pd.Timestamp(
            acquisition_day
        ).strftime("%Y-%m-%d")
    )

    end_date = start_date.advance(
        1,
        "day"
    )

    daily_images = collection.filterDate(
        start_date,
        end_date
    )

    return (
        daily_images
        .map(scl_to_valid_mask)
        .max()
        .rename("valid")
    )


def get_daily_vegetation_quality_table(
    collection,
    geometry
):

    acquisition_days = get_acquisition_days(
        collection
    )

    features = []

    for day in acquisition_days:

        daily_valid = build_daily_valid_mask(
            collection,
            day
        )

        valid_on_vegetation = (
            daily_valid
            .updateMask(VEGETATION_MASK)
        )

        valid_fraction = (
            valid_on_vegetation
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=20,
                maxPixels=100000
            )
            .get("valid")
        )

        features.append(
            ee.Feature(
                None,
                {
                    "date":
                        day.strftime("%Y-%m-%d"),
                    "valid_fraction":
                        valid_fraction
                }
            )
        )

    if not features:
        return pd.DataFrame(
            columns=[
                "date",
                "valid_pct"
            ]
        )

    result = (
        ee.FeatureCollection(features)
        .getInfo()
    )

    quality = pd.DataFrame(
        [
            feature["properties"]
            for feature in result["features"]
        ]
    )

    quality["date"] = pd.to_datetime(
        quality["date"]
    )

    quality["valid_pct"] = (
        quality["valid_fraction"] * 100
    )

    return (
        quality[
            ["date", "valid_pct"]
        ]
        .sort_values("date")
        .reset_index(drop=True)
    )

# ---------------------------------------------------------------------
# Sentinel-2 reflectance
# ---------------------------------------------------------------------

def mask_sentinel_reflectance(image):

    valid_mask = scl_to_valid_mask(image)

    return image.updateMask(
        valid_mask
    )


def build_daily_reflectance_mosaic(
    collection,
    acquisition_day
):

    start_date = ee.Date(
        pd.Timestamp(
            acquisition_day
        ).strftime("%Y-%m-%d")
    )

    end_date = start_date.advance(
        1,
        "day"
    )

    daily_images = collection.filterDate(
        start_date,
        end_date
    )

    masked_images = daily_images.map(
        mask_sentinel_reflectance
    )

    return masked_images.mosaic()


# ---------------------------------------------------------------------
# NDVI / NDMI batch extraction
# ---------------------------------------------------------------------

def extract_acquisition_stats_batch(
    collection,
    geometry,
    good_quality
):

    if good_quality.empty:
        return pd.DataFrame()

    reducer = (
        ee.Reducer.mean()
        .combine(
            reducer2=ee.Reducer.median(),
            sharedInputs=True
        )
        .combine(
            reducer2=ee.Reducer.percentile(
                [10, 90]
            ),
            sharedInputs=True
        )
    )

    features = []

    for acquisition_date in good_quality["date"]:

        reflectance = (
            build_daily_reflectance_mosaic(
                collection,
                acquisition_date
            )
        )

        ndvi = (
            reflectance
            .normalizedDifference(
                ["B8", "B4"]
            )
            .rename("NDVI")
            .updateMask(
                VEGETATION_MASK
            )
        )

        ndmi = (
            reflectance
            .normalizedDifference(
                ["B8", "B11"]
            )
            .rename("NDMI")
            .updateMask(
                VEGETATION_MASK
            )
        )

        ndvi_stats = ndvi.reduceRegion(
            reducer=reducer,
            geometry=geometry,
            scale=10,
            maxPixels=1000000
        )

        ndmi_stats = ndmi.reduceRegion(
            reducer=reducer,
            geometry=geometry,
            scale=20,
            maxPixels=1000000
        )

        properties = (
            ee.Dictionary(ndvi_stats)
            .combine(
                ee.Dictionary(ndmi_stats),
                overwrite=True
            )
            .set(
                "selected_date",
                pd.Timestamp(
                    acquisition_date
                ).strftime("%Y-%m-%d")
            )
        )

        features.append(
            ee.Feature(
                None,
                properties
            )
        )

    result = (
        ee.FeatureCollection(features)
        .getInfo()
    )

    stats = pd.DataFrame(
        [
            feature["properties"]
            for feature in result["features"]
        ]
    )

    stats["selected_date"] = pd.to_datetime(
        stats["selected_date"]
    )

    quality_lookup = (
        good_quality[
            ["date", "valid_pct"]
        ]
        .rename(
            columns={
                "date": "selected_date"
            }
        )
    )

    return (
        stats
        .merge(
            quality_lookup,
            on="selected_date",
            how="left"
        )
        .sort_values("selected_date")
        .reset_index(drop=True)
    )

# ---------------------------------------------------------------------
# Adaptive image selection
# ---------------------------------------------------------------------

def select_adaptive_image(
    quality,
    reference_date,
    rules=FINAL_RULE
):
    reference_date = pd.Timestamp(
        reference_date
    )

    for window_days, min_valid_pct in rules:

        window_start = (
            reference_date
            - pd.Timedelta(days=window_days)
        )

        candidates = quality[
            (quality["date"] >= window_start)
            & (quality["date"] < reference_date)
            & (
                quality["valid_pct"]
                >= min_valid_pct
            )
        ]

        if not candidates.empty:

            selected = (
                candidates
                .sort_values("date")
                .iloc[-1]
            )

            return {
                "selected_date":
                    selected["date"],

                "valid_pct":
                    selected["valid_pct"],

                "age_days":
                    (
                        reference_date
                        - selected["date"]
                    ).days,

                "window_used":
                    window_days,

                "threshold_used":
                    min_valid_pct
            }

    return {
        "selected_date": pd.NaT,
        "valid_pct": None,
        "age_days": None,
        "window_used": None,
        "threshold_used": None
    }


# ---------------------------------------------------------------------
# Vegetation temporal trend
# ---------------------------------------------------------------------

def calculate_vegetation_trend(
    acquisition_series,
    reference_date,
    window_days=60,
    min_observations=3
):
    reference_date = pd.Timestamp(
        reference_date
    )

    window_start = (
        reference_date
        - pd.Timedelta(days=window_days)
    )

    trend_data = (
        acquisition_series[
            (
                acquisition_series["selected_date"]
                >= window_start
            )
            & (
                acquisition_series["selected_date"]
                < reference_date
            )
        ]
        .drop_duplicates(
            subset="selected_date"
        )
        .sort_values("selected_date")
        .copy()
    )

    n_observations = len(
        trend_data
    )

    if n_observations < min_observations:

        return {
            "vegetation_obs_60d":
                n_observations,

            "NDVI_slope_30d":
                None,

            "NDMI_slope_30d":
                None
        }

    trend_data["days"] = (
        trend_data["selected_date"]
        - trend_data["selected_date"].min()
    ).dt.days

    ndvi_slope_daily = (
        trend_data["NDVI_mean"].cov(
            trend_data["days"]
        )
        / trend_data["days"].var()
    )

    ndmi_slope_daily = (
        trend_data["NDMI_mean"].cov(
            trend_data["days"]
        )
        / trend_data["days"].var()
    )

    return {
        "vegetation_obs_60d":
            n_observations,

        "NDVI_slope_30d":
            ndvi_slope_daily * 30,

        "NDMI_slope_30d":
            ndmi_slope_daily * 30
    }


# ---------------------------------------------------------------------
# Cell-year acquisition table
# ---------------------------------------------------------------------

def build_cell_year_acquisition_table(
    cell_id,
    year,
    vegetation_fraction
):
    geometry = get_cell_ee_geometry(
        cell_id
    )

    if vegetation_fraction <= 0:
        return pd.DataFrame()

    reference_dates = make_reference_dates(
        year
    )

    start_date = (
        min(reference_dates)
        - pd.Timedelta(days=60)
    )

    end_date = max(
        reference_dates
    )

    collection = (
        SENTINEL2
        .filterBounds(geometry)
        .filterDate(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
    )

    quality = (
        get_daily_vegetation_quality_table(
            collection,
            geometry
        )
    )

    good_quality = (
        quality[
            quality["valid_pct"] >= 85
        ]
        .copy()
        .reset_index(drop=True)
    )

    acquisition_table = (
        extract_acquisition_stats_batch(
            collection,
            geometry,
            good_quality
        )
    )

    return acquisition_table


# ---------------------------------------------------------------------
# Weekly vegetation features
# ---------------------------------------------------------------------

def build_cell_year_weekly_features(
    cell_id,
    year,
    acquisition_table,
    vegetation_fraction
):
    weekly_rows = []

    # No usable vegetation / Sentinel observations
    if acquisition_table.empty:

        for reference_date in make_reference_dates(year):

            weekly_rows.append({
                "cell_id": cell_id,
                "reference_date": reference_date,
                "vegetation_fraction": vegetation_fraction,
                "selected_date": pd.NaT,
                "image_age_days": None,
                "valid_pct": None,

                "NDVI_mean": None,
                "NDVI_median": None,
                "NDVI_p10": None,
                "NDVI_p90": None,

                "NDMI_mean": None,
                "NDMI_median": None,
                "NDMI_p10": None,
                "NDMI_p90": None,

                "vegetation_obs_60d": 0,
                "NDVI_slope_30d": None,
                "NDMI_slope_30d": None
            })

        return pd.DataFrame(
            weekly_rows
        )

    quality_table = (
        acquisition_table[
            [
                "selected_date",
                "valid_pct"
            ]
        ]
        .rename(
            columns={
                "selected_date": "date"
            }
        )
    )

    for reference_date in make_reference_dates(year):

        # -------------------------------------------------------------
        # Current vegetation state
        # -------------------------------------------------------------

        selection = select_adaptive_image(
            quality_table,
            reference_date,
            rules=FINAL_RULE
        )

        current_stats = {
            "NDVI_mean": None,
            "NDVI_median": None,
            "NDVI_p10": None,
            "NDVI_p90": None,

            "NDMI_mean": None,
            "NDMI_median": None,
            "NDMI_p10": None,
            "NDMI_p90": None
        }

        if not pd.isna(
            selection["selected_date"]
        ):

            selected_row = (
                acquisition_table[
                    acquisition_table[
                        "selected_date"
                    ]
                    == selection[
                        "selected_date"
                    ]
                ]
                .iloc[0]
            )

            current_stats = {
                "NDVI_mean":
                    selected_row["NDVI_mean"],

                "NDVI_median":
                    selected_row["NDVI_median"],

                "NDVI_p10":
                    selected_row["NDVI_p10"],

                "NDVI_p90":
                    selected_row["NDVI_p90"],

                "NDMI_mean":
                    selected_row["NDMI_mean"],

                "NDMI_median":
                    selected_row["NDMI_median"],

                "NDMI_p10":
                    selected_row["NDMI_p10"],

                "NDMI_p90":
                    selected_row["NDMI_p90"]
            }

        # -------------------------------------------------------------
        # 60-day trend
        # -------------------------------------------------------------

        trend = calculate_vegetation_trend(
            acquisition_table,
            reference_date
        )

        weekly_rows.append({
            "cell_id": cell_id,
            "reference_date": reference_date,
            "vegetation_fraction":
                vegetation_fraction,

            "selected_date":
                selection["selected_date"],

            "image_age_days":
                selection["age_days"],

            "valid_pct":
                selection["valid_pct"],

            **current_stats,
            **trend
        })

    return pd.DataFrame(
        weekly_rows
    )   

def process_cell_year_final(
    cell_id,
    year,
    vegetation_fraction
):
    acquisitions = (
        build_cell_year_acquisition_table(
            cell_id,
            year,
            vegetation_fraction
        )
    )

    weekly = (
        build_cell_year_weekly_features(
            cell_id,
            year,
            acquisitions,
            vegetation_fraction
        )
    )

    return weekly


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def load_project_data():

    grid = gpd.read_parquet(
        GRID_PATH
    )

    grid_vegetation = pd.read_parquet(
        VEGETATION_FRACTION_PATH
    )

    vegetation_fraction_lookup = dict(
        zip(
            grid_vegetation["cell_id"],
            grid_vegetation["vegetation_fraction"]
        )
    )

    return (
        grid,
        grid_vegetation,
        vegetation_fraction_lookup
    )


# ---------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Extract weekly Sentinel-2 vegetation "
            "features for Mainland Portugal."
        )
    )

    parser.add_argument(
        "--year",
        type=int,
        required=True,
        choices=range(2017, 2026),
        help="Year to process (2017-2025)."
    )

    parser.add_argument(
    "--cell",
    type=str,
    default=None,
    help="Optional cell_id for a single-cell test."
    )

    parser.add_argument(
    "--run-all",
    action="store_true",
    help="Process all grid cells for the selected year."
    )

    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Optional limit for a production test."
    )


    return parser.parse_args()


# ---------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------

def get_cell_year_output_path(
    cell_id,
    year
):
    return (
        SENTINEL_PARTS_DIR
        / f"{cell_id}_{year}.parquet"
    )


def save_cell_year_result(
    weekly,
    cell_id,
    year
):
    output_path = get_cell_year_output_path(
        cell_id,
        year
    )

    temp_path = output_path.with_suffix(
        ".tmp.parquet"
    )

    weekly.to_parquet(
        temp_path,
        index=False
    )

    temp_path.replace(
        output_path
    )

    return output_path


# ---------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------

def process_cell_year_with_retry(
    cell_id,
    year,
    vegetation_fraction,
    max_attempts=3
):
    for attempt in range(
        1,
        max_attempts + 1
    ):

        try:
            return process_cell_year_final(
                cell_id,
                year,
                vegetation_fraction
            )

        except Exception as error:

            if attempt == max_attempts:
                raise

            wait_seconds = 10 * attempt

            print(
                f"  Retry {attempt}/"
                f"{max_attempts - 1} "
                f"after error: {error}"
            )

            print(
                f"  Waiting {wait_seconds}s..."
            )

            time.sleep(
                wait_seconds
            )

# ---------------------------------------------------------------------
# Production runner
# ---------------------------------------------------------------------

def run_sentinel_extraction(
    cell_ids,
    year
):
    total = len(cell_ids)

    skipped = 0
    failed = []

    start_total = time.perf_counter()

    for i, cell_id in enumerate(
        cell_ids,
        start=1
    ):

        output_path = (
            get_cell_year_output_path(
                cell_id,
                year
            )
        )

        if output_path.exists():

            skipped += 1

            print(
                f"[{i}/{total}] "
                f"SKIP {cell_id} {year}"
            )

            continue

        vegetation_fraction = (
            VEGETATION_FRACTION_LOOKUP[
                cell_id
            ]
        )

        try:

            start = time.perf_counter()

            weekly = (
                process_cell_year_with_retry(
                    cell_id,
                    year,
                    vegetation_fraction
                )
            )

            save_cell_year_result(
                weekly,
                cell_id,
                year
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            print(
                f"[{i}/{total}] "
                f"DONE {cell_id} {year} "
                f"({elapsed:.1f}s)"
            )

        except Exception as error:

            failed.append({
                "cell_id": cell_id,
                "year": year,
                "error": str(error)
            })

            print(
                f"[{i}/{total}] "
                f"FAILED {cell_id} {year}: "
                f"{error}"
            )

    total_elapsed = (
        time.perf_counter()
        - start_total
    )

    print("\nExtraction finished.")
    print("Skipped:", skipped)
    print("Failed:", len(failed))
    print(
        f"Elapsed: "
        f"{total_elapsed / 60:.1f} min"
    )

    return pd.DataFrame(
        failed
    )

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    global GRID
    global VEGETATION_FRACTION_LOOKUP

    args = parse_args()

    print("Initializing Earth Engine...")
    initialize_earth_engine()

    print("Loading project data...")

    (
        GRID,
        grid_vegetation,
        VEGETATION_FRACTION_LOOKUP
    ) = load_project_data()

    print(f"Grid cells: {len(GRID)}")

    print(
        "Vegetation fractions:",
        len(VEGETATION_FRACTION_LOOKUP)
    )
    
    if args.cell is not None:

        if args.cell not in VEGETATION_FRACTION_LOOKUP:
            raise ValueError(
                f"Unknown cell_id: {args.cell}"
            )

        vegetation_fraction = (
            VEGETATION_FRACTION_LOOKUP[
                args.cell
            ]
        )

        print(
            f"\nProcessing {args.cell} "
            f"for {args.year}..."
        )

        start = time.perf_counter()

        weekly = process_cell_year_final(
            args.cell,
            args.year,
            vegetation_fraction
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        print(
            "Shape:",
            weekly.shape
        )

        print(
            f"Elapsed: {elapsed:.1f}s"
        )

        print(
            weekly.head()
        )
    if args.run_all:

        cell_ids = GRID[
            "cell_id"
        ].tolist()

        if args.max_cells is not None:
            cell_ids = cell_ids[
                :args.max_cells
            ]

        print(
            f"\nStarting extraction for "
            f"{args.year}"
        )

        print(
            f"Cells to consider: "
            f"{len(cell_ids)}"
        )

        failures = run_sentinel_extraction(
            cell_ids,
            args.year
        )

        failures_path = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / f"sentinel_failures_{args.year}.csv"
        )

        failures.to_csv(
            failures_path,
            index=False
        )

        print(
            f"Failure report: "
            f"{failures_path}"
        )

if __name__ == "__main__":
    main()
import os

import pandas as pd
import psycopg2
import psycopg2.extras


FINAL_CSV_FILENAME = "benchmark_outputs/collated_results.csv"
TABLE_NAME = "diffusers_benchmarks"

if __name__ == "__main__":
    conn = psycopg2.connect(
        host=os.getenv("PGHOST"),
        database=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
    )
    cur = conn.cursor()

    cur.execute(f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        scenario       TEXT,
        model_cls      TEXT,
        num_params_M   REAL,
        flops_M        REAL,
        time_plain_s   REAL,
        mem_plain_GB   REAL,
        time_compile_s REAL,
        mem_compile_GB REAL,
        fullgraph      BOOLEAN,
        mode           TEXT,
        github_sha     TEXT
    );
    """)
    conn.commit()

    df = pd.read_csv(FINAL_CSV_FILENAME)

    # Helper to cast values (or None) given a dtype
    def _cast_value(val, dtype: str):
        if pd.isna(val):
            return None

        if dtype == "text":
            return str(val).strip()

        if dtype == "float":
            try:
                return float(val)
            except ValueError:
                return None

        if dtype == "bool":
            s = str(val).strip().lower()
            if s in ("true", "t", "yes", "1"):
                return True
            if s in ("false", "f", "no", "0"):
                return False
            if val in (1, 1.0):
                return True
            if val in (0, 0.0):
                return False
            return None

        return val

    rows_to_insert = []
    for _, row in df.iterrows():
        scenario = _cast_value(row.get("scenario"), "text")
        model_cls = _cast_value(row.get("model_cls"), "text")
        num_params_M = _cast_value(row.get("num_params_M"), "float")
        flops_M = _cast_value(row.get("flops_M"), "float")
        time_plain_s = _cast_value(row.get("time_plain_s"), "float")
        mem_plain_GB = _cast_value(row.get("mem_plain_GB"), "float")
        time_compile_s = _cast_value(row.get("time_compile_s"), "float")
        mem_compile_GB = _cast_value(row.get("mem_compile_GB"), "float")
        fullgraph = _cast_value(row.get("fullgraph"), "bool")
        mode = _cast_value(row.get("mode"), "text")

        # If "github_sha" column exists in the CSV, cast it; else default to None
        if "github_sha" in df.columns:
            github_sha = _cast_value(row.get("github_sha"), "text")
        else:
            github_sha = None

        rows_to_insert.append(
            (
                scenario,
                model_cls,
                num_params_M,
                flops_M,
                time_plain_s,
                mem_plain_GB,
                time_compile_s,
                mem_compile_GB,
                fullgraph,
                mode,
                github_sha,
            )
        )

    # Batch-insert all rows (with NULL for any None)
    insert_sql = """
    INSERT INTO benchmarks (
        scenario,
        model_cls,
        num_params_M,
        flops_M,
        time_plain_s,
        mem_plain_GB,
        time_compile_s,
        mem_compile_GB,
        fullgraph,
        mode,
        github_sha
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    """

    psycopg2.extras.execute_batch(cur, insert_sql, rows_to_insert)
    conn.commit()

    cur.close()
    conn.close()

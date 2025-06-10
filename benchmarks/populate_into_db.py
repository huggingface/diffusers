import datetime
import os
import uuid

import pandas as pd
import psycopg2
import psycopg2.extras


FINAL_CSV_FILENAME = "collated_results.csv"
# https://github.com/huggingface/transformers/blob/593e29c5e2a9b17baec010e8dc7c1431fed6e841/benchmark/init_db.sql#L27
BENCHMARKS_TABLE_NAME = "benchmarks"
MEASUREMENTS_TABLE_NAME = "model_measurements"

if __name__ == "__main__":
    try:
        conn = psycopg2.connect(
            host=os.getenv("PGHOST"),
            database=os.getenv("PGDATABASE"),
            user=os.getenv("PGUSER"),
            password=os.getenv("PGPASSWORD"),
        )
        print("DB connection established successfully.")
    except Exception:
        raise
    cur = conn.cursor()

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

    try:
        rows_to_insert = []
        id_for_benchmark = str(uuid.uuid4()) + "_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        for _, row in df.iterrows():
            scenario = _cast_value(row.get("scenario"), "text")
            model_cls = _cast_value(row.get("model_cls"), "text")
            num_params_B = _cast_value(row.get("num_params_B"), "float")
            flops_G = _cast_value(row.get("flops_G"), "float")
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

            if github_sha:
                benchmark_id = f"{model_cls}-{scenario}-{github_sha}"
            else:
                benchmark_id = f"{model_cls}-{scenario}-{id_for_benchmark}"

            measurements = {
                "repository": "huggingface/diffusers",
                "scenario": scenario,
                "model_cls": model_cls,
                "num_params_B": num_params_B,
                "flops_G": flops_G,
                "time_plain_s": time_plain_s,
                "mem_plain_GB": mem_plain_GB,
                "time_compile_s": time_compile_s,
                "mem_compile_GB": mem_compile_GB,
                "fullgraph": fullgraph,
                "mode": mode,
                "github_sha": github_sha,
            }
            rows_to_insert.append((benchmark_id, measurements))

        # Batch-insert all rows
        insert_sql = f"""
        INSERT INTO {MEASUREMENTS_TABLE_NAME} (
            benchmark_id,
            measurements
        )
        VALUES (%s, %s);
        """

        psycopg2.extras.execute_batch(cur, insert_sql, rows_to_insert)
        conn.commit()

        cur.close()
        conn.close()
    except Exception as e:
        print(f"Exception: {e}")

"""Recover the DOT-id -> IATA-code mapping the October slice needs.

The 2015 USDOT feed changes key encoding partway through the year: for one
month the airport columns carry numeric DOT ids (14747) instead of the IATA
codes (SEA) used everywhere else. Nothing in the shipped reference files
(airports.csv holds IATA only) can decode them, so a naive join silently
drops 486,165 rows.

Rather than hard-coding a lookup table from an external source, we recover it
from the data itself. Two independent signals are used, because neither is
sufficient alone:

  1. FLIGHT-NUMBER VOTE. A flight number flies the same route all year, so
     key on (AIRLINE, FLIGHT_NUMBER, DISTANCE) and read off the IATA code
     that the other eleven months show on the same key.

     Caveat handled here: some carriers reuse one flight number for both legs
     of a round trip, so that key can describe A->B on some days and B->A on
     others, letting the origin code collect votes for the destination
     airport. We therefore vote only from keys whose direction is
     unambiguous. This is accurate for codes with many votes and unreliable
     in the thin tail (some codes get as few as two votes).

  2. GEOMETRIC FIT. Every row already carries its own route DISTANCE, so once
     enough codes are known, an unknown code can be located by trilateration:
     score every candidate airport by how well it reproduces the observed
     distances to its known partners, and take the best fit.

Signal 1 bootstraps signal 2. Signal 2 then repairs the tail that signal 1
gets wrong. Crucially, the final check is per-code rather than aggregate: a
mis-mapped airport is wrong on *every* row it appears in, so it shows up as a
large median error, whereas pooled statistics hide it (307 codes, ~1500 rows
each, so five bad codes barely move the global median).

Note that swapping a route's endpoints leaves its great-circle distance
unchanged, so signal 2 cannot by itself detect a transposition - which is
exactly why signal 1 is direction-aware.

Writes data/dot_to_iata.csv.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from pyspark.sql import Window, functions as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spark_session import build_spark, path  # noqa: E402

RAW_PARQUET = path("data", "parquet", "flights_raw")
OUT_CSV = path("data", "dot_to_iata.csv")

NUMERIC = "^[0-9]+$"
EARTH_MI = 3958.7613
# A correct mapping reproduces DISTANCE to within a couple of miles (airport
# reference point vs. published route distance). 25 mi is far outside that.
ERR_TOL_MI = 25.0


def haversine_np(lat1, lon1, lat2, lon2):
    """Great-circle distance in statute miles, vectorised over numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_MI * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ---------------------------------------------------------------------------
# Signal 1: direction-aware flight-number vote (distributed)
# ---------------------------------------------------------------------------
def vote_mapping(encoded, clean) -> pd.DataFrame:
    key = ["AIRLINE", "FLIGHT_NUMBER", "DISTANCE"]

    def one_directional(frame):
        single = (frame.groupBy(*key)
                  .agg(F.countDistinct(F.concat_ws(">", "ORIGIN_AIRPORT",
                                                   "DESTINATION_AIRPORT"))
                       .alias("n_pairs"))
                  .filter(F.col("n_pairs") == 1).select(*key))
        return frame.join(F.broadcast(single), key, "inner")

    enc, cln = one_directional(encoded), one_directional(clean)

    votes = []
    for col_name in ("ORIGIN_AIRPORT", "DESTINATION_AIRPORT"):
        e = (enc.filter(F.col(col_name).rlike(NUMERIC))
                .select(*key, F.col(col_name).alias("dot_code")).distinct())
        c = (cln.filter(~F.col(col_name).rlike(NUMERIC))
                .select(*key, F.col(col_name).alias("iata_code")).distinct())
        votes.append(e.join(c, key, "inner").select("dot_code", "iata_code"))

    tally = (votes[0].union(votes[1])
             .groupBy("dot_code", "iata_code").agg(F.count("*").alias("votes")))
    ranked = tally.withColumn("rk", F.row_number().over(
        Window.partitionBy("dot_code").orderBy(F.desc("votes"), "iata_code")))
    total = tally.groupBy("dot_code").agg(F.sum("votes").alias("total_votes"))

    return (ranked.filter(F.col("rk") == 1).drop("rk")
            .join(total, "dot_code")
            .select("dot_code", "iata_code", "votes", "total_votes")
            .toPandas())


# ---------------------------------------------------------------------------
# Signal 2: geometric fit (driver-side; the input is only a few thousand rows)
# ---------------------------------------------------------------------------
def per_code_error(routes: pd.DataFrame, mapping: dict, coords: pd.DataFrame) -> pd.Series:
    """Median |great-circle - DISTANCE| for each mapped code, over its routes."""
    lat = coords["lat"].to_dict()
    lon = coords["lon"].to_dict()

    o_iata = routes["origin"].map(mapping)
    d_iata = routes["dest"].map(mapping)
    ok = o_iata.notna() & d_iata.notna()
    if not ok.any():
        return pd.Series(dtype="float64")

    sub = routes[ok]
    gc = haversine_np(o_iata[ok].map(lat).to_numpy(float),
                      o_iata[ok].map(lon).to_numpy(float),
                      d_iata[ok].map(lat).to_numpy(float),
                      d_iata[ok].map(lon).to_numpy(float))
    err = np.abs(gc - sub["distance"].to_numpy(float))

    long = pd.concat([
        pd.DataFrame({"code": sub["origin"].to_numpy(), "err": err}),
        pd.DataFrame({"code": sub["dest"].to_numpy(), "err": err}),
    ])
    return long.groupby("code")["err"].median()


# A fit is accepted only if it is both tight in absolute terms and clearly
# better than the runner-up. The margin test is what makes a one- or
# two-anchor fit safe: a single distance constrains the airport to a circle,
# so the fit is only trustworthy when no other free candidate sits near it.
FIT_TOL_MI = 10.0
FIT_MARGIN_MI = 25.0


def geometric_refit(code: str, routes: pd.DataFrame, trusted: dict,
                    coords: pd.DataFrame, taken: set):
    """Locate one unknown code by fitting distances to its known partners.

    Returns (best_candidate, score, margin, n_anchors). The remaining
    unresolved codes are small or seasonal airports that fly only a handful
    of routes, so demanding many anchors would reject exactly the cases that
    still need solving; the margin test carries the safety instead.
    """
    lat = coords["lat"].to_dict()
    lon = coords["lon"].to_dict()

    as_origin = routes[routes["origin"] == code].assign(partner=lambda d: d["dest"])
    as_dest = routes[routes["dest"] == code].assign(partner=lambda d: d["origin"])
    obs = pd.concat([as_origin, as_dest])
    obs = obs.assign(p_iata=obs["partner"].map(trusted)).dropna(subset=["p_iata"])
    if obs.empty:
        return None, float("inf"), 0.0, 0

    p_lat = obs["p_iata"].map(lat).to_numpy(float)
    p_lon = obs["p_iata"].map(lon).to_numpy(float)
    dist = obs["distance"].to_numpy(float)

    scored = []
    for cand in (c for c in coords.index if c not in taken):
        gc = haversine_np(np.full_like(p_lat, lat[cand]),
                          np.full_like(p_lon, lon[cand]), p_lat, p_lon)
        scored.append((float(np.median(np.abs(gc - dist))), cand))
    if not scored:
        return None, float("inf"), 0.0, len(obs)

    scored.sort()
    best_score, best = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else float("inf")
    return best, best_score, runner_up - best_score, len(obs)


def main() -> None:
    spark = build_spark("build-airport-code-map")
    try:
        df = spark.read.parquet(RAW_PARQUET).select(
            "MONTH", "AIRLINE", "FLIGHT_NUMBER", "DISTANCE",
            "ORIGIN_AIRPORT", "DESTINATION_AIRPORT")

        # ---- 1. locate the schema drift ------------------------------------
        print("== rows carrying a numeric airport code, by month ==")
        drift = (df.withColumn("numeric",
                               (F.col("ORIGIN_AIRPORT").rlike(NUMERIC) |
                                F.col("DESTINATION_AIRPORT").rlike(NUMERIC)).cast("int"))
                 .groupBy("MONTH").agg(F.sum("numeric").alias("numeric_rows"),
                                       F.count("*").alias("rows"))
                 .orderBy("MONTH"))
        drift.show(12, truncate=False)

        bad_months = [r["MONTH"] for r in drift.collect() if r["numeric_rows"] > 0]
        print("affected months: %s" % bad_months)
        if not bad_months:
            print("no schema drift found - nothing to do")
            return

        encoded = df.filter(F.col("MONTH").isin(bad_months))
        clean = df.filter(~F.col("MONTH").isin(bad_months))

        all_codes = sorted(r["c"] for r in
                           encoded.select(F.col("ORIGIN_AIRPORT").alias("c"))
                           .union(encoded.select(F.col("DESTINATION_AIRPORT").alias("c")))
                           .filter(F.col("c").rlike(NUMERIC)).distinct().collect())
        print("distinct numeric codes to resolve: %d" % len(all_codes))

        # The encoded month collapses to a few thousand distinct routes, small
        # enough to finish the combinatorial part on the driver.
        routes = (encoded.groupBy(F.col("ORIGIN_AIRPORT").alias("origin"),
                                  F.col("DESTINATION_AIRPORT").alias("dest"),
                                  F.col("DISTANCE").alias("distance"))
                  .count().toPandas())
        print("distinct routes in the encoded month: %d" % len(routes))

        airports = spark.read.csv(path("data", "airports.csv"), header=True,
                                  inferSchema=True).toPandas()
        coords = (airports.rename(columns={"IATA_CODE": "iata", "LATITUDE": "lat",
                                           "LONGITUDE": "lon"})
                  .dropna(subset=["lat", "lon"]).set_index("iata")[["lat", "lon"]])
        print("airports with coordinates: %d" % len(coords))

        # ---- 2. flight-number vote -----------------------------------------
        votes = vote_mapping(encoded, clean)
        votes = votes[votes["iata_code"].isin(coords.index)]
        mapping = dict(zip(votes["dot_code"], votes["iata_code"]))
        source = {c: "vote" for c in mapping}
        print("\nvote resolved %d / %d codes" % (len(mapping), len(all_codes)))

        # ---- 3. score every voted code on its own, and quarantine the bad ---
        err = per_code_error(routes, mapping, coords)
        suspect = sorted(set(all_codes) - set(mapping)
                         | set(err[err > ERR_TOL_MI].index))
        print("codes failing the per-code geometric check (>%.0f mi): %d"
              % (ERR_TOL_MI, len(err[err > ERR_TOL_MI])))
        for c in err[err > ERR_TOL_MI].sort_values(ascending=False).index:
            print("   %s -> %s  median_err=%.1f mi" % (c, mapping[c], err[c]))
        print("codes unresolved by the vote: %d" % len(set(all_codes) - set(mapping)))

        # ---- 4. repair the tail geometrically ------------------------------
        trusted = {c: i for c, i in mapping.items() if c not in suspect}
        for rnd in range(6):
            if not suspect:
                break
            taken = set(trusted.values())
            # Score every open code, then commit the most confident fits first:
            # each assignment removes a candidate and so sharpens the rest.
            scored = []
            for code in suspect:
                cand, score, margin, anchors = geometric_refit(
                    code, routes, trusted, coords, taken)
                scored.append((score, code, cand, margin, anchors))
            scored.sort()

            fixed, still = [], []
            for score, code, cand, margin, anchors in scored:
                # Either the runner-up is far away in absolute terms, or the
                # fit is so much tighter than the runner-up that the gap is
                # decisive anyway: a 0.3 mi fit against a 12 mi alternative is
                # unambiguous even though 12 mi is a small absolute margin.
                runner_up = score + margin
                decisive = (margin >= FIT_MARGIN_MI
                            or runner_up >= 5 * max(score, 0.5))
                accept = (cand is not None and cand not in taken
                          and score <= FIT_TOL_MI and decisive)
                if accept:
                    trusted[code] = cand
                    source[code] = "geometric"
                    taken.add(cand)
                    fixed.append((code, cand, score, margin, anchors))
                else:
                    still.append((code, cand, score, margin, anchors))

            print("\nrefit round %d: repaired %d, still open %d"
                  % (rnd + 1, len(fixed), len(still)))
            for code, cand, score, margin, anchors in fixed:
                print("   %s -> %-4s (vote said %-4s)  fit_err=%.2f mi  "
                      "margin=%.0f mi  anchors=%d"
                      % (code, cand, mapping.get(code, "-"), score, margin, anchors))
            for code, cand, score, margin, anchors in still:
                print("   OPEN %s  best=%s score=%.1f margin=%.1f anchors=%d"
                      % (code, cand, score, margin, anchors))
            suspect = [c for c, *_ in still]
            if not fixed:
                break

        mapping = trusted
        if suspect:
            print("\nUNRESOLVED after refit: %s" % suspect)

        # ---- 5. final per-code verification --------------------------------
        err = per_code_error(routes, mapping, coords)
        print("\n== final verification (per code, over the encoded month) ==")
        print("  codes mapped            : %d / %d" % (len(mapping), len(all_codes)))
        print("  median of per-code error: %.2f mi" % err.median())
        print("  worst per-code error    : %.2f mi" % err.max())
        print("  codes above %.0f mi       : %d" % (ERR_TOL_MI, (err > ERR_TOL_MI).sum()))
        print("  distinct IATA targets   : %d (injective: %s)"
              % (len(set(mapping.values())),
                 len(set(mapping.values())) == len(mapping)))

        worst = err.sort_values(ascending=False).head(5)
        print("  worst five:")
        for code, e in worst.items():
            print("    %s -> %s  %.2f mi" % (code, mapping[code], e))

        # coverage: what fraction of encoded rows can now be joined
        covered = routes[routes["origin"].isin(mapping) & routes["dest"].isin(mapping)]
        print("  encoded rows now joinable: {:,} / {:,}".format(
            int(covered["count"].sum()), int(routes["count"].sum())))

        out = (pd.DataFrame({"dot_code": list(mapping),
                             "iata_code": [mapping[c] for c in mapping]})
               .assign(source=lambda d: d["dot_code"].map(source),
                       median_err_mi=lambda d: d["dot_code"].map(err).round(3))
               .sort_values("dot_code"))
        out.to_csv(OUT_CSV, index=False)
        print("\nwrote %s (%d rows)" % (OUT_CSV, len(out)))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

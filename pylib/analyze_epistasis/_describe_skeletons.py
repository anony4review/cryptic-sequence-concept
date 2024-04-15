import itertools as it
import typing

import numpy as np
import pandas as pd

from ._jackknife_skeleton import jackknife_skeleton


def describe_skeletons(
    skeletons: typing.List[np.array],
    test_knockout: typing.Callable,
) -> pd.DataFrame:
    skeletons_df = _make_skeletons_df(skeletons)
    skeletonization_df = _make_skeletonization_df(skeletons_df)
    jackknifes_df = _make_jackknifes_df(skeletons, test_knockout)

    res = pd.DataFrame.merge(
        skeletonization_df,
        jackknifes_df,
        how="outer",
        on=["site"],
    )
    assert res["skeleton outcome count, excluded"].isna().sum() == 0
    assert res["skeleton outcome frequency, excluded"].isna().sum() == 0
    assert res["skeleton outcome count, included"].isna().sum() == 0
    assert res["skeleton outcome frequency, included"].isna().sum() == 0
    return res


def _make_jackknifes_df(
    skeletons: typing.List[np.array], test_knockout: typing.Callable
) -> np.array:
    jackknife_results = (
        pd.concat(
            jackknife_skeleton(skeleton, test_knockout)
            for skeleton in skeletons
        )
        if len(skeletons)  # make robust to numpy types
        else pd.DataFrame(
            {
                "jackknife dose": [],
                "raw jackknife result": [],
                "jackknife result": [],
                "site": [],
            },
        )
    )
    for col, ext in it.product(
        ["jackknife dose", "raw jackknife result", "jackknife result"],
        ["std", "ptp"],
    ):
        jackknife_results[f"{col} {ext}"] = jackknife_results[col]

    agg_df = jackknife_results.groupby(["site"]).agg(
        {
            "jackknife dose": "mean",
            "jackknife dose std": "std",
            "jackknife dose ptp": np.ptp,
            "raw jackknife result": "mean",
            "raw jackknife result std": "std",
            "raw jackknife result ptp": np.ptp,
            "jackknife result": "mean",
            "jackknife result std": "std",
            "jackknife result ptp": np.ptp,
        },
    )
    return agg_df.reset_index()


def _make_skeletons_df(skeletons: typing.List[np.array]) -> np.array:
    records = []
    for i, skeleton in enumerate(skeletons):
        for j, skeleton_value in enumerate(skeleton):
            records.append(
                {
                    "skeleton": i,
                    "skeleton dose": skeleton.astype(bool).sum(),
                    "skeleton exclusion order": skeleton_value or np.nan,
                    "skeleton outcome": bool(skeleton_value),
                    "site": j,
                },
            )

    res = pd.DataFrame.from_records(
        records
        or {
            "skeleton": [],
            "skeleton dose": [],
            "skeleton exclusion order": [],
            "skeleton outcome": [],
            "site": [],
        },
    )
    assert len(res) == sum(map(len, skeletons))
    return res


def _make_skeletonization_df(skeletons_df: pd.DataFrame) -> pd.DataFrame:
    skeletons_df["skeleton outcome count"] = 1
    skeletons_df["skeleton outcome frequency"] = 1
    for col, ext in it.product(
        ["skeleton dose", "skeleton exclusion order"],
        ["std", "ptp"],
    ):
        skeletons_df[f"{col} {ext}"] = skeletons_df[col]
    agg_df = (
        skeletons_df.groupby(
            ["site", "skeleton outcome"],
        )
        .agg(
            {
                "skeleton dose": "mean",
                "skeleton dose std": "std",
                "skeleton dose ptp": np.ptp,
                "skeleton exclusion order": "mean",
                "skeleton exclusion order std": "std",
                "skeleton exclusion order ptp": np.ptp,
                "skeleton outcome count": "sum",
                "skeleton outcome frequency": "sum",
            },
        )
        .reset_index()
    )
    agg_df["skeleton outcome frequency"] /= skeletons_df["skeleton"].nunique()
    assert len(agg_df) >= skeletons_df["site"].nunique()

    skeleton_excluded_df = (  # cover empty case
        agg_df[agg_df["skeleton outcome"]] if len(agg_df) else agg_df
    )
    skeleton_included_df = (
        agg_df[~agg_df["skeleton outcome"]] if len(agg_df) else agg_df
    )
    res = pd.DataFrame.merge(
        # errors ignore covers empty case
        skeleton_excluded_df.drop("skeleton outcome", axis=1, errors="ignore"),
        skeleton_included_df.drop("skeleton outcome", axis=1, errors="ignore"),
        how="outer",
        on=["site"],
        suffixes=[", excluded", ", included"],
    )
    assert len(res) == skeletons_df["site"].nunique()
    res.fillna(
        {
            "skeleton outcome count, excluded": 0,
            "skeleton outcome frequency, excluded": 0,
            "skeleton outcome count, included": 0,
            "skeleton outcome frequency, included": 0,
        },
        inplace=True,
    )
    assert res["skeleton outcome count, excluded"].isna().sum() == 0
    assert res["skeleton outcome frequency, excluded"].isna().sum() == 0
    assert res["skeleton outcome count, included"].isna().sum() == 0
    assert res["skeleton outcome frequency, included"].isna().sum() == 0
    return res.astype(
        {
            "skeleton outcome count, excluded": int,
            "skeleton outcome count, included": int,
        },
    )

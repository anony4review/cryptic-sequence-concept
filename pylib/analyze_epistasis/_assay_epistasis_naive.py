import typing

import numpy as np


def assay_epistasis_naive(
    skeletons_description_df: typing.Sequence[np.array],
    exclusion_frequency_thresh: float = 0.3,
    jackknife_severity_thresh: float = 0.2,
) -> dict:
    """Estimate the number of sites with epistatic fitness effects via skeleton
    samples.

    This function first subsets skeleton sites with exclusion frequencies below
    the specified threshold. Then, it uses a quantile of jackknife severity from
    the subset data to estimate a second threshold.  Finally, among the entire
    set of genome sites appearing in at least one skeleton, sites exceeding both
    thresholds are counted as epistasis sites.

    'Exclusion frequency' refers to the proportion of skeletons where a site is
    not retained. 'Jackknife severity' measures the impact of removing a site on
    the fitness of skeletons that include it. 'Skeletonization' is the process
    of removing sites from a genome until removing any more would affect
    detectability of fitness impact.

    Parameters
    ----------
    skeletons_description_df : pandas.DataFrame
        A DataFrame describing replicate skeletonizations. It should be produced
        by the `describe_skeletons` function. Providing at least ten to twenty
        replicate skeletons is recommended.

    exclusion_frequency_thresh : float, default 0.3
        The threshold for excluding sites based on their frequency of occurrence
        in skeletonization outcomes.

    jackknife_severity_thresh : float, default 0.2
        The quantile used to determine the threshold for jackknife severity.

    Returns
    -------
    dict
        A dictionary containing the estimated number of epistasis sites, along
        with the cutoff values used for skeleton exclusion frequency and
        jackknife severity.

    Notes
    -----
    Plotting the relationship between exclusion frequency and jackknife severity
    can help in determining appropriate cutoff values. This also verifies the
    expected distribution where some sites (e.g., additive sites) show a
    decrease in jackknife knockout severity with increased exclusion frequency.
    Epistasis sites are typically characterized by both high exclusion frequency
    and high jackknife severity.
    """
    rarely_excluded_mask = (
        skeletons_description_df["skeleton outcome frequency, excluded"]
        <= exclusion_frequency_thresh
    )
    rarely_excluded = skeletons_description_df[rarely_excluded_mask]

    severity_thresh = rarely_excluded["jackknife result"].quantile(
        jackknife_severity_thresh,
    )
    severity_mask = (
        skeletons_description_df["jackknife result"] >= severity_thresh
    )

    commonly_excluded_mask = ~rarely_excluded_mask
    estimate = (commonly_excluded_mask & severity_mask).sum()
    return {
        "num epistasis sites estimate": estimate,
        "exclusion frequency cutoff": exclusion_frequency_thresh,
        "jackknife severity cutoff": severity_thresh,
    }

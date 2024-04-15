import typing
import warnings

from iterpop import iterpop as ip
import numpy as np

from ..auxlib._jackknife_mark_recapture_estimate import (
    jackknife_mark_recapture_estimate,
)


def assay_agnostic_naive(
    skeletons_description_df: typing.Sequence[np.array],
) -> dict:
    """Apply capture/release estimation over sites included within replicate
    skeletonizations to estimate the prevalence of functional sites (i.e.,
    epistatis and/or additive effects).

    The mark-recapture estimation procedure is described to handle scenarios
    involving variable capture probabilities (e.g., trap-shyness).

    Parameters
    ----------
    skeletons_description_df : pandas.DataFrame
        DataFrame describing replicate skeletonizations, produced by
        `describe_skeletons`.

        At least five skeletons should be provided for good-quality estimation.

    Returns
    -------
    dict
        A dictionary with the following keys:
        - 'num sites estimate': The estimated total number of functional sites
        within the genome, based on mark-recapture estimation over functional
        sites detected by skeletonization.
        - 'num sites 95% CI': A tuple containing lower and upper bound for a
        95% confidence interval for the estimated number of functional sites.

    Raises
    ------
    Warning
        If fewer than five skeletons are provided, a warning is raised to inform
        the user that the estimation may not be reliable.

    See Also
    --------
    jackknife_mark_recapture_estimate
        For details on the mark-recapture estimation procedure and references
        to the literature describing the method.
    """
    num_skeletons = ip.pophomogeneous(
        skeletons_description_df["skeleton outcome count, included"]
        + skeletons_description_df["skeleton outcome count, excluded"]
    )
    if num_skeletons < 5:
        warnings.warn(
            "Burnham and Overton recommend at least five capture/recapture "
            f"observations, but only {num_skeletons} provided.",
        )

    f = np.bincount(
        skeletons_description_df["skeleton outcome count, included"].astype(
            int
        ),
    )[1:].copy()
    f.resize(num_skeletons)
    estimate, ci = jackknife_mark_recapture_estimate(f)

    return {
        "num sites estimate": estimate,
        "num sites 95% CI": ci,
    }

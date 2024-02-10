from scipy import stats as scipy_stats

from .. import orchestration as orch
from ...analysis import score_competition
from ...common import records as rec
from ._impl import add_competition


def dispatch_depended_assays(assayDocument: dict) -> int:
    return 0


def dispatch_depended_competitions(assayDocument: dict) -> int:
    if orch.get_num_assay_competitions(assayDocument["assayId"]):
        return 0

    num_dispatched = 0
    for replicate in range(100):
        add_competition(
            assayDocument=assayDocument,
            genomeIdAlpha=assayDocument["genomeIdAlpha"],
            genomeIdBeta=assayDocument["genomeIdAlpha"],
            knockoutSites="",
            competitionDesignation={"replicate": replicate},
        )
        num_dispatched += 1

    return num_dispatched


def finalize_result(assayDocument: dict) -> dict:
    competition_results = rec.get_assay_competition_results(
        assayDocument["assayId"],
    )
    competition_scores = [
        m
        * score_competition(
            resultNumAlpha=resultNumAlpha,
            resultNumBeta=resultNumBeta,
            resultUpdatesElapsed=resultUpdatesElapsed,
        )
        for resultNumAlpha, resultNumBeta, resultUpdatesElapsed in competition_results
        for m in [1, 0, -1.0]
    ]
    result = {
        "sampledScoreQuantiles": {
            p: scipy_stats.scoreatpercentile(competition_scores, p)
            for p in range(0, 101)
        },
        "sampledScores": competition_scores,
    }
    rec.add_assay_result(
        assayId=assayDocument["assayId"],
        assayResult=result,
        submissionId=assayDocument["submissionId"],
        userEmail=assayDocument["userEmail"],
    )
    orch.complete_assay(assayDocument["assayId"])
    return result

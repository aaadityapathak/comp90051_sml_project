# Histogram gradient boosting results

Implementation: [`models/histogram_gradient_boosting/`](../../models/histogram_gradient_boosting/).
The historical [reported summary](source_provenance/reported_summary.docx)
contains four-arm, ten-fold results transcribed from prior output. Its raw
prediction and tuning bundle is absent from this checkout. Those scores are
reported evidence, not independently recalculated results.

The runner and checker now use `formal/` beneath this folder. Fresh runs can
generate the missing artifacts; none were regenerated during this cleanup.
The [observations](../../docs/Histogram_Gradient_Boosting_Observations.md) explain
the reported winner, tuning scope, missing evidence and next experiments.

# Preregistration: T0 NextGQA 89-Video CPU Reproduction Benchmark

## Experiment Description
CPU-only 89-video/255-event retrieval reproduction using the current pipeline of IRIS.

## Metrics Definition

### Primary Metric
- **Recall@5**: Fraction of events where at least one of the top 5 retrieved frames falls within the ground-truth temporal interval.

### Secondary Metrics
- **Recall@1**: Recall evaluated at top 1 retrieved frame.
- **Recall@10**: Recall evaluated at top 10 retrieved frames.
- **MRR (Mean Reciprocal Rank)**: Reciprocal rank of the first retrieved frame falling in the ground-truth interval, or 0 if none.
- **mAP (mean Average Precision)**: Mean Average Precision over all events, where AP is computed as:
  $$AP = \frac{\sum_{i=1}^K P(i) \cdot rel(i)}{\sum_{i=1}^K rel(i)}$$
  where $K$ is the video-specific matched frame budget, $P(i)$ is precision at rank $i$, and $rel(i)$ is 1 if the $i$-th ranked frame falls in the ground truth interval, 0 otherwise. If no relevant frames are retrieved, AP is 0.0.
- **Precision@5**: Average precision at top 5 retrieved frames (average fraction of top 5 frames that are relevant).
- **Retrieval Latency**: CPU inference/retrieval time per query in seconds.
- **Index-Build Time**: Time taken to parse and index the video.
- **Frames Retained**: Average number of frames selected per video.
- **Retention Percentage**: Fraction of total video frames that are selected (admitted).
- **Index Size**: Serialized index size on disk.
- **Peak Process Memory**: Peak RAM usage recorded during the run.

### Diagnostic Metric
- **WindowCoverage at 10% retention**: Fraction of events where at least one frame from the ground-truth interval is selected (admitted) at the 10% retention budget.
  *Note: Saturated in previous runs (Proposed: 1.0, Random: 0.9997).*

## Pre-registered Expectations
- Proposed graph Recall@5 remains in the neighborhood of the previous value: **0.5765**.
- Proposed graph strongly exceeds uniform and random at Recall@5.
- The proposed graph vs Hybrid PPR Recall@10 difference (0.7137 vs 0.7216) may not be statistically significant.
- Updated code may change exact values; any change must be explained rather than tuned away.

## Reproduction Warning Threshold
- If Proposed Graph Recall@5 differs from **0.5765** by more than **0.05** absolute (i.e. < 0.5265 or > 0.6265), pause after the run to diagnose provenance and configurations.
- Apply the same **0.05** absolute warning threshold to Recall@10 (reference: **0.7137**).
- No parameters or code may be altered after viewing final test results.

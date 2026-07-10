# IRIS Curated Test Dataset (NExT-QA Style)

This directory contains a curated benchmark dataset of 10 short video clips (each under 15 seconds) for evaluating the IRIS video QA pipeline. 

## Source Clips
The clips are pulled from two highly reliable and public repositories:
1.  **Intel IoT SDK Sample Videos**: Curated video sequences representing daily activities (traffic, pedestrians, retail stores, classroom environments). These clips are hosted on GitHub.
2.  **w3schools Public Domain Video Samples**: Standard short video clips (`movie.mp4` and `mov_bbb.mp4`) hosted on w3schools.

By using these public clips, we completely bypass the fragile and rate-limited Flickr API/web blocks that affect the full-sized VidOR/NExT-QA dataset, while maintaining perfect offline compatibility and fast runtimes during pipeline evaluation.

## Annotations
The questions and answers are manually annotated following the **NExT-QA** dataset's semantic design:
*   **Causal (CH/CW)**: Reasoning about "why" and "how" events happen (e.g. *Why do the cars slow down near the intersection?*).
*   **Temporal (TN/TC)**: Reasoning about what happened before, after, or next (e.g. *What does the second person do after entering the walkway?*).
*   **Descriptive (DC/DL)**: Identifying objects, counts, and locations (e.g. *Where is the blue water bottle placed on the table?*).

The annotations are stored in `test_data/ground_truth.json` with the following schema:
```json
[
  {
    "clip_id": "automobile_detection",
    "video_path": "test_data/clips/automobile_detection.mp4",
    "video_url": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/automobile-detection.mp4",
    "question": "Why do the cars slow down near the intersection?",
    "answer": "to avoid colliding with the turning vehicle",
    "question_type": "causal"
  }
]
```

## Reproducibility and Download
The evaluation suite (`eval_suite.py`) is equipped with an automatic download manager that reads this JSON file and downloads any missing clips directly to the `test_data/clips/` directory on the first execution. No manual setup is required.

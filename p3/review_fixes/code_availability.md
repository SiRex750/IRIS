# Code availability of related systems (review follow-up D)

Checked 2026-10-03 by web search and by reading public pages and the GitHub REST API only. Nothing was downloaded,
cloned or run, so "runnable?" is judged from the README alone. There is no script for this item; the sources are
listed under each entry.

| system | public code found? | link | licence | runnable? |
|---|---|---|---|---|
| CodecSight (arXiv 2604.06036) | **No** | – | – | – (the paper says code comes "upon publication") |
| MVTrack (arXiv 2608.10790) | **No** for this paper | (unrelated repo with the same name: github.com/MobileLLM/MVTrack) | – | – |
| CMC (ASPLOS 2024) | **No** | – | – | – |

## CodecSight: arXiv 2604.06036 (Zou, Chen, Chen, Rajan, Park, Nitin, Tao, Romero, Ustiugov)
- v1 of this arXiv ID was titled "CoStream: Codec-Guided Resource-Efficient System for Video Streaming Analytics";
  v4 (15 Sep 2026) is "CodecSight". Cite the version you read.
- v4 footnote 1 says: "We will release the code upon publication." The paper gives no repository URL.
- A GitHub repository search for "CodecSight" returns 0 results. awesomepapers.io lists the code as "-".
- The paper describes the system as built on vLLM (about 6.1K lines of Python), with H.264 MV extraction via NVDEC /
  PyNvVideoCodec. So a re-implementation would need an NVIDIA GPU stack.
- Sources: https://arxiv.org/abs/2604.06036, https://arxiv.org/html/2604.06036,
  https://awesomepapers.io/computer-vision/papers/2604.06036

## MVTrack: arXiv 2608.10790 (Erregue, Nasrollahi, Escalera; ECCV 2026 workshop)
- The arXiv abstract, comments and HTML full text give no code link and no promise to release code.
- A GitHub search for "MVTrack" finds 19 repositories. None is by these authors or refers to this paper or VIRAT.
- **Name clash to avoid miscitation:** github.com/MobileLLM/MVTrack ("Object tracking with motion vector in video
  bitstream") is a different, earlier project. It was created 2024-02-26, last pushed 2024-02-29, has 5 commits and
  no licence file, and does not cite this paper. Its README lists Python 3.7 + FFmpeg 4.4.1 with a modified
  `h264_cabac.c` for MV extraction, then detector + MV refinement steps. It may be runnable with that old toolchain
  (untested). Because it has no licence, reuse is all rights reserved by default. It is **not** the code of arXiv
  2608.10790.
- Sources: https://arxiv.org/abs/2608.10790, https://arxiv.org/html/2608.10790v1,
  https://github.com/MobileLLM/MVTrack, https://api.github.com/search/repositories?q=MVTrack

## CMC: "Video Transformer Acceleration via CODEC Assisted Matrix Condensing" (Song, Qi, Liu, Jing, Liang; ASPLOS 2024, doi 10.1145/3620665.3640393)
- The first author's homepage lists only [paper] and [cite] links for CMC, with no code.
- The first author's GitHub (songzhuoran, 12 repositories) has no repository named for CMC, codecs or video
  transformers. Two unnamed video repositories exist (`video-block-based-acc`, 1 commit, no README visible;
  `video-frame-based-acc`, "video detection using frame-based mapping tech"). Neither mentions CMC or ASPLOS, and
  neither has a licence. They look like earlier video-detection accelerator work, not CMC.
- The ACM DL page (artifact badges) returned HTTP 403 to the fetcher, so whether CMC has an ACM artifact badge
  is **unverified**. Check it in a browser before claiming "no artifact".
- CMC is an algorithm–accelerator co-design (simulated hardware), so even released code would be a simulator, not a
  deployable gate.
- Sources: https://songzhuoran.github.io/, https://github.com/songzhuoran?tab=repositories,
  https://dl.acm.org/doi/10.1145/3620665.3640393 (403)

## Suggested paper wording
"At the time of writing, no public implementation was available for CodecSight (code promised on publication), MVTrack
or CMC, so we compare against their published descriptions rather than re-running them."

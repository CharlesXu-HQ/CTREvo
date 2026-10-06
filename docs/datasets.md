# Dataset selection

Checked on 2026-10-06. Feature counts refer to original released fields, not embedding dimensions or arbitrarily derived crosses.

| Dataset | Relevant richness | Access / decision |
|---|---|---|
| Criteo Display Advertising Challenge | 13 numeric + 26 categorical, 45,840,617 labeled impressions | Initial adapter. Complete official Azure archive accessible; original shortened download URL returned 404. |
| AntM2C | Paper describes 200+ features, but its first-release footnote specifies 29 ID + 2 text features | Promising richer semantic/text extension. Do not equate headline internal dataset with public release; public files and schema must be verified first. |
| ShareChat RecSys 2023 | Ad/user/context features, historical engagement and embeddings; click and install outcomes | Official challenge requires registration; primary competition task is install prediction. CTR adaptation needs explicit label and timing audit. |
| Taobao advertising | Joined user/ad profiles plus behavior logs | Potentially valuable for multi-sequence studies; smaller flat field count does not capture sequence richness. Access, event cutoff and joins must be checked before use. |
| Avazu | About 22 original predictor fields | Accessible alternative but fewer original fields than Criteo; not selected as the first benchmark. |

Sources:

- [Criteo datasets](https://ailab.criteo.com/ressources/)
- [Official DLRM dataset preparation](https://github.com/facebookresearch/dlrm/blob/main/torchrec_dlrm/README.MD)
- [Full original archive](https://criteostorage.blob.core.windows.net/criteo-research-datasets/kaggle-display-advertising-challenge-dataset.tar.gz)
- [Archive checksum metadata](https://api.figshare.com/v2/articles/5732310): 4,576,820,670 bytes, MD5 `df9b1b3766d9ff91d5ca3eb3d23bed27`. Mirror metadata is not a replacement for original distributor terms.
- [AntM2C paper](https://arxiv.org/abs/2308.16437), especially first-release footnote; [release portal](https://www.atecup.cn/dataSetDetailOpen/1).
- [ShareChat official challenge](https://www.recsyschallenge.com/2023/) and [dataset paper](https://arxiv.org/abs/2308.02568).
- [Taobao benchmark data notes](https://github.com/huaxz1986/BARS/blob/master/datasets/Taobao/README.md).

Criteo's separate 1TB release has 24 days of data and the same broad feature count, but is not the Kaggle release. CTREvo does not substitute one for the other.

The prepared benchmark consumes every labeled input row. Its competition test file is unlabeled and is not used as a scored test set. The host creates an explicit held-out partition from labeled data instead.

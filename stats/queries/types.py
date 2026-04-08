from typing import Literal


Scope = Literal[
    "subject",
    "single_channel",
    "all_channels",
    "single_edge",
    "all_edges",
]

TestKind = Literal[
    "t_test",
    "wilcoxon_rank_sum",
    "spearman",
    "one_way_anova",
    "two_way_anova",
]

CorrectionKind = Literal[
    "fdr_bh",
]

PostHocKind = Literal[
    "tukey_hsd",
]
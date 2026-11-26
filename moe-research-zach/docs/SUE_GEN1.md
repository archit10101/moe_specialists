# SUE engineering generation 1 — support/EMA

Written 2026-11-17T15:29:29-05:00 before any run of this generation.

Measured failure (U2): noisy

Evidence: {"100": {"split_half_r": -0.08446284036272243, "margin_onset": 0.0, "margin_end": 0.00513204000890255, "mi_sem": 0.15774931684905596, "mi_ref_c6": 0.16035937535973796}, "101": {"split_half_r": 0.2149885992372581, "margin_onset": 0.0, "margin_end": 0.0026864020619541407, "mi_sem": 0.15796263806051025, "mi_ref_c6": 0.1471724034831787}}

Change family: support/EMA (prereg section 9). Candidates (50M, seeds 100/101, SHUF controls):

- {"gamma": 0.003, "tau": 0.2, "rho": 0.998, "onset": 100, "layers": null, "probe_passes": 8}

Acceptance: prereg section-8 rule.

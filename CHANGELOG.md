# Changelog

All notable changes to this project should be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-01-13

### Added
- Prepare for pypi publishing

## [0.1.0] - 2026-01-08

### Added

- Initial implementation of polars `TableOne` class based on `tableone` package.
- Support for categorical and continuous variable detection and summary statistics.
- Stratification support with `groupby` parameter.
- Hypothesis testing (Chi-squared, ANOVA, Kruskal-Wallis, T-test, Fisher's exact).
- Multiple testing adjustment.
- Great Tables integration for beautiful outputs.
- CSV, HTML, and LaTeX export support.
- Automated data quality tests (normality, multimodality, outliers) with footnotes.
- Embedded mini-histograms (nanoplots) in table outputs.

[unreleased]: https://github.com/dcronkite/polartableone/compare/v0.1.1..HEAD

[0.1.1]: https://github.com/dcronkite/polartableone/compare/v0.1.0..v0.1.1

[0.1.0]: https://github.com/dcronkite/polartableone/releases/tag/v0.1.0

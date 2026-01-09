import polars as pl
import numpy as np
from typing import List, Optional, Union, Dict, Any, Callable
from scipy import stats
from statsmodels.stats.multitest import multipletests
from great_tables import GT, px, nanoplot_options
from tabulate import tabulate


class TableOne:
    """
    Polars implementation of the tableone library.

    Create an instance of the tableone summary table.

    Parameters
    ----------
    data : pandas DataFrame
        The dataset to be summarised. Rows are observations, columns are
        variables.
    columns : list, optional
        List of columns in the dataset to be included in the final table.
        Setting the argument to None will include all columns by default.
    categorical : list, optional
        List of columns that contain categorical variables.
        If the argument is set to None (or omitted), we attempt to detect
        categorical variables. Set to an empty list to indicate explicitly
        that there are no variables of this type to be included.
    continuous : list, optional
        List of columns that contain continuous variables.
        If the argument is set to None (or omitted), we attempt to detect
        continuous variables. Set to an empty list to indicate explicitly
        that there are no variables of this type to be included.
    groupby : str, default: None
        Optional column for stratifying the final table.
    nonnormal : list, default: None
        List of columns that contain non-normal variables.
    min_max: list, optional
        List of variables that should report minimum and maximum, instead of
        standard deviation (for normal) or Q1-Q3 (for non-normal).
    pval : bool, default: False
        Display computed P-Values.
    pval_adjust : str, optional
        Method used to adjust P-Values for multiple testing.
        The P-values from the unadjusted table (default when pval=True)
        are adjusted to account for the number of total tests that were
        performed.
        These adjustments would be useful when many variables are being
        screened to assess if their distribution varies by the variable in the
        groupby argument.
        For a complete list of methods, see documentation for statsmodels
        multipletests.
        Available methods include ::

        `None` : no correction applied.
        `bonferroni` : one-step correction
        `sidak` : one-step correction
        `holm-sidak` : step down method using Sidak adjustments
        `simes-hochberg` : step-up method (independent)
        `hommel` : closed method based on Simes tests (non-negative)
    pval_digits : int, default=3
        Number of decimal places to display for p-values.
    pval_threshold : float, optional
        Threshold below which p-values are marked with an asterisk (*).
        For example, if set to 0.05, all p-values less than 0.05 will be
        displayed with a trailing asterisk (e.g., '0.012*').
    htest_name : bool, default: False
        Display a column with the names of hypothesis tests.
    htest : dict, optional
        Dictionary of custom hypothesis tests. Keys are variable names and
        values are functions. Functions must take a list of Numpy Arrays as
        the input argument and must return a test result.
        e.g. htest = {'age': myfunc}
    ttest_equal_var : bool, default=False
        Whether to assume equal population variances when performing two-sample
        t-tests. Set to False (default) to use Welch’s t-test, which is more robust
        to unequal variances.
    missing : bool, default: True
        Display a count of null values.
    ddof : int, default: 1
        Degrees of freedom for standard deviation calculations.
    rename : dict, optional
        Dictionary of alternative names for variables.
        e.g. `rename = {'sex':'gender', 'trt':'treatment'}`
    sort : bool or str, optional
        If `True`, sort the variables alphabetically. If a string
        (e.g. `'P-Value'`), sort by the specified column in ascending order.
        Default (`False`) retains the sequence specified in the `columns`
        argument. Currently the only columns supported are: `'Missing'`,
        `'P-Value'`, `'P-Value (adjusted)'`, and `'Test'`.
    limit : int or dict, optional
        Limit to the top N most frequent categories. If int, apply to all
        categorical variables. If dict, apply to the key (e.g. {'sex': 1}).
    order : dict, optional
        Specify an order for categorical variables. Key is the variable, value
        is a list of values in order.  {e.g. 'sex': ['f', 'm', 'other']}
    label_suffix : bool, default: True
        Append summary type (e.g. "mean (SD); median [Q1,Q3], n (%); ") to the
        row label.
    decimals : int or dict, optional
        Number of decimal places to display. An integer applies the rule to all
        variables (default: 1). A dictionary (e.g. `decimals = {'age': 0)`)
        applies the rule per variable, defaulting to 1 place for unspecified
        variables. For continuous variables, applies to all summary statistics
        (e.g. mean and standard deviation). For categorical variables, applies
        to percentage only.
    overall : bool, default: True
        If True, add an "overall" column to the table. Smd and p-value
        calculations are performed only using stratified columns.
    row_percent : bool, optional
        If True, compute "n (%)" percentages for categorical variables across
        "groupby" rows rather than columns.
    display_all : bool, default: False
        If True, set pd. display_options to display all columns and rows.
    dip_test : bool, default: False
        Run Hartigan's Dip Test for multimodality. If variables are found to
        have multimodal distributions, a remark will be added below the
        Table 1.
    normal_test : bool, default: False
        Test the null hypothesis that a sample come from a normal distribution.
        Uses scipy.stats.normaltest. If variables are found to have non-normal
        distributions, a remark will be added below the Table 1.
    tukey_test : bool, default: False
        Run Tukey's test for far outliers. If variables are found to
        have far outliers, a remark will be added below the Table 1.
    include_null : bool, default: True
        Include None/Null values for categorical variables by treating them as a
        category level.
    show_histograms : bool | int, default=False
        Whether to include mini-histograms for continuous variables. If True, bin size is 10, or specify int to
        set bin size for histogram.
    clip_histograms : tuple or None, default (1, 99)
        If show_histograms=True, specify a (lower_percentile, upper_percentile) range to clip the
        data before generating histograms. This reduces the influence of extreme outliers.
        For example, (1, 99) clips to the 1st and 99th percentiles.
        Set to None to disable clipping and use the full range of values.

    Attributes
    ----------
    _table : dataframe
        Summary of the data (i.e., the "Table 1").

    Examples
    --------
    >>> df = pl.DataFrame({'size': [1, 2, 60, 1, 1],
    ...                   'fruit': ['peach', 'orange', 'peach', 'peach',
    ...                             'orange'],
    ...                   'tasty': ['yes', 'yes', 'no', 'yes', 'no']})

    >>> df

    shape: (5, 3)
    ┌──────┬────────┬───────┐
    │ size ┆ fruit  ┆ tasty │
    │ ---  ┆ ---    ┆ ---   │
    │ i64  ┆ str    ┆ str   │
    ╞══════╪════════╪═══════╡
    │ 1    ┆ peach  ┆ yes   │
    │ 2    ┆ orange ┆ yes   │
    │ 60   ┆ peach  ┆ no    │
    │ 1    ┆ peach  ┆ yes   │
    │ 1    ┆ orange ┆ no    │
    └──────┴────────┴───────┘

    >>> TableOne(df, overall=False, groupby="fruit", pval=True).tabulate()
    +-----------------+---------+-----------+-----------+-------------+-----------+
    |                 | Level   | Missing   | orange    | peach       | P-Value   |
    +=================+=========+===========+===========+=============+===========+
    | n               |         |           | 2         | 3           |           |
    +-----------------+---------+-----------+-----------+-------------+-----------+
    | size, mean (SD) |         | 0         | 1.5 (0.7) | 20.7 (34.1) | 0.433     |
    +-----------------+---------+-----------+-----------+-------------+-----------+
    | tasty, n (%)    | no      |           | 1 (50.0)  | 1 (33.3)    | 1.000     |
    +-----------------+---------+-----------+-----------+-------------+-----------+
    | tasty, n (%)    | yes     |           | 1 (50.0)  | 2 (66.7)    |           |
    +-----------------+---------+-----------+-----------+-------------+-----------+
    ...
    """

    def __init__(
            self,
            data: pl.DataFrame,
            columns: Optional[List[str]] = None,
            categorical: Optional[List[str]] = None,
            continuous: Optional[List[str]] = None,
            groupby: Optional[str] = None,
            nonnormal: Optional[List[str]] = None,
            min_max: Optional[List[str]] = None,
            pval: bool = False,
            pval_adjust: Optional[str] = None,
            pval_digits: int = 3,
            pval_threshold: Optional[float] = None,
            htest_name: bool = False,
            htest: Optional[Dict[str, Callable]] = None,
            ttest_equal_var: bool = False,
            missing: bool = True,
            ddof: int = 1,
            rename: Optional[Dict[str, str]] = None,
            sort: Union[bool, str] = False,
            limit: Optional[Union[int, Dict[str, int]]] = None,
            order: Optional[Dict[str, List[Any]]] = None,
            label_suffix: bool = True,
            decimals: Union[int, Dict[str, int]] = 1,
            overall: bool = True,
            row_percent: bool = False,
            display_all: bool = False,
            dip_test: bool = False,
            normal_test: bool = False,
            tukey_test: bool = False,
            include_null: bool = True,
            show_histograms: Union[bool, int] = False,
            clip_histograms: Optional[tuple] = (1, 99),
            labels: Optional[Dict[str, str]] = None,
            smd: bool = False,
    ):
        self.data = data
        self.columns = columns or data.columns
        self.categorical = categorical
        self.continuous = continuous
        self.groupby = groupby
        self.nonnormal = nonnormal or []
        self.min_max = min_max or []
        if isinstance(self.min_max, dict):
            self.min_max = [k for k, v in self.min_max.items() if v]

        self.pval = pval
        self.pval_adjust = pval_adjust
        self.pval_digits = pval_digits
        self.pval_threshold = pval_threshold
        self.htest_name = htest_name
        self.htest = htest or {}
        self.ttest_equal_var = ttest_equal_var
        self.missing = missing
        self.ddof = ddof
        self.rename = rename or {}
        self.sort = sort
        self.limit = limit
        self.order = order or {}
        self.label_suffix = label_suffix
        self.decimals = decimals
        self.overall = overall
        self.row_percent = row_percent
        self.display_all = display_all
        self.dip_test = dip_test
        self.normal_test = normal_test
        self.tukey_test = tukey_test
        self.include_null = include_null
        self.show_histograms = 10 if show_histograms is True else show_histograms
        self.clip_histograms = clip_histograms
        self.labels = labels or {}
        self.smd = smd

        if self.display_all:
            pl.Config.set_tbl_rows(-1)
            pl.Config.set_tbl_cols(-1)

        # rename columns if requested
        if self.rename:
            # check for duplicate display names
            display_names = list(self.rename.values())
            if len(display_names) != len(set(display_names)):
                raise ValueError('Duplicate display names in rename parameter')

            # check if original names exist
            for old_name in self.rename:
                if old_name not in self.data.columns:
                    raise ValueError(f'Column {old_name} not found in data for renaming')

            self.data = self.data.rename(self.rename)
            self.columns = [self.rename.get(c, c) for c in self.columns]
            if self.groupby in self.rename:
                self.groupby = self.rename[self.groupby]

        # internal state
        self._table = None  # stores visual representation (e.g., for histograms)
        self._itable = None  # table for GT-generation (stores vectors)
        self._categorical_vars = []
        self._continuous_vars = []
        self._limit_levels = {}
        self._footnotes = []

        # validation and classification
        self._validate_inputs()
        self._classify_variables()

        # compute the table
        self._compute_table()

    def _validate_inputs(self):
        # all specified variables exist in DataFrame
        for col in self.columns:
            if col not in self.data.columns:
                raise ValueError(f'Column {col} not found in data')

        if self.groupby and self.groupby not in self.data.columns:
            raise ValueError(f'Groupby column {self.groupby} not found in data')

    def _classify_variables(self):
        # auto-detect categorical vs continuous if not provided
        if self.categorical is None:
            self._categorical_vars = []
            for col in self.columns:
                if col == self.groupby:
                    continue

                dtype = self.data.schema[col]
                # assume all non-numerical and date columns are categorical
                if not dtype.is_numeric() and not dtype.is_temporal():
                    self._categorical_vars.append(col)
                elif dtype.is_numeric():
                    # check proportion of unique values if numerical
                    # likely_flag = 1.0 * data[var].nunique()/data[var].count() < 0.005
                    count = self.data[col].count()
                    if count > 0:
                        nunique = self.data[col].n_unique()
                        if 1.0 * nunique / count < 0.005:
                            self._categorical_vars.append(col)
        else:
            self._categorical_vars = self.categorical or []

        if self.continuous is None:
            self._continuous_vars = [c for c in self.columns if c not in self._categorical_vars and c != self.groupby]
        else:
            self._continuous_vars = self.continuous or []

        # cast continuous variables to numeric if they are strings
        for var in self._continuous_vars:
            if self.data.schema[var] == pl.Utf8:
                try:
                    self.data = self.data.with_columns(pl.col(var).str.strip_chars().cast(pl.Float64))
                except Exception:
                    # if it fails, maybe it's not actually continuous? 
                    # for now just let it raise error later or keep as is
                    pass

        # ensure no overlap
        overlap = set(self._categorical_vars) & set(self._continuous_vars)
        if overlap:
            raise ValueError(f'Variables {overlap} specified as both categorical and continuous')

    def _compute_table(self):
        # 0. Pre-calculate limits for categorical variables
        if self.limit:
            for var in self._categorical_vars:
                l = self.limit
                if isinstance(l, dict):
                    l = l.get(var)

                if l is not None and l > 0:
                    data = self.data[var]
                    if not self.include_null:
                        data = data.drop_nulls()

                    # value_counts returns a df with column var and 'count' (or 'n' depending on polars version)
                    # in newer polars it's 'count'
                    vc = data.value_counts(sort=True)
                    count_col = vc.columns[1]
                    top = vc.head(l)[var].to_list()
                    self._limit_levels[var] = top

        # 0.1 Data quality tests (Phase 15)
        non_normal_vars = []
        multimodal_vars = []
        outlier_vars = []

        for var in self._continuous_vars:
            data = self.data[var].drop_nulls()
            if len(data) < 3:
                continue

            # normality test
            if self.normal_test:
                stat, p = stats.normaltest(data.to_numpy())
                if p < 0.05:
                    non_normal_vars.append(var)
                    # automatically populate nonnormal if not manually specified
                    if var not in self.nonnormal:
                        self.nonnormal.append(var)

            # dip test (multimodality)
            if self.dip_test:
                try:
                    from diptest import diptest
                    dip, p = diptest(data.to_numpy())
                    if p < 0.05:
                        multimodal_vars.append(var)
                except ImportError:
                    pass  # or warn

            # tukey test (outliers)
            if self.tukey_test:
                q1 = data.quantile(0.25)
                q3 = data.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 3.0 * iqr
                upper = q3 + 3.0 * iqr
                if ((data < lower) | (data > upper)).any():
                    outlier_vars.append(var)

        if non_normal_vars:
            self._footnotes.append(f"Variables with non-normal distributions (p < 0.05): {', '.join(non_normal_vars)}")
        if multimodal_vars:
            self._footnotes.append(
                f"Variables with multimodal distributions (Hartigan Dip Test, p < 0.05): {', '.join(multimodal_vars)}")
        if outlier_vars:
            self._footnotes.append(f"Variables with far outliers (Tukey, 3×IQR): {', '.join(outlier_vars)}")

        # compute Overall stats
        overall_stats = self._compute_stats(self.data, label='Overall')

        # compute Stratified stats if groupby is present
        stratified_stats = []
        if self.groupby:
            # handle nulls in groupby
            groups = self.data[self.groupby].unique().drop_nulls().sort()
            for group in groups:
                group_data = self.data.filter(pl.col(self.groupby) == group)
                stratified_stats.append(self._compute_stats(group_data, label=str(group)))

        # compute p-values
        p_values = None
        if self.pval and self.groupby:
            p_values = self._compute_pvalues()

        # build the final dataframe
        self._itable = self._build_summary_frame(overall_stats, stratified_stats, p_values)
        if self.show_histograms:
            self._table = self._build_summary_frame(overall_stats, stratified_stats, p_values, hist_blocks=True)
        else:
            self._table = self._itable

    def _compute_stats(self, df: pl.DataFrame, label: str) -> Dict[str, Any]:
        stats_dict = {'label': label, 'n': len(df), 'vars': {}}

        # categorical
        for var in self._categorical_vars:
            stats_dict['vars'][var] = self._get_categorical_stats(df, var)

        # continuous
        for var in self._continuous_vars:
            stats_dict['vars'][var] = self._get_continuous_stats(df, var)

        return stats_dict

    def _get_categorical_stats(self, df: pl.DataFrame, var: str) -> Dict[str, Any]:
        total_n = len(df)
        if total_n == 0:
            return {
                'counts': pl.DataFrame({var: [], 'n': [], 'percent': []}),
                'total_n': 0,
                'missing': 0
            }

        # handle nulls
        working_df = df.select(var)
        if not self.include_null:
            working_df = working_df.filter(pl.col(var).is_not_null())
            total_n = len(working_df)

        # handle limit/Other aggregation
        if var in self._limit_levels:
            top_levels = self._limit_levels[var]
            # convert top_levels to strings for comparison after cast
            top_levels_str = [str(t) for t in top_levels]
            working_df = working_df.with_columns(
                pl.col(var).cast(pl.Utf8)
                .map_elements(lambda x: x if x in top_levels_str else 'Other', return_dtype=pl.Utf8)
                .alias(var)
            )

        counts = working_df.group_by(var).agg(pl.len().alias('n'))

        if total_n > 0:
            counts = counts.with_columns(
                (pl.col('n') / total_n * 100).alias('percent')
            )
        else:
            counts = counts.with_columns(
                pl.lit(0.0).alias('percent')
            )

        # sort by order if provided, otherwise by value
        if var in self.order:
            order_list = self.order[var]
            # if Other was created and not in order, it will go to the end
            counts = counts.sort(
                pl.col(var).map_elements(lambda x: order_list.index(x) if x in order_list else len(order_list),
                                         return_dtype=pl.Int32)
            )
        else:
            # sorting with nulls, spec says 'Other' category appears last
            if var in self._limit_levels:
                counts = counts.with_columns(
                    pl.when(pl.col(var) == 'Other').then(pl.lit(1)).otherwise(pl.lit(0)).alias('_is_other')
                ).sort('_is_other', var, descending=[False, False], nulls_last=True).drop('_is_other')
            else:
                counts = counts.sort(var, descending=False, nulls_last=True)

        return {
            'counts': counts,
            'total_n': total_n,
            'missing': df[var].is_null().sum()
        }

    def _get_continuous_stats(self, df: pl.DataFrame, var: str) -> Dict[str, Any]:
        data = df[var].drop_nulls()
        n = len(data)
        missing = df[var].is_null().sum()

        if n == 0:
            return {
                'mean': None, 'std': None, 'median': None,
                'q1': None, 'q3': None, 'min': None, 'max': None,
                'n': 0, 'missing': missing
            }

        if var in self.nonnormal:
            median = data.median()
            q1 = data.quantile(0.25, interpolation='linear')  # align with numpy, R
            q3 = data.quantile(0.75, interpolation='linear')
            stats = {
                'median': median,
                'q1': q1,
                'q3': q3,
                'n': n,
                'missing': missing,
                'min': data.min(),
                'max': data.max()
            }
        else:
            mean = data.mean()
            std = data.std(ddof=self.ddof)
            stats = {
                'mean': mean, 'std': std,
                'n': n, 'missing': missing,
                'min': data.min(), 'max': data.max()
            }

        if self.show_histograms:
            hist_data = data
            if self.clip_histograms:
                low = data.quantile(self.clip_histograms[0] / 100, interpolation='linear')
                high = data.quantile(self.clip_histograms[1] / 100, interpolation='linear')
                # polars clip is on Series
                hist_data = data.clip(low, high)

            # simple binning
            counts, _ = np.histogram(hist_data.to_numpy(), bins=self.show_histograms)
            blocks = '▁▂▃▄▅▆▇█'
            hist_normalized = np.floor((counts / counts.max()) * (len(blocks) - 1)).astype(int)

            stats['hist_blocks'] = ''.join(blocks[i] for i in hist_normalized)
            stats['hist_counts'] = ' '.join(map(str, counts))

        return stats

    def _compute_pvalues(self) -> Dict[str, Any]:
        pvals = {}
        if not self.groupby:
            return pvals

        groups = self.data[self.groupby].unique().sort().to_list()

        for var in self._categorical_vars:
            # custom test
            if var in self.htest:
                try:
                    # prepare groups for custom test
                    group_vals = []
                    for g in groups:
                        if g is None: continue
                        vals = self.data.filter(pl.col(self.groupby) == g)[var].drop_nulls().to_numpy()
                        group_vals.append(vals)

                    res = self.htest[var](*group_vals)
                    if isinstance(res, (list, tuple)):
                        p = res[1]
                    else:
                        p = res
                    pvals[var] = {'p': p, 'test': getattr(self.htest[var], '__name__', 'Custom')}
                    continue
                except Exception as e:
                    # graceful error with variable name
                    # for now just log or print, then fallback or nan
                    pvals[var] = {'p': np.nan, 'test': 'Error'}
                    continue

            # chi-square or Fisher's test
            # create contingency table
            ct_df = self.data.pivot(
                index=var,
                on=self.groupby,
                values=var,
                aggregate_function='len'
            ).fill_null(0).drop(var)

            ct = ct_df.to_numpy()

            try:
                # fisher's exact if any expected freq < 5 and it's a 2x2 table
                # stats.chi2_contingency returns expected_freqs as 4th element
                chi2, p, dof, ex = stats.chi2_contingency(ct)
                if (ex < 5).any() and ct.shape == (2, 2):
                    try:
                        oddsratio, p = stats.fisher_exact(ct)
                        pvals[var] = {'p': p, 'test': "Fisher's exact"}
                    except Exception:
                        pvals[var] = {'p': p, 'test': 'Chi-squared'}
                else:
                    pvals[var] = {'p': p, 'test': 'Chi-squared'}
            except Exception:
                pvals[var] = {'p': np.nan, 'test': 'Chi-squared'}

        for var in self._continuous_vars:
            if var in self.htest:
                try:
                    group_vals = []
                    for g in groups:
                        if g is None: continue
                        vals = self.data.filter(pl.col(self.groupby) == g)[var].drop_nulls().to_numpy()
                        group_vals.append(vals)

                    res = self.htest[var](*group_vals)
                    if isinstance(res, (list, tuple)):
                        p = res[1]
                    else:
                        p = res
                    pvals[var] = {'p': p, 'test': getattr(self.htest[var], '__name__', 'Custom')}
                    continue
                except Exception:
                    pvals[var] = {'p': np.nan, 'test': 'Error'}
                    continue

            group_data = [
                self.data.filter(pl.col(self.groupby) == g)[var].drop_nulls().to_numpy()
                for g in groups if g is not None
            ]

            # remove empty groups
            group_data = [g for g in group_data if len(g) > 0]

            if len(group_data) < 2:
                pvals[var] = {'p': np.nan, 'test': '—'}
                continue

            if var in self.nonnormal:
                # kruskal-Wallis
                try:
                    stat, p = stats.kruskal(*group_data)
                    pvals[var] = {'p': p, 'test': 'Kruskal-Wallis'}
                except Exception:
                    pvals[var] = {'p': np.nan, 'test': 'Kruskal-Wallis'}
            else:
                # aNOVA or t-test
                try:
                    if len(group_data) == 2:
                        # t-test
                        stat, p = stats.ttest_ind(group_data[0], group_data[1], equal_var=self.ttest_equal_var)
                        test_name = "Student's t-test" if self.ttest_equal_var else "Welch's t-test"
                        pvals[var] = {'p': p, 'test': test_name}
                    else:
                        stat, p = stats.f_oneway(*group_data)
                        pvals[var] = {'p': p, 'test': 'ANOVA (one-way)'}
                except Exception:
                    pvals[var] = {'p': np.nan, 'test': 'ANOVA (one-way)'}

        # multiple testing correction
        if self.pval_adjust and pvals:
            raw_pvals = [v['p'] for v in pvals.values() if not np.isnan(v['p'])]
            if raw_pvals:
                rejected, adjusted, _, _ = multipletests(raw_pvals, method=self.pval_adjust)
                adj_idx = 0
                for var in pvals:
                    if not np.isnan(pvals[var]['p']):
                        pvals[var]['p_adj'] = adjusted[adj_idx]
                        adj_idx += 1
                    else:
                        pvals[var]['p_adj'] = np.nan

        return pvals

    def _build_summary_frame(self, overall_stats, stratified_stats, p_values, *, hist_blocks=False):
        rows = []

        # header row: n
        n_row = {'': 'n', 'Level': ''}
        if self.missing:
            n_row['Missing'] = ''

        if self.overall:
            n_row['Overall'] = str(overall_stats['n'])
            if self.show_histograms:
                n_row['Histogram'] = ''

        for s in stratified_stats:
            n_row[s['label']] = str(s['n'])
            if self.show_histograms:
                n_row[f"{s['label']} Hist"] = ''

        if self.pval and self.groupby:
            n_row['P-Value'] = ''
            if self.pval_adjust:
                n_row['P-Value (adjusted)'] = ''
            if self.htest_name:
                n_row['Test'] = ''

        rows.append(n_row)

        # data rows
        for var in self.columns:
            if var == self.groupby:
                continue

            if var in self._categorical_vars:
                rows.extend(self._build_categorical_rows(var, overall_stats, stratified_stats, p_values))
            elif var in self._continuous_vars:
                rows.append(self._build_continuous_row(var, overall_stats, stratified_stats, p_values,
                                                       hist_blocks=hist_blocks))

        out_df = pl.from_dicts(rows)

        # ensure column order
        cols = ['', 'Level']
        if self.missing:
            cols.append('Missing')
        if self.overall:
            cols.append('Overall')
            if self.show_histograms:
                cols.append('Histogram')
        for s in stratified_stats:
            cols.append(s['label'])
            if self.show_histograms:
                cols.append(f"{s['label']} Hist")
        if self.pval and self.groupby:
            cols.append('P-Value')
            if self.pval_adjust:
                cols.append('P-Value (adjusted)')
            if self.htest_name:
                cols.append('Test')

        return out_df.select(cols)

    def _format_value(self, val, var, is_percent=False):
        if val is None:
            return '—'

        d = self.decimals
        if isinstance(d, dict):
            d = d.get(var, 1)

        return f'{val:.{d}f}'

    def _format_pval(self, p):
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return ''

        digits = self.pval_digits
        threshold = 10 ** (-digits)

        if p < threshold:
            formatted = f'< {threshold:.{digits}f}'
        else:
            formatted = f'{p:.{digits}f}'

        # significance flagging
        if self.pval_threshold is not None:
            if isinstance(self.pval_threshold, list):
                sorted_thresholds = sorted(self.pval_threshold)
                for i, t in enumerate(sorted_thresholds):
                    if p < t:
                        formatted += '*' * (len(sorted_thresholds) - i)
                        break
            elif p < self.pval_threshold:
                formatted += '*'

        return formatted

    def _build_categorical_rows(self, var, overall_stats, stratified_stats, p_values):
        rows = []

        # get all unique levels across overall and stratified
        all_levels = overall_stats['vars'][var]['counts'][var].to_list()

        label = self.labels.get(var, var)
        if self.label_suffix:
            label = f'{label}, n (%)'

        for i, level in enumerate(all_levels):
            row = {
                '': label,
                'Level': str(level) if level is not None else 'None'
            }

            if self.missing:
                row['Missing'] = ''

            # if row_percent is True, we need the total n for this level across all stratified groups
            if self.row_percent and self.groupby:
                row_total = 0
                for s in stratified_stats:
                    if level is None:
                        stats_df = s['vars'][var]['counts'].filter(pl.col(var).is_null())
                    else:
                        stats_df = s['vars'][var]['counts'].filter(pl.col(var) == level)
                    if len(stats_df) > 0:
                        row_total += stats_df['n'][0]
            else:
                row_total = None

            if self.overall:
                if level is None:
                    stats_df = overall_stats['vars'][var]['counts'].filter(pl.col(var).is_null())
                else:
                    stats_df = overall_stats['vars'][var]['counts'].filter(pl.col(var) == level)

                if len(stats_df) > 0:
                    n = stats_df['n'][0]
                    p = stats_df['percent'][0]
                    row['Overall'] = f'{n} ({self._format_value(p, var, True)})'
                else:
                    row['Overall'] = f'0 (0.0)'

                if self.show_histograms:
                    row['Histogram'] = ''

            for s in stratified_stats:
                if level is None:
                    stats_df = s['vars'][var]['counts'].filter(pl.col(var).is_null())
                else:
                    stats_df = s['vars'][var]['counts'].filter(pl.col(var) == level)

                if len(stats_df) > 0:
                    n = stats_df['n'][0]
                    if self.row_percent and row_total > 0:
                        p = (n / row_total) * 100
                    else:
                        p = stats_df['percent'][0]
                    row[s['label']] = f'{n} ({self._format_value(p, var, True)})'
                else:
                    row[s['label']] = f'0 (0.0)'

                if self.show_histograms:
                    row[f"{s['label']} Hist"] = ''

            if self.pval and self.groupby:
                if i == 0 and var in p_values:
                    row['P-Value'] = self._format_pval(p_values[var]['p'])
                    if self.pval_adjust:
                        row['P-Value (adjusted)'] = self._format_pval(p_values[var].get('p_adj', np.nan))
                    if self.htest_name:
                        row['Test'] = p_values[var]['test']
                else:
                    row['P-Value'] = ''
                    if self.pval_adjust:
                        row['P-Value (adjusted)'] = ''
                    if self.htest_name:
                        row['Test'] = ''

            rows.append(row)
        return rows

    def _build_continuous_row(self, var, overall_stats, stratified_stats, p_values, *, hist_blocks=False):
        label = self.labels.get(var, var)
        is_nonnormal = var in self.nonnormal
        is_minmax = var in self.min_max

        if self.label_suffix:
            if is_minmax:
                label = f'{label}, min – max'
            elif is_nonnormal:
                label = f'{label}, median [Q1,Q3]'
            else:
                label = f'{label}, mean (SD)'

        row = {'': label, 'Level': ''}

        if self.missing:
            row['Missing'] = str(overall_stats['vars'][var]['missing'])

        def format_stat(stat_dict):
            if stat_dict['n'] == 0:
                return '—'

            if is_minmax:
                low = self._format_value(stat_dict['min'], var)
                high = self._format_value(stat_dict['max'], var)
                return f'{low} – {high}'

            if is_nonnormal:
                m = self._format_value(stat_dict['median'], var)
                low = self._format_value(stat_dict['q1'], var)
                high = self._format_value(stat_dict['q3'], var)
                return f'{m} [{low},{high}]'
            else:
                m = self._format_value(stat_dict['mean'], var)
                s = self._format_value(stat_dict['std'], var)
                return f'{m} ({s})'

        if self.overall:
            row['Overall'] = format_stat(overall_stats['vars'][var])
            if self.show_histograms:
                row['Histogram'] = overall_stats['vars'][var].get(
                    'hist_blocks' if hist_blocks else 'hist_counts', '')

        for s in stratified_stats:
            row[s['label']] = format_stat(s['vars'][var])
            if self.show_histograms:
                row[f"{s['label']} Hist"] = s['vars'][var].get('hist', '')

        if self.pval and self.groupby:
            if var in p_values:
                row['P-Value'] = self._format_pval(p_values[var]['p'])
                if self.pval_adjust:
                    row['P-Value (adjusted)'] = self._format_pval(p_values[var].get('p_adj', np.nan))
                if self.htest_name:
                    row['Test'] = p_values[var]['test']
            else:
                row['P-Value'] = ''
                if self.pval_adjust:
                    row['P-Value (adjusted)'] = ''
                if self.htest_name:
                    row['Test'] = ''

        return row

    def __str__(self):
        if self._table is not None:
            return str(self._table)
        return 'TableOne (not yet computed)'

    def to_gt(self):
        """
        Convert the summary table to a Great Tables object.
        """
        # create GT object from the internal table
        gt_table = GT(self._itable)

        # add basic formatting
        gt_table = gt_table.tab_header(
            title='Table 1',
            subtitle=f'Grouped by {self.groupby}' if self.groupby else None,
        )

        # align variable names to left and stats to center
        gt_table = gt_table.cols_align(
            align='left',
            columns=['', 'Level']
        )

        non_label_cols = [c for c in self._itable.columns if c not in ['', 'Level']]
        if non_label_cols:
            gt_table = gt_table.cols_align(
                align='center',
                columns=non_label_cols
            )

        # add footnotes
        for note in self._footnotes:
            gt_table = gt_table.tab_source_note(source_note=note)

        # format nanoplots for histograms
        if self.show_histograms:
            hist_cols = [c for c in self._itable.columns if 'Hist' in c]
            for col in hist_cols:
                # hide column label for hist columns
                gt_table = gt_table.cols_label(**{col: ''})
                if (self._itable[col].str.len_chars() > 0).any():
                    gt_table = gt_table.fmt_nanoplot(columns=col, plot_type='bar', autoscale=True)
                    # gt_table = gt_table.fmt_nanoplot(columns=col, plot_type='bar')  # this won't raise an error
                else:
                    # hide or remove columns?
                    pass

        return gt_table

    def show(self):
        return self.to_gt().show()

    def to_html(self, path=None, encoding='utf8'):
        html = self.to_gt().as_raw_html()
        if path:
            with open(path, 'w', encoding=encoding) as out:
                out.write(html)
        else:
            return html

    def to_latex(self, path = None, encoding='utf8'):
        try:
            text = self.to_gt().as_latex()
        except AttributeError:
            raise ValueError('LaTeX export not supported.')
        if path:
            with open(path, 'w', encoding=encoding) as out:
                out.write(text)
        else:
            return text

    def to_csv(self, path=None):
        if path:
            self._table.write_csv(path)
        return self._table.write_csv()

    def to_excel(self, path, **kwargs):
        self._table.write_excel(path, **kwargs)

    def _repr_html_(self):
        if self._itable is not None:
            return self.to_html()
        return 'TableOne (not yet computed)'

    def tabulate(self, tablefmt='grid', **kwargs):
        """
        Pretty-print tableone data. Wrapper for the Python 'tabulate' library.

        Args:
            headers (list): Defines a list of column headers to be used.
            tablefmt (str): Defines how the table is formatted. Table formats
                include: 'plain','simple','github','grid','fancy_grid','pipe',
                'orgtbl','jira','presto','psql','rst','mediawiki','moinmoin',
                'youtrack','html','latex','latex_raw','latex_booktabs',
                and 'textile'.

        Examples:
            To output tableone in github syntax, call tabulate with the
                'tablefmt="github"' argument.

            >>> print(tableone.tabulate(tablefmt='fancy_grid'))
        """

        return tabulate(self._table.rows(), headers=self._table.columns, tablefmt=tablefmt, **kwargs)

    write_excel = to_excel
    write_csv = to_csv
    write_latex = to_latex
    write_html = to_html

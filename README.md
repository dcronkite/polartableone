# `polartableone`

A polars-based implementation of the `tableone` library for generating summary statistics tables, often referred to as a 'Table 1'.

## Features

- Fast summary statistics for a 'Table 1' using polars
- Automatic detection of categorical and continuous variables
- Support for stratified tables and p-values
- Beautiful table formatting via Great Tables
- Export to HTML, CSV, Excel, LaTeX, and many tabular formats

### Differences with `tableone` Library

* `polars` rather than `pandas`
* Use of `great_tables` to display nicer tables.
* In some cases, use of `polars` operations rather than `numpy`, though these are almost always identical.
* Automatic determination of nonnormal columns and will display q1, median, q3 (whereas `tableone` defaults to showing `mean` even in these cases).
* Specify number of bins for histograms using `show_histograms=8`; if `show_histograms` is set to `True`, the default is 10 bins (`tableone` defaults to 8)


### But I Prefer `tableone`

Great! Just convert your `polars` dataframe to `pandas` (`df.to_pandas()`) and then use `tableone`!


## Installation

```bash
pip install polartableone
```

## Usage

```python
import polars as pl
from polartableone import TableOne

# load your data
df = pl.read_csv('tests/datasets/pn2012.csv')

# columns to summarize
columns = ['Age', 'SysABP', 'Height', 'Weight', 'ICU', 'death']
# columns containing categorical variables
categorical = ['ICU']
# columns containing categorical variables
continuous = ['Age', 'SysABP', 'Height', 'Weight']
# non-normal variables
nonnormal = ['Age']
# limit the binary variable "death" to a single row
limit = {"death": 1}
# set the order of the categorical variables
order = {"ICU": ["MICU", "SICU", "CSRU", "CCU"]}
# set decimal places for age to 0
decimals = {"Age": 0}
# optionally, a categorical variable for stratification
groupby = 'death'
# rename the death column
labels = {'death': 'Mortality'}

# display minimum and maximum for listed variables
min_max = ['Height']

table = TableOne(df, columns=columns, categorical=categorical, continuous=continuous,
                  groupby=groupby, nonnormal=nonnormal, rename=labels, label_suffix=True,
                  decimals=decimals, limit=limit, min_max=min_max, show_histograms=True,
                  dip_test=True, normal_test=True, tukey_test=True)


# Export to CSV
table.write_csv('table1.csv')

# Export to Great Tables object for further styling
gt_table = table.to_gt()
gt_table.show()
```

|                        | Level   | Missing   | Overall             | Histogram   | 0                   | 1                  |
|------------------------|---------|-----------|---------------------|-------------|---------------------|--------------------|
| n                      |         |           | 1000                |             | 864                 | 136                |
| Age, median [Q1,Q3]    |         | 0         | 68 [53,79]          | ▂▁▂▃▅▆▆▇▇█  | 66 [53,78]          | 75 [62,83]         |
| SysABP, median [Q1,Q3] |         | 291       | 119.0 [102.0,137.0] | ▂▁▁▁▄█▇▅▂▁  | 119.0 [104.0,137.0] | 115.0 [90.0,136.0] |
| Height, min – max      |         | 475       | 13.0 – 406.4        | ▁▁▆▅▅▆█▅▁▁  | 13.0 – 406.4        | 144.8 – 188.0      |
| Weight, median [Q1,Q3] |         | 302       | 80.0 [66.1,96.0]    | ▂▅▇█▆▃▃▂▁▁  | 80.5 [66.3,96.0]    | 77.0 [64.8,96.7]   |
| ICU, n (%)             | CCU     |           | 162 (16.2)          |             | 137 (15.9)          | 25 (18.4)          |
| ICU, n (%)             | CSRU    |           | 202 (20.2)          |             | 194 (22.5)          | 8 (5.9)            |
| ICU, n (%)             | MICU    |           | 380 (38.0)          |             | 318 (36.8)          | 62 (45.6)          |
| ICU, n (%)             | SICU    |           | 256 (25.6)          |             | 215 (24.9)          | 41 (30.1)          |

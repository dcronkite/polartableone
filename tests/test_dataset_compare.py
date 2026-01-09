import pytest
import polars as pl
from polartableone import TableOne
from pathlib import Path
from polars.testing import assert_frame_equal

TEST_DIR = Path(__file__).parent
DATASETS_DIR = TEST_DIR / 'datasets'
EXPECTED_DIR = TEST_DIR / 'expected'

params = [
    ({}, 'noarg'),
    ({'dip_test': True, 'normal_test': True, 'tukey_test': True, 'show_histograms': 8}, 'tests'),
    ({'columns': ['Age', 'SysABP', 'Height', 'Weight', 'ICU', 'death'],
      'categorical': ['ICU'],
      'continuous': ['Age', 'SysABP', 'Height', 'Weight'],
      'nonnormal': ['Age'],
      'limit': {"death": 1},
      'order': {"ICU": ["MICU", "SICU", "CSRU", "CCU"]},
      'decimals': {"Age": 0},
      'groupby': 'death',
      'labels': {'death': 'Mortality'},
      'min_max': ['Height'],
      }, 'groupby'),
]

datasets = ['iris', 'pbc', 'pn2012', 'rhc']
test_cases = [(x, ds, y) for x, y in params for ds in datasets]


@pytest.mark.parametrize('kwargs, input_name, version', test_cases)
def test_compare(kwargs, input_name, version):
    # test expected first to skip quickly if not related
    expected_path = EXPECTED_DIR / f'{input_name}-{version}.csv'
    if not expected_path.exists():
        pytest.skip(f'Missing file: {expected_path}')

    input_path = DATASETS_DIR / f'{input_name}.csv'
    if not input_path.exists():
        pytest.fail(f'Input file missing: {input_path}')

    input_df = pl.read_csv(input_path, infer_schema_length=10000, null_values='NA', encoding='latin1')
    table_instance = TableOne(data=input_df, **kwargs)
    actual_df = table_instance._table  # actual df

    expected_df = pl.read_csv(expected_path, infer_schema_length=0).fill_null('')

    assert actual_df.height == expected_df.height
    assert actual_df.width == expected_df.width

    expected_df.columns = actual_df.columns
    assert_frame_equal(actual_df, expected_df, check_dtype=False)


def test_pbc_stratified():
    pbc_df = pl.read_csv(DATASETS_DIR / 'pbc.csv', null_values='NA')

    table = TableOne(pbc_df, groupby='trt', pval=True)

    assert 'P-Value' in table._table.columns
    assert '1' in table._table.columns  # trt group 1
    assert '2' in table._table.columns  # trt group 2


def test_rhc_nonnormal():
    rhc_df = pl.read_csv(DATASETS_DIR / 'rhc.csv', null_values='NA')

    nonnormal = ['age', 'edu']
    # explicitly set continuous to ensure age is treated as such
    table = TableOne(rhc_df, continuous=['age', 'edu', 'das2d3pc'], nonnormal=nonnormal)

    # check if age is in columns
    found_age = False
    for val in table._table[''].to_list():
        if 'age' in val and 'median [Q1,Q3]' in val:
            found_age = True
            break
    assert found_age

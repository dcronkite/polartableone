import polars as pl
import numpy as np
from polartableone import TableOne


def test_min_max():
    df = pl.DataFrame({
        'name': ['Väinämöinen', 'Ilmarinen', 'Aino', 'Joukahainen', 'Lemminkäinen', 'Louhi'],
        'age': [1000, 900, 20, 25, 30, 800],
        'power': [10.5, 9.8, 1.2, 3.5, 5.0, 9.5],
        'side': ['Good', 'Good', 'Good', 'Bad', 'Bad', 'Bad']
    })

    # test min_max on continuous variable
    t = TableOne(df, continuous=['age'], min_max=['age'])
    output = t.to_csv()
    # age row should have min - max
    assert '20.0 – 1000.0' in output
    assert 'min – max' in output


def test_rename():
    df = pl.DataFrame({
        'v_age': [100, 200],
        'v_side': ['A', 'B']
    })
    t = TableOne(df, rename={'v_age': 'Age', 'v_side': 'Side'}, columns=['Age', 'Side'])
    assert 'Age' in t._table.columns[0] or 'Age' in t._table[''][1]
    # check if renamed columns exist in the output
    output = t.to_csv()
    assert 'Age' in output
    assert 'Side' in output


def test_pval_digits_and_threshold():
    df = pl.DataFrame({
        'val': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'group': ['A', 'A', 'A', 'B', 'B', 'B']
    })
    # specify as continuous to avoid categorical auto-detection
    t = TableOne(df, groupby='group', continuous=['val'], pval=True, pval_digits=4, pval_threshold=0.05)
    output = t.to_csv()
    # p-value for val should be significant now
    assert '*' in output


def test_limit():
    df = pl.DataFrame({
        'god': ['Väinämöinen'] * 10 + ['Ilmarinen'] * 5 + ['Aino'] * 2 + ['Joukahainen'] * 1,
        'group': [1] * 18
    })
    # auto-detection should work for strings
    t = TableOne(df, limit=2)
    output = t.to_csv()
    assert 'Väinämöinen' in output
    assert 'Ilmarinen' in output
    assert 'Other' in output
    assert 'Aino' not in output


def test_row_percent():
    df = pl.DataFrame({
        'gender': ['M', 'M', 'F', 'F', 'F'],
        'group': ['A', 'B', 'A', 'B', 'A']
    })
    # specify as categorical
    t = TableOne(df, groupby='group', categorical=['gender'], row_percent=True)
    # for M level:
    # group A: 1 (50.0%)
    # group B: 1 (50.0%)
    output = t.to_csv()
    assert '1 (50.0)' in output


def test_normal_test():
    # normal data
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, 100)
    # highly non-normal data
    nonnormal_data = np.random.exponential(1, 100)

    df = pl.DataFrame({
        'normal': normal_data,
        'skewed': nonnormal_data
    })

    # specify as continuous
    t = TableOne(df, continuous=['normal', 'skewed'], normal_test=True)
    # 'skewed' should be in nonnormal
    assert 'skewed' in t.nonnormal
    assert any('skewed' in note for note in t._footnotes)


def test_tukey_test():
    df = pl.DataFrame({
        'val': [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 100.0]  # 100 is an outlier
    })
    # specify as continuous
    t = TableOne(df, continuous=['val'], tukey_test=True)
    assert any('val' in note for note in t._footnotes)
    assert 'far outliers' in "".join(t._footnotes)


def test_htest_custom():
    df = pl.DataFrame({
        'val': [1, 2, 3, 4],
        'group': ['A', 'A', 'B', 'B']
    })

    def my_test(g1, g2):
        return 0.1234

    t = TableOne(df, groupby='group', pval=True, htest={'val': my_test})
    output = t.to_csv()
    assert '0.123' in output

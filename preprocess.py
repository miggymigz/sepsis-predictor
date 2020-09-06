from functools import reduce
from pathlib import Path
from tqdm import tqdm

from constants import FEATURE_MAP

import fire
import logging
import math
import numpy as np
import pandas as pd

logging.basicConfig(
    filename='preprocess.logs',
    filemode='a',
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG,
)
logger = logging.getLogger('default')


def transform_tobacco_values(df: pd.DataFrame):
    tobacco_ids = FEATURE_MAP['tobacco']
    tobacco_mapping = {
        'Current use or use within 1 month of admission': 1,
        'Stopped more than 1 month ago, but less than 1 year ago': 0.75,
        'Former user - stopped more than 1 year ago': 0.5,
        'Never used': 0,
        '1': 1,
        '0': 0,
        '1.0': 1,
        '0.0': 0,
    }

    mask = df['itemid'].isin(tobacco_ids)
    tobacco_values = pd.unique(df.loc[mask, 'value']).astype('str').tolist()
    assert isinstance(tobacco_values, list)

    if tobacco_values:
        for v in tobacco_values:
            assert v in tobacco_mapping.keys(), \
                f'[ERROR] Unknown tobacco value: {v}, type: {type(v)}'
        for k, v in tobacco_mapping.items():
            df.loc[mask & (df['value'] == k), 'valuenum'] = v

    return df


def filter_chartevents(
    input_csv: Path,
    output_csv: Path,
    chunksize: int = 10_000_000,
    total_rows: int = 327_363_274,
):
    logger.info('`filter_chartevents` has started')
    all_feature_ids = reduce(lambda x, y: x.union(y), FEATURE_MAP.values())
    columns = [
        'subject_id', 'hadm_id', 'stay_id',
        'itemid', 'charttime', 'valuenum', 'valueuom',
    ]
    iterator = pd.read_csv(
        input_csv, iterator=True,
        chunksize=chunksize, low_memory=False,
    )

    for i, df_chunk in tqdm(enumerate(iterator), total=math.ceil(total_rows/chunksize)):
        df_chunk.columns = map(str.lower, df_chunk.columns)
        df_chunk = transform_tobacco_values(df_chunk)

        mask = df_chunk['itemid'].isin(all_feature_ids)
        df_chunk = df_chunk[mask].dropna(axis=0, how='any', subset=columns)

        if i == 0:
            df_chunk.to_csv(output_csv, index=False, columns=columns)
        else:
            df_chunk.to_csv(output_csv, index=False,
                            columns=columns, header=None, mode='a')

    logger.info('`filter_chartevents` has ended')


def create_stay_blocks(input_csv, output_csv):
    logger.info('`create_stay_blocks` has started')

    # read input csv with column names in lowercase
    df = pd.read_csv(input_csv)
    df.columns = map(str.lower, df.columns)

    # create a mapping of feature id → feature name
    features_reversed = {v2: k for k, v1 in FEATURE_MAP.items() for v2 in v1}

    # rows in chartevents contain the date and time of an item
    # here, we extract only the date
    df['chartday'] = df['charttime'].astype(
        'str').str.split(' ').apply(lambda x: x[0])

    # combination of stay_id and chartday 
    # concatenated by an underscore
    df['stay_day'] = df['stay_id'].astype('str') + '_' + df['chartday']

    # fill in the feature name using itemid (which is the same as feature id)
    df['feature'] = df['itemid'].apply(lambda x: features_reversed[x])

    # create a dictionary that maps stay_day → subject_id
    icu_subject_mapping = dict(zip(df['stay_day'], df['subject_id']))

    # do necessary conversions
    mask = (df['itemid'] == 226707) | (df['itemid'] == 1394)
    df.loc[mask, 'valuenum'] *= 2.54

    # a specific value may have been measured multiple times in one day
    # here, we average those feature values in one day
    df_mean = pd.pivot_table(
        df,
        index='stay_day',
        columns='feature',
        values='valuenum',
        fill_value=np.nan,
        aggfunc=np.nanmean,
        dropna=False,
    )

    # get the standard deviation of the multiple measurements in one day
    df_std = pd.pivot_table(
        df,
        index='stay_day',
        columns='feature',
        values='valuenum',
        aggfunc=np.nanstd,
        fill_value=np.nan,
        dropna=False,
    )
    assert len(df_std.columns) == len(df_mean.columns)
    df_std.columns = [f'{i}_std' for i in df_mean.columns]

    # get the minimum value of the multiple measurements
    df_min = pd.pivot_table(
        df,
        index='stay_day',
        columns='feature',
        values='valuenum',
        aggfunc=np.nanmin,
        fill_value=np.nan,
        dropna=False,
    )
    assert len(df_min.columns) == len(df_mean.columns)
    df_min.columns = [f'{i}_min' for i in df_mean.columns]

    # get the maximum value of the multiple measurements
    df_max = pd.pivot_table(
        df,
        index='stay_day',
        columns='feature',
        values='valuenum',
        aggfunc=np.nanmax,
        fill_value=np.nan,
        dropna=False,
    )
    assert len(df_max.columns) == len(df_mean.columns)
    df_max.columns = [f'{i}_max' for i in df_mean.columns]

    # a feature values' min, max, and std can serve as another feature
    # thus becoming four features from being just one feature
    # so we concatenate the results here
    df = pd.concat([df_mean, df_std, df_min, df_max], axis=1)

    # below are the features who doesn't need to be aggregated
    # but since they have been already aggregated, we should delete them
    not_aggregatable = ['tobacco', 'daily weight', 'blood culture', 'diabetes']
    for feature in not_aggregatable:
        del df[f'{feature}_std']
        del df[f'{feature}_min']
        del df[f'{feature}_max']

    # combine INR and PT (and delete PT)
    df['INR'] += df['PT']
    df['INR_std'] += df['PT_std']
    df['INR_max'] += df['PT_max']
    df['INR_min'] += df['PT_min']
    del df['PT'], df['PT_std'], df['PT_max'], df['PT_min']

    # insert back information related to the patient 
    # (for persistence and for adding more info in the later steps)
    df['stay_day'] = df.index
    df['stay_id'] = df['stay_day'].str.split(
        '_').apply(lambda x: x[0]).astype('int')
    df['subject_id'] = df['stay_day'].apply(lambda x: icu_subject_mapping[x])

    # save result
    df.to_csv(output_csv, index=False)
    logger.info('`create_stay_blocks` has ended')


def preprocess_mimic4(
    dataset_dir: str = 'mimic-iv-0.4',
    output_dir: str = 'preprocess_outputs',
    redo: bool = False,
):
    # verify mimic4 dataset existence
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError('MIMIC-IV dataset does not exist')

    # create directory of output files
    output_path = Path(output_dir)
    output_path.mkdir(parents=False, exist_ok=True)

    # chartevents contain many measurements of
    # various values but we don't need all of them
    # so we filter out the ones that we need
    input_csv = dataset_path / 'icu' / 'chartevents.csv'
    output_csv = output_path / 'filtered_chartevents.csv'
    if redo or not output_csv.exists():
        filter_chartevents(input_csv, output_csv)

    # chartevents contain all of the feature values
    # but it is not grouped into unique ICU stays
    # so we group them here
    input_csv = output_csv
    output_csv = output_path / 'events_as_blocks.csv'
    if redo or not output_csv.exists():
        create_stay_blocks(input_csv, output_csv)


if __name__ == '__main__':
    fire.Fire(preprocess_mimic4)

# Script to download and recode the ACS PUMS data to match the format of the model-generated data

import numpy as np
import pandas as pd
from folktables import ACSDataSource


def PUMS_to_model_generated(dataset):
    """ Recode the PUMS data such that it has identical format as the model-generated datasets """

    var_of_interest = [
        'SEX', 'AGEP', 'HISP', 'RAC1P', 'NATIVITY', 'CIT',
        'SCH', 'SCHL', 'LANX', 'ENG',
        'HICOV', 'DEAR', 'DEYE',
        'MAR', 'FER', 'GCL',
        'MIL',
        'WRK', 'ESR', 'JWTRNS', 'WKL', 'WKWN', 'WKHP', 'COW', 'PINCP', 'ST', 'PWGTP']

    dataset = dataset[var_of_interest]

    # Recode relevant variables
    dataset['AGER'] = pd.cut(dataset['AGEP'], [-1, 4, 15, 30, 40, 50, 64, 99], labels=np.arange(7) + 1)
    dataset['HISPR'] = pd.cut(dataset['HISP'], [-1, 1, 24], labels=[2, 1])
    dataset['RAC1PR'] = pd.cut(dataset['RAC1P'], [-1, 1, 2, 5, 6, 8, 9], labels=np.arange(6) + 1)
    dataset['SCHLR'] = pd.cut(dataset['SCHL'], [-1, 1, 15, 17, 21, 24], labels=np.arange(5) + 1)
    dataset['WKHPR'] = pd.cut(dataset['WKHP'], [-1, 9, 19, 34, 44, 59, 98], labels=np.arange(6) + 1)
    dataset['COWR'] = dataset['COW']
    dataset['PINCPR'] = pd.cut(dataset['PINCP'], [-19999, 0, 12490, 52000, 120000, 4209995], labels=np.arange(5) + 1)
    dataset['WKWN'] = pd.cut(dataset['WKWN'], [-1, 13, 26, 39, 47, 52], labels=np.arange(5) + 1)

    dataset.drop(['AGEP', 'HISP', 'RAC1P', 'SCHL', 'WKHP', 'COW', 'PINCP'], axis=1, inplace=True)

    # Now process the nans
    below_5yo = dataset['AGER'] == 1
    dataset.loc[below_5yo, 'SCH'] = np.nan
    dataset.loc[below_5yo, 'SCHLR'] = np.nan
    dataset.loc[below_5yo, 'LANX'] = np.nan

    not_speak_other_language = dataset['LANX'] == 2
    dataset.loc[not_speak_other_language, 'ENG'] = np.nan

    below_15yo = dataset['AGER'] <= 2
    dataset.loc[below_15yo, 'MAR'] = np.nan
    dataset.loc[below_15yo, 'FER'] = np.nan

    above_15yo = dataset['AGER'] >= 3
    dataset.loc[above_15yo, 'MIL'].replace(to_replace=np.nan, value=4, inplace=True)

    return dataset


def download_and_recode_PUMS(save_name):
    print('Loading ACS...')
    data_source = ACSDataSource(survey_year='2019', horizon='1-Year', survey='person')
    data = data_source.get_data(download=True)

    print('Converting...')
    data = PUMS_to_model_generated(data)

    data.to_csv(save_name, index=False)

    print('Done')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str)
    args = parser.parse_args()

    download_and_recode_PUMS(args.save_dir + 'pums_2019.csv')

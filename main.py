import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn import preprocessing
from sklearn.linear_model import Ridge

DAYS_PER_YEAR = 112
NUMBERS = '0123456789'
SPECIALTIES = ["Technical", "Quick", "Head", "Powerful", "Unpredictable", "Resilient", "Support"]


def extract_data():
    with open('raw_data.html', mode='r') as f:
        data = f.read()
    soup = BeautifulSoup(data)
    head_row = soup.thead.tr
    ths = head_row.find_all('th')
    column_names = [th.string.strip() for th in ths]
    tbody = soup.tbody
    trs = tbody.find_all('tr')
    trs = [tr.find_all('td') for tr in trs]
    trs = [[td.get_text().strip() for td in tr] for tr in trs]
    df = pd.DataFrame(trs)
    df.columns = column_names
    return df


def process_age(value):
    split_value = value.split(' ')
    years = int(split_value[0])
    days = int(split_value[1][1:-1])
    age = years + days / (DAYS_PER_YEAR - 1)
    return age


def process_form(value):
    form = value.split(' ')[1]
    form = int(form[1:-1])
    return form


def process_tsi(value):
    value = [char for char in value if char in NUMBERS]
    value = ''.join(value)
    tsi = int(value)
    return tsi


def process_price(value):
    split_value = value.split(' ')

    if split_value[0] == '*':
        return None

    value = [char for char in value if char in NUMBERS]
    value = ''.join(value)
    price = int(value)
    return price


def populate_specialties(row):
    specialty = row['Specialty']
    if specialty != '':
        row[specialty] = 1

    return row


def preprocess_data(df):
    removed_column_names = ['ID', 'Date']
    kept_column_names = list(set(list(df.columns.values)).difference(removed_column_names))
    df = df[kept_column_names]

    df['Age'] = df['Age'].apply(process_age)
    df['Form'] = df['Form'].apply(process_form)
    df['TSI'] = df['TSI'].apply(process_tsi)
    df['Price'] = df['Price'].apply(process_price)

    for specialty in SPECIALTIES:
        df[specialty] = 0

    df = df.apply(populate_specialties, axis=1)

    df.drop('Specialty', axis=1, inplace=True)

    for index, row in df.iterrows():
        if pd.isnull(row['Price']):
            test = row.to_frame().transpose()
            test = test.drop('Price', axis=1)
            break

    df = df[np.isfinite(df['Price'])]

    return df, test


def split_x_y(column='Price'):
    y = data[column]
    x = data.drop(column, axis=1)

    return x, y


def scale(data, scaler=None):
    data = data.values

    if scaler is None:
        min_max_scaler = preprocessing.MinMaxScaler()
        scaled_data = min_max_scaler.fit_transform(data)
        scaler = min_max_scaler
    else:
        scaled_data = scaler.transform(data)

    return scaled_data, scaler


if __name__ == '__main__':
    data = extract_data()
    data, test = preprocess_data(data)

    x, y = split_x_y()

    x, x_scaler = scale(x)
    test, _ = scale(test, x_scaler)
    model = Ridge()
    model.fit(x, y)
    pred = model.predict(test)

    pass

import numpy as np
import pandas as pd


def get_data(request):
    newAPI = pd.DataFrame(data=request, index=[0])
    return newAPI


categories = ['pants', 'jeans', 'shirt', 't-shirt', 'jacket', 'coat', 'hoodies', 'sweatshirts', 'blazer', 'sneaker',
              'boot', 'oxford', 'blouseClean', 'skirtClean', 'tie']


def prepare_api(request):
    newAPI = get_data(request)
    selected_columns = ['category', 'subCategory', 'ProductId', 'ratingCount', 'ratingAvg']
    df_selected = newAPI[selected_columns]
    encoded_df = pd.DataFrame(columns=categories)
    return df_selected, encoded_df


def encode_api_data(request):
    df_selected, encoded_df = prepare_api(request)
    for index, row in df_selected.iterrows():
        sub_category = row['subCategory']
        if ((sub_category is not None) or (sub_category != np.nan)) and (sub_category in categories):
            encoded_row = pd.Series(index=categories, data=0)
            encoded_row[sub_category] = 1
        else:
            category = row['category']
            if ((category is not None) or (category != np.nan)) and (category in categories):
                encoded_row = pd.Series(index=categories, data=0)
                encoded_row[category] = 1
            else:
                print('This category doesn`t exist...')
                return None
        encoded_df = encoded_df._append(encoded_row, ignore_index=True)
        final_df = pd.concat([df_selected[['ProductId', 'ratingCount', 'ratingAvg']], encoded_df], axis=1)
        final_df = final_df.rename(index={0: int(df_selected['ProductId'].values)})

        return final_df


def add_product(request):
    final_df = encode_api_data(request)
    if final_df is not None:
        AI = pd.read_csv('database/product.csv')
        AI = pd.concat([final_df, AI], axis=0)
        AI.to_csv('database/product.csv', index=False)
        return 'Success'
    else:
        return 'Failed'

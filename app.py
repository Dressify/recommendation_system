from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from utility import *
import pypyodbc as odbc

import pandas as pd
import numpy as np
import json
import tensorflow as tf
from joblib import load
from sklearn.model_selection import train_test_split  # I just need scikit-learn to be installed

app = Flask(__name__)
api = Api(app)

product_path = './database/product.csv'
product = pd.read_csv(product_path)

model = tf.keras.models.load_model("recommend_sys")

scalerItem = load('./Scaling/item_scaler.bin')
scalerUser = load('./Scaling/user_scaler.bin')
scalerTarget = load('./Scaling/target_scaler.bin')


class Predict(Resource):
    def post(self):
        try:
            user = request.json
            user = list(user.values())
            print(user)

            user_vec = np.array(user)

            # getting sample of products
            sample = product.sample(800)
            print(sample)
            item_vecs = np.array(sample)
            # item_vecs = np.array(product)

            user_vecs = np.tile(user_vec, (len(item_vecs), 1))
            # scale our user and item vectors
            suser_vecs = scalerUser.transform(user_vecs[:, :])
            sitem_vecs = scalerItem.transform(item_vecs[:, 2:])
            y_p = model.predict([suser_vecs, sitem_vecs])
            y_pu = scalerTarget.inverse_transform(y_p)
            # yyy = y_pu * y_pu

            sorted_index = np.argsort(-y_pu, axis=0).reshape(-1).tolist()  # negate to get largest rating first
            sorted_ypu = y_pu[sorted_index]
            sorted_items = item_vecs[sorted_index]
            sorted_ypuDF = pd.DataFrame(sorted_ypu)
            sorted_itemsDF = pd.DataFrame(sorted_items)

            print(sorted_itemsDF.loc[:10, :])

            sorted_itemsDF.rename(
                columns={0: "ProductID", 1: "category", 2: "ratingCount",
                         3: "ratingAvg", 4: "pants", 5: "jeans", 6: "shirt", 7: "t-shirt", 8: "jacket", 9: "coat",
                         10: "hoodies", 11: "sweatshirts", 12: "blazer", 13: "sneaker", 14: "boot", 15: "oxford",
                         16: "blouseClean", 17: "skirtClean", 18: "tie"}, inplace=True)

            result2 = sorted_itemsDF.loc[:50, 'ProductID'].T.to_json()
            # result2 = sorted_itemsDF.loc[:50, 'ProductID'].values.tolist()
            print(result2)
            return result2
        except Exception as e:
            return {'error': str(e)}, 400


class AddProduct(Resource):
    def post(self):
        try:
            # print(get_data(request.json))
            result = add_product(request.json)
            # row = pd.DataFrame(data=[row])
            #
            # item_vecs = pd.read_csv(database_path)
            # item_vecs = pd.concat([item_vecs, row], ignore_index=True)
            #
            # item_vecs.to_csv('./database/database.csv', index=False, mode='w')
            #
            # test = pd.read_csv(database_path)
            # print(test.iloc[-5:, :])
            # return 'success', 200

            # data = request.json
            # print(request)
            # message = test()
            # data_preprocessed = convert_products_data_format(data)
            # conn = connect_to_database()
            # add_products_to_database(conn, data_preprocessed)
            return result
            # return True

        except Exception as e:
            return {'error': str(e)}, 400


class PredictAll(Resource):
    def post(self):
        try:
            user = request.json
            user = list(user.values())
            print(user)

            user_vec = np.array(user)

            # getting sample of products
            # sample = product.sample(800)
            # print(sample)
            # item_vecs = np.array(sample)
            item_vecs = np.array(product)

            user_vecs = np.tile(user_vec, (len(item_vecs), 1))
            # scale our user and item vectors
            suser_vecs = scalerUser.transform(user_vecs[:, :])
            sitem_vecs = scalerItem.transform(item_vecs[:, 2:])
            y_p = model.predict([suser_vecs, sitem_vecs])
            y_pu = scalerTarget.inverse_transform(y_p)
            # yyy = y_pu * y_pu

            sorted_index = np.argsort(-y_pu, axis=0).reshape(-1).tolist()  # negate to get largest rating first
            sorted_ypu = y_pu[sorted_index]
            sorted_items = item_vecs[sorted_index]
            sorted_ypuDF = pd.DataFrame(sorted_ypu)
            sorted_itemsDF = pd.DataFrame(sorted_items)

            print(sorted_itemsDF.loc[:10, :])

            sorted_itemsDF.rename(
                columns={0: "ProductID", 1: "category", 2: "ratingCount",
                         3: "ratingAvg", 4: "pants", 5: "jeans", 6: "shirt", 7: "t-shirt", 8: "jacket", 9: "coat",
                         10: "hoodies", 11: "sweatshirts", 12: "blazer", 13: "sneaker", 14: "boot", 15: "oxford",
                         16: "blouseClean", 17: "skirtClean", 18: "tie"}, inplace=True)

            result2 = sorted_itemsDF.loc[:50, 'ProductID'].T.to_json()
            # result2 = sorted_itemsDF.loc[:50, 'ProductID'].values.tolist()
            print(result2)
            return result2
        except Exception as e:
            return {'error': str(e)}, 400


class Test(Resource):
    def get(self):
        return 'tested Successfully! yes!!', 200


    # APIs EndPoints
api.add_resource(Predict, '/predict')
api.add_resource(PredictAll, '/predict_all')
api.add_resource(AddProduct, '/add_product')
api.add_resource(Test, '/test')

if __name__ == '__main__':
    app.config['ENV'] = 'development'
    app.config['DEBUG'] = True
    app.config['TESTING'] = True
    app.run(debug=True)
    # app.run()

# @app.route('/')
# def hello_world():  # put application's code here
#     return 'Hello World!'
#

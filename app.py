from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from utility import *

import pandas as pd
import numpy as np
import json
import tensorflow as tf
from joblib import load
from sklearn.model_selection import train_test_split  # I just need scikit-learn to be installed

app = Flask(__name__)
api = Api(app)

product_path = './database/product.csv'

model = tf.keras.models.load_model("recommend_sys")

scalerItem = load('./Scaling/item_scaler.bin')
scalerUser = load('./Scaling/user_scaler.bin')
scalerTarget = load('./Scaling/target_scaler.bin')


class Predict(Resource):
    def post(self):
        try:
            user = request.json
            user = list(user.values())

            user_vec = np.array(user)
            product = pd.read_csv(product_path)
            # getting sample of products
            sample = product.sample(800)
            item_vecs = np.array(sample)
            # item_vecs = np.array(product)

            user_vecs = np.tile(user_vec, (len(item_vecs), 1))
            # scale our user and item vectors
            suser_vecs = scalerUser.transform(user_vecs[:, :])
            sitem_vecs = scalerItem.transform(item_vecs[:, 1:])
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
                columns={0: "ProductID", 1: "ratingCount",
                         2: "ratingAvg", 3: "pants", 4: "jeans", 5: "shirt", 6: "t-shirt", 7: "jacket", 8: "coat",
                         9: "hoodies", 10: "sweatshirts", 11: "blazer", 12: "sneaker", 13: "boot", 14: "oxford",
                         15: "blouseClean", 16: "skirtClean", 17: "tie"}, inplace=True)

            # result2 = sorted_itemsDF.loc[:50, 'ProductID'].T.to_json()
            result2 = sorted_itemsDF.loc[:50, 'ProductID'].values
            result = list(map(int, result2))
            # result = json.dumps(result2)
            print(result)
            return result
        except Exception as e:
            return {'error': str(e)}, 400


class AddProduct(Resource):
    def post(self):
        try:
            result = add_product(request.json)
            if result == 'Failed':
                return {'error': 'This category doesn`t exist...'}, 400
            else:
                return result
        except Exception as e:
            return {'error': str(e)}, 400


class PredictAll(Resource):
    def post(self):
        try:
            user = request.json
            user = list(user.values())


            user_vec = np.array(user)

            product = pd.read_csv(product_path)
            item_vecs = np.array(product)

            user_vecs = np.tile(user_vec, (len(item_vecs), 1))
            # scale our user and item vectors
            suser_vecs = scalerUser.transform(user_vecs[:, :])
            sitem_vecs = scalerItem.transform(item_vecs[:, 1:])
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
                columns={0: "ProductID", 1: "ratingCount",
                         2: "ratingAvg", 3: "pants", 4: "jeans", 5: "shirt", 6: "t-shirt", 7: "jacket", 8: "coat",
                         9: "hoodies", 10: "sweatshirts", 11: "blazer", 12: "sneaker", 13: "boot", 14: "oxford",
                         15: "blouseClean", 16: "skirtClean", 17: "tie"}, inplace=True)

            result2 = sorted_itemsDF.loc[:50, 'ProductID'].T.to_json()
            # result2 = sorted_itemsDF.loc[:50, 'ProductID'].values.tolist()
            print(result2)
            return result2
        except Exception as e:
            return {'error': str(e)}, 400


class Test(Resource):
    def get(self):
        return 'tested Successfully! yes!', 200


    # APIs EndPoints
api.add_resource(Predict, '/predict')
api.add_resource(PredictAll, '/predictAll')
api.add_resource(AddProduct, '/add_product')
api.add_resource(Test, '/test')

if __name__ == '__main__':
    # app.config['ENV'] = 'development'
    # app.config['DEBUG'] = True
    # app.config['TESTING'] = True
    # app.run(debug=True)
    app.run()

# @app.route('/')
# def hello_world():  # put application's code here
#     return 'Hello World!'
#

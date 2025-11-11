import _setup_path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import importlib.util
import os


from utils.models.LinearRegresssion import LinearRegressionOneVariable


def main():


#from ... import ml_util.models.LinearRegresssion import LinearRegressionOneVariable


    DATASET="../dataset/housePrice.csv"
    COLS=["size", "price"]

    # Read housing dataset
    housingData = pd.read_csv(DATASET, usecols=COLS)

    # print(housingData.head(5))

    # Axis
    x = housingData["size"]
    y = housingData["price"]

    #Linear function parameter

    w = 200
    b = 100

    tmp_f_wb = LinearRegressionOneVariable(x, w, b)

    # Plot
    plt.title("Housing Price")
    plt.ylabel("Price")
    plt.xlabel("Size")

    # Plot the prediction line
    plt.plot(x, tmp_f_wb, c='b', label='prediction')

    # Plot the data
    plt.scatter(x,y, marker='x', c='r')


    # Plot one prediction point: 1200 size
    #tmp_f_wb = LinearRegressionOneVariable(1200, w, b)
    x_pred = 1200
    y_pred = w * x_pred + b

    # Point prediction
    plt.scatter(x_pred, y_pred, color="green", s = 100, label="Prediction for size=1200")
    plt.plot([x_pred, x_pred], [0, y_pred], color="green", linestyle="--")
    plt.plot([0, x_pred], [y_pred, y_pred], color="green", linestyle="--")

    # Disegno gli assi
    plt.xlim(0, x.max() + 200)
    plt.ylim(0, tmp_f_wb.max() + 200)
    #
    plt.axvline(0, color="green", linestyle="--", alpha=0.5)
    plt.axhline(0, color="green", linestyle="--", alpha=0.5)

    plt.text(x_pred + 200, y_pred - 120000, "Prediction:\nsize = 1200\nPrice = " + str(y_pred), fontsize = 6, bbox = dict(facecolor = 'white', alpha = 0.8))

    plt.show()


if __name__ == "main":
    main()
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt

#1
def get_stock_information(name, start_date, end_date):
    stock_data = yf.download(name, start=start_date, end=end_date)
    columns_to_use = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    data = stock_data[columns_to_use]

    scaler = MinMaxScaler(feature_range=(0.1, 1))
    data_scaled = scaler.fit_transform(data)

    scaler_y = MinMaxScaler(feature_range=(0.1, 1))
    y = scaler_y.fit_transform(data[['Adj Close']])

    X = [data_scaled[i][:5] for i in range(len(data_scaled))]
    X, y = np.array(X), np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    return X_train, X_test, y_train, y_test, scaler_y



params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

def stocks_model(X_train, X_test, y_train, y_test, params):
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    evals_result = {}
    num_rounds = 100
    bst = xgb.train(params, dtrain, num_rounds, evals=[(dtrain, 'train'), (dtest, 'eval')], evals_result=evals_result)

    # Extracting the RMSE values
    train_rmse = evals_result['train']['rmse']
    val_rmse = evals_result['eval']['rmse']
    
    return bst, train_rmse, val_rmse, dtrain, dtest

# Plotting the loss
def loss_plot_stock(bst, train_rmse, val_rmse, dtrain, dtest):
    plt.figure(figsize=(10, 6))
    plt.plot(train_rmse, label='Train RMSE')
    plt.plot(val_rmse, label='Validation RMSE')
    plt.xlabel('Number of Rounds')
    plt.ylabel('RMSE')
    plt.title('Training and Validation RMSE')
    plt.legend()
    plt.show()


# Make predictions
def stock_predictor(bst, train_rmse, val_rmse, dtrain, dtest, X_train, X_test, y_train, y_test, scaler_y):
    y_train_pred = bst.predict(dtrain)
    y_test_pred = bst.predict(dtest)

    # Invert scaling to get actual prices
    y_train_actual = scaler_y.inverse_transform(y_train.reshape(-1, 1))
    y_train_pred_actual = scaler_y.inverse_transform(y_train_pred.reshape(-1, 1))

    y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    y_test_pred_actual = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1))
    

ArithmeticError# Plotting
def plot_predictions(y_train_actual, y_train_pred_actual, y_test_actual, y_test_pred_actual):
    plt.figure(figsize=(14, 7))

    # Plot training set
    plt.plot(y_train_actual[len(y_train_actual)-100:], label='Actual (Train)', color='blue')
    plt.plot(y_train_pred_actual[len(y_train_pred_actual)-100:], label='Predicted (Train)', color='green')


    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Actual vs Predicted Prices (First 100 Variables)')
    plt.legend()
    plt.show()

    plt.figure(figsize=(14, 7))

    # Plot test set
    plt.plot(np.arange(len(y_train_actual)-100, len(y_train_actual)), y_test_actual[:100], label='Actual (Test)', color='red')
    plt.plot(np.arange(len(y_train_pred_actual)-100, len(y_train_pred_actual)), y_test_pred_actual[:100], label='Predicted (Test)', color='orange')


    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Actual vs Predicted Prices (First 100 Variables)')
    plt.legend()
    plt.show()

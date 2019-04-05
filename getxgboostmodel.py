
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

def get_xgboost_model(train_X, train_y):
    '''
    Generic xgboost fit using several grid searches

    :param train_X: numpy dataset
    :param train_y: numpy target
    :return: xgboost model
    '''
    model = Pipeline([('xgb', XGBRegressor())])

    # 1) Tune max depth
    param_grid = [{
        'xgb__n_estimators': [100],
        'xgb__learning_rate': [0.1],
        'xgb__max_depth': [1, 2, 4, 6, 8],
        'xgb__subsample': [1.00]
    }]
    gs1 = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=2)
    gs1 = gs1.fit(train_X, train_y)
    max_depth = gs1.best_params_['xgb__max_depth']
    # print(gs1.best_score_)
    # print(gs1.best_params_)

    # 2) Tune subsample
    param_grid = [{
        'xgb__n_estimators': [100],
        'xgb__learning_rate': [0.1],
        'xgb__max_depth': [max_depth],
        'xgb__subsample': [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    }]
    gs2 = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=2)
    gs2 = gs2.fit(train_X, train_y)
    subsample = gs2.best_params_['xgb__subsample']
    # print(gs2.best_score_)
    # print(gs2.best_params_)

    # 3) Tune n_estimators
    param_grid = [{
        'xgb__n_estimators': [50, 100, 150, 200],
        'xgb__learning_rate': [0.1],
        'xgb__max_depth': [max_depth],
        'xgb__subsample': [subsample]
    }]
    gs3 = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=2)
    gs3 = gs3.fit(train_X, train_y)
    n_estimators = gs3.best_params_['xgb__n_estimators']
    # print(gs3.best_score_)
    # print(gs3.best_params_)

    # 4) Tune learning rate
    param_grid = [{
        'xgb__n_estimators': [n_estimators],
        'xgb__learning_rate': [0.1],
        'xgb__max_depth': [max_depth],
        'xgb__subsample': [subsample]
    },
                  {
                      'xgb__n_estimators': [n_estimators * 3],
                      'xgb__learning_rate': [0.03],
                      'xgb__max_depth': [max_depth],
                      'xgb__subsample': [subsample]
                  }]
    gs4 = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=2)
    gs4 = gs4.fit(train_X, train_y)
    n_estimators = gs4.best_params_['xgb__n_estimators']
    learning_rate = gs4.best_params_['xgb__learning_rate']
    # print(gs4.best_score_)
    # print(gs4.best_params_)

    # 5) Tune n_estimators
    param_grid = [{
        'xgb__n_estimators': [
            int(0.8 * n_estimators),
            int(1.0 * n_estimators),
            int(1.2 * n_estimators)
        ],
        'xgb__learning_rate': [learning_rate],
        'xgb__max_depth': [max_depth],
        'xgb__subsample': [subsample]
    }]
    gs5 = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=2)
    gs5 = gs5.fit(train_X, train_y)
    n_estimators = gs5.best_params_['xgb__n_estimators']
    # print(gs5.best_score_)
    # print(gs5.best_params_)

    # 6) Tune sampling by tree
    param_grid = [{
        'xgb__n_estimators': [n_estimators],
        'xgb__learning_rate': [learning_rate],
        'xgb__max_depth': [max_depth],
        'xgb__subsample': [subsample],
        'xgb__colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'xgb__colsample_bylevel': [1.0]
    }]
    gs6 = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=2)
    gs6 = gs6.fit(train_X, train_y)
    colsample_bytree = gs6.best_params_['xgb__colsample_bytree']
    # print(gs6.best_score_)
    # print(gs6.best_params_)

    # 7) Tune subsample
    param_grid = [{
        'xgb__n_estimators': [n_estimators],
        'xgb__learning_rate': [learning_rate],
        'xgb__max_depth': [max_depth],
        'xgb__subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'xgb__colsample_bytree': [colsample_bytree],
        'xgb__colsample_bylevel': [1.0]
    }]
    gs7 = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=2)
    gs7 = gs7.fit(train_X, train_y)
    subsample = gs7.best_params_['xgb__subsample']
    # print(gs7.best_score_)
    # print(gs7.best_params_)

    # 8) Tune sampling by level
    n_estimators = gs7.best_params_['xgb__n_estimators']
    param_grid = [{
        'xgb__n_estimators': [n_estimators],
        'xgb__learning_rate': [learning_rate],
        'xgb__max_depth': [max_depth],
        'xgb__subsample': [subsample],
        'xgb__colsample_bytree': [colsample_bytree],
        'xgb__colsample_bylevel': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    }]
    gs8 = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=2)
    gs8 = gs8.fit(train_X, train_y)
    colsample_bylevel = gs8.best_params_['xgb__colsample_bylevel']
    # print(gs8.best_score_)
    # print(gs8.best_params_)

    # 9) Tune sampling fields
    n_estimators = gs8.best_params_['xgb__n_estimators']
    subsample = 0.9 if subsample == 1.0 else subsample
    colsample_bytree = 0.6 if colsample_bytree == 0.5 else colsample_bytree
    colsample_bylevel = 0.9 if colsample_bylevel == 1.0 else colsample_bylevel
    param_grid = [{
        'xgb__n_estimators': [n_estimators],
        'xgb__learning_rate': [learning_rate],
        'xgb__max_depth': [max_depth],
        'xgb__subsample': [subsample, subsample + 0.1],
        'xgb__colsample_bytree': [colsample_bytree - 0.1, colsample_bytree],
        'xgb__colsample_bylevel': [colsample_bylevel, colsample_bylevel + 0.1]
    }]
    gs9 = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=2)
    gs9 = gs9.fit(train_X, train_y)
    subsample = gs9.best_params_['xgb__subsample']
    colsample_bytree = gs9.best_params_['xgb__colsample_bytree']
    colsample_bylevel = gs9.best_params_['xgb__colsample_bylevel']
    # print(gs9.best_score_)
    # print(gs9.best_params_)

    # 10) Tune alpha
    param_grid = [{
        'xgb__n_estimators': [n_estimators],
        'xgb__learning_rate': [learning_rate],
        'xgb__max_depth': [max_depth],
        'xgb__subsample': [subsample],
        'xgb__colsample_bytree': [colsample_bytree],
        'xgb__colsample_bylevel': [colsample_bylevel],
        'xgb__reg_lambda': [0.001, 0.01, 0.1, 0.3, 1, 3, 10, 100, 1000]
    }]
    gs10 = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=2)
    gs10 = gs10.fit(train_X, train_y)
    # print(gs10.best_score_)
    # print(gs10.best_params_)

    # Find the best model
    # Sometimes the best model isn't the last one, so checking all of them
    best_model = gs1
    best_model_score = gs1.best_score_
    if gs2.best_score_ > best_model_score:
        best_model = gs2
        best_model_score = gs2.best_score_
    if gs2.best_score_ > best_model_score:
        best_model = gs2
        best_model_score = gs2.best_score_
    if gs3.best_score_ > best_model_score:
        best_model = gs3
        best_model_score = gs3.best_score_
    if gs4.best_score_ > best_model_score:
        best_model = gs4
        best_model_score = gs4.best_score_
    if gs5.best_score_ > best_model_score:
        best_model = gs5
        best_model_score = gs5.best_score_
    if gs6.best_score_ > best_model_score:
        best_model = gs6
        best_model_score = gs6.best_score_
    if gs7.best_score_ > best_model_score:
        best_model = gs7
        best_model_score = gs7.best_score_
    if gs8.best_score_ > best_model_score:
        best_model = gs8
        best_model_score = gs8.best_score_
    if gs9.best_score_ > best_model_score:
        best_model = gs9
        best_model_score = gs9.best_score_
    if gs10.best_score_ > best_model_score:
        best_model = gs10
        best_model_score = gs10.best_score_

    # Return the best model
    return XGBRegressor(**best_model.best_params_)


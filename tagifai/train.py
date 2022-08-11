from imblearn.over_sampling import RandomOverSampler
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
import mlflow
from torch import frac, threshold
import optuna
import data, predict, utils, evaluate

def train(args, df, trial=None):
    #setups
    utils.set_seeds()
    if args.shuffle: 
        df = df.sample(frac=1).reset_index(drop=True)
    df = df[:args.subset]
    df = data.preprocess(df, lower=args.lower, stem=args.stem)
    label_encoder = data.LabelEncoder().fit(df.tag)
    X_train, X_val, X_test, y_train, y_val, y_test = data.get_data_splits(
        X=df.text.to_numpy(), y=label_encoder.encode(df.tag)
    )
    test_df = pd.DataFrame({"text": X_test, "tag": label_encoder.decode(y_test)})

    #tf-idf
    vectorizer = TfidfVectorizer(
        analyzer=args.analyzer,
        ngram_range=(2, args.ngram_max_range)
    )
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)
    X_test = vectorizer.transform(X_test)

    #oversampling
    oversampling = RandomOverSampler(sampling_strategy="all")
    X_over, y_over = oversampling.fit_resample(X_train, y_train)

    #model
    model = SGDClassifier(
        loss="log", penalty="l2", alpha=args.alpha,
        max_iter=1, learning_rate="constant", eta0=args.learning_rate, 
        power_t=args.power_t, warm_start=True
    )

    #traning
    for epoch in range(args.num_epochs):
        model.fit(X_over, y_over)
        train_loss = log_loss(y_train, model.predict_proba(X_train))
        val_loss = log_loss(y_val, model.predict_proba(X_val))
        if not epoch%10:
            print(
                f"Epoch: {epoch:02d} | "
                f"Train Loss: {train_loss:.5f} | "
                f"Val Loss: {val_loss:.5f}"
            )
        
        #log
        if not trial:
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)


         # Pruning (for optimization in next section)
        if trial:  # pragma: no cover, optuna pruning
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()



    #threshold
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)
    args.threshold = np.quantile(
        [y_prob[i][j] for i, j in enumerate(y_pred)], q=0.25
    )

    #evaluate
    other_index = label_encoder.class_to_index["other"]
    y_prob = model.predict_proba(X_test)
    y_pred = predict.custom_predict(y_prob=y_prob, threshold=args.threshold, index=other_index)
    performance = evaluate.get_metrics(
        y_true=y_test, y_pred=y_pred,
        classes=label_encoder.classes, df=test_df
    )

    return {
        "args": args,
        "label_encoder": label_encoder,
        "vectorizer": vectorizer,
        "model": model,
        "performance": performance
    }

def objective(args, df, trial):
    """objective function for optimization trials"""
    args.analyzer = trial.suggest_categorical("analyzer", ["word", "char", "char_wb"])
    args.ngram_max_range = trial.suggest_int("ngram_max_range", 3, 10)
    args.learning_rate = trial.suggest_loguniform("learning_rate", 1e-2, 1e0)
    args.power_t = trial.suggest_uniform("power_t", 0.1, 0.5)

    #train and evaluate
    artifacts = train(args=args, df=df, trial=trial)

    #set additional attributes
    overall_performance = artifacts["performance"]["overall"]
    print(json.dumps(overall_performance, indent=2))
    trial.set_user_attr("precision", overall_performance["precision"])
    trial.set_user_attr("recall", overall_performance["recall"])
    trial.set_user_attr("f1", overall_performance["f1"])

    return overall_performance["f1"]


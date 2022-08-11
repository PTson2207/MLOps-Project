import numpy as np

def custom_predict(y_prob, threshold, index):
    y_pred = [np.argmax(p) if max(p) > threshold else index for p in y_prob]
    return np.array(y_pred)
    

def predict(texts, artifacts):
    """Predict tag for given text"""
    x = artifacts["vectorizer"].transform(texts)
    y_pred = custom_predict(
        y_prob=artifacts["model"].predict_proba(x),
        threshold=artifacts["args"].threshold,
        index=artifacts["label_encoder"].class_to_index["other"]
    )
    tags = artifacts["label_encoder"].decode(y_pred)
    predictions = [
        {
            "input_text": texts[i],
            "predict_tags": tags[i]
        }
        for i in range(len(tags))
    ]
    return predictions
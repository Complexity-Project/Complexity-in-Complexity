from linear_regression import line_regression
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut


RANDOM_SEED = 1


def _append(d, k, x):
    if k not in d:
        d[k] = [x]
    else:
        d[k].append(x)


def cross_validate(
    df,
    N=3,
    M=1,
    stratify=False,
    target="predicted_complexity",
    model_strs=None,
    return_preds=False,
):
    # model --> metrics (one array for each)
    model_results = {}

    for m in range(M):
        if stratify:
            # seed needs to increment to get different splits
            kf = StratifiedKFold(n_splits=N, shuffle=True, random_state=RANDOM_SEED + m)
            splits = list(kf.split(df, df["subcat"]))
        else:
            kf = KFold(n_splits=N, shuffle=True, random_state=RANDOM_SEED + m)
            splits = list(kf.split(df))

        for train_idxs, test_idxs in splits:
            df_train, df_test = df.iloc[train_idxs], df.iloc[test_idxs]

            # iterate over models
            def fit_mod(mod_strs, df_tr, df_te):
                for s in mod_strs:
                    if s not in model_results:
                        model_results[s] = {}
                    results = line_regression(target, s, df_tr, df_te, return_preds)

                    if return_preds:
                        # Initialize predictions dict if it doesn't exist
                        if "predictions" not in model_results[s]:
                            model_results[s]["predictions"] = {}

                        # Add predictions for each image ID
                        for img_id, pred in results["predictions"].items():
                            if img_id not in model_results[s]["predictions"]:
                                model_results[s]["predictions"][img_id] = []
                            model_results[s]["predictions"][img_id].append(pred)

                        # Remove predictions from results before adding other metrics
                        del results["predictions"]

                    for k, v in results.items():
                        _append(model_results[s], k, v)

            fit_mod(model_strs, df_train, df_test)
    return model_results

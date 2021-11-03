def compare_models(model1, model2, observed,  truth):
    models = (model1, model2)
    predicted = [model.predict(*observed) for model in models]
    correct = [p==truth for p in predicted]
    idxs = correct[0] & ~correct[1]
    for idx in idxs:
        print(observed[0][idx], observed[1][idx])
        print(model1.diagnostics(idx))
        print(model2.diagnostics(idx))

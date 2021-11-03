import numpy as np

class ModelBothAlleles:
    def predict(self, k1, k2):
        probs = [self.logpmf(k1, k2, g) for g in range(3)]
        return np.argmax(probs, axis=0)
    

class ComboModelBothAlleles(ModelBothAlleles):
    def __init__(self, model_ref, model_alt):
        self._model_ref = model_ref
        self._model_alt = model_alt
        self._n_variants = self._model_ref._n_variants

    def diagnostics(self, idx):
        return {name+key: value for name, model in [("ref", self._model_ref), ("alt", self._model_alt)] for key, value in model.diagnostics(idx).items()}

    def logpmf(self, k1, k2, genotype):
        ref_probs = self._model_ref.logpmf(k1, 2-genotype)
        alt_probs = self._model_alt.logpmf(k2, genotype)
        return ref_probs+alt_probs

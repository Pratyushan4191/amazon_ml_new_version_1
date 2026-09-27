class JaroWinkler:
    @staticmethod
    def normalized_similarity(a, b):
        # Minimal, deterministic, order-sensitive-ish stand-in. NOT the real
        # rapidfuzz Jaro-Winkler implementation -- only used to let dependent
        # code IMPORT and RUN for structural/memory/control-flow testing in
        # this sandbox (no network access to pip install the real package).
        # Any similarity SCORE produced using this stub is not meaningful
        # and must not be treated as a real feature-quality measurement.
        a, b = str(a), str(b)
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        matches = sum(1 for x, y in zip(a, b) if x == y)
        return matches / max(len(a), len(b))

    @staticmethod
    def similarity(a, b):
        return JaroWinkler.normalized_similarity(a, b)

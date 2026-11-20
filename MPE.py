"""Most probable explanation for small networks."""

from BNReasoner import BNReasoner


class MPE:
    def run(self, bn, evidence=None):
        return BNReasoner(bn).mpe(evidence)

"""Maximum a posteriori assignment for selected variables."""

from BNReasoner import BNReasoner


class MAP:
    def run(self, bn, query, evidence=None, order="min-fill"):
        return BNReasoner(bn).map(query, evidence, order)

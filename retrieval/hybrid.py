from config import Config


class HybridSearch:
    """
    Utility class for hybrid score fusion.
    """

    @staticmethod
    def normalize(scores: list[float]) -> list[float]:
        """
        Min-Max normalize scores to [0, 1].
        """

        if not scores:
            return []

        minimum = min(scores)
        maximum = max(scores)

        if minimum == maximum:
            return [1.0] * len(scores)

        return [
            (score - minimum) / (maximum - minimum)
            for score in scores
        ]

    @staticmethod
    def fuse(results: dict) -> list:
        """
        Calculate hybrid scores for merged retrieval results.
        """

        if not results:
            return []

        dense_scores = [
            item["dense_score"]
            for item in results.values()
        ]

        sparse_scores = [
            item["bm25_score"]
            for item in results.values()
        ]

        dense_normalized = HybridSearch.normalize(
            dense_scores
        )

        sparse_normalized = HybridSearch.normalize(
            sparse_scores
        )

        for index, item in enumerate(results.values()):

            item["hybrid_score"] = (

                Config.HYBRID_ALPHA
                * dense_normalized[index]

                +

                (1 - Config.HYBRID_ALPHA)
                * sparse_normalized[index]

            )

        return sorted(
            results.values(),
            key=lambda item: item["hybrid_score"],
            reverse=True
        )
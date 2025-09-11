from langchain_community.query_constructors.chroma import (
    ChromaTranslator as BaseChromaTranslator,
)
from langchain_core.structured_query import Comparator, Comparison

class CustomChromaTranslator(BaseChromaTranslator):
    def __init__(self):
        super().__init__()
        # `allowed_comparators` is a list; convert to set, then add `LIKE`, then back to list
        if self.allowed_comparators is None:
            self.allowed_comparators = []
        allowed_comparator_set = set(self.allowed_comparators)
        allowed_comparator_set.add(Comparator.LIKE)
        self.allowed_comparators = list(allowed_comparator_set)

    def visit_comparison(self, comparison: Comparison):
        """
        Chroma does NOT allow '$contains' or substring filtering out-of-the-box.
        We'll interpret `LIKE(attribute, value)` as an array-membership check:
            {"attribute": {"$in": [value]}}
        For this to work, your metadata must store `attribute` as a list of items.
        """
        if comparison.comparator == Comparator.LIKE:
            # Use $in to check if 'comparison.value' is in the attribute's list.
            return {comparison.attribute: {"$in": [comparison.value]}}
        # Otherwise, do default logic for eq, gt, gte, lt, lte, etc.
        return super().visit_comparison(comparison)
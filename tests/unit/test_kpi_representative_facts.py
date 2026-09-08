from __future__ import annotations

import unittest
from collections import Counter

from financial_vlm.evaluation.kpi_representative_facts import (
    RepresentativeFactRecord,
    stratified_cap_facts,
)


def fact(
    fact_id: str,
    document_id: str = "doc1",
    page_id: str = "p1",
    company_key: str = "co1",
    metric: str = "sales",
) -> RepresentativeFactRecord:
    return RepresentativeFactRecord(
        fact_id=fact_id,
        document_id=document_id,
        page_id=page_id,
        instance_id=0,
        company_key=company_key,
        metric=metric,
        bbox=(0, 0, 1, 1),
        period_text="2022",
        question="What was sales in 2022?",
        gold_answer="100",
        ocr_outcome="match",
        verification_status="automated_pass",
    )


class StratifiedCapFactsTest(unittest.TestCase):
    def test_caps_per_page(self) -> None:
        facts = [fact(f"f{i}", page_id="pageA", company_key=f"co{i}") for i in range(10)]
        result = stratified_cap_facts(facts, max_per_page=3, max_per_company=100)
        self.assertEqual(len(result), 3)

    def test_caps_per_company(self) -> None:
        facts = [fact(f"f{i}", page_id=f"page{i}", company_key="acme") for i in range(10)]
        result = stratified_cap_facts(facts, max_per_page=100, max_per_company=4)
        self.assertEqual(len(result), 4)

    def test_caps_per_metric_when_given(self) -> None:
        facts = [fact(f"f{i}", page_id=f"page{i}", company_key=f"co{i}", metric="sales") for i in range(10)]
        result = stratified_cap_facts(facts, max_per_page=100, max_per_company=100, max_per_metric=2)
        self.assertEqual(len(result), 2)

    def test_no_metric_cap_by_default(self) -> None:
        facts = [fact(f"f{i}", page_id=f"page{i}", company_key=f"co{i}") for i in range(10)]
        result = stratified_cap_facts(facts, max_per_page=100, max_per_company=100)
        self.assertEqual(len(result), 10)

    def test_deterministic_for_a_given_seed(self) -> None:
        facts = [fact(f"f{i}", page_id="pageA") for i in range(20)]
        first = stratified_cap_facts(facts, max_per_page=3, max_per_company=100, shuffle_seed=42)
        second = stratified_cap_facts(facts, max_per_page=3, max_per_company=100, shuffle_seed=42)
        self.assertEqual([f.fact_id for f in first], [f.fact_id for f in second])

    def test_no_single_dimension_exceeds_its_cap_under_combined_constraints(self) -> None:
        facts = []
        for doc_idx in range(3):
            for page_idx in range(5):
                for i in range(20):
                    facts.append(
                        fact(
                            f"d{doc_idx}p{page_idx}i{i}",
                            document_id=f"doc{doc_idx}",
                            page_id=f"page{page_idx}",
                            company_key=f"co{doc_idx}",
                            metric=["sales", "EBITDA"][i % 2],
                        )
                    )
        result = stratified_cap_facts(facts, max_per_page=3, max_per_company=10, max_per_metric=15)
        page_counts = Counter((f.document_id, f.page_id) for f in result)
        company_counts = Counter(f.company_key for f in result)
        metric_counts = Counter(f.metric for f in result)
        self.assertTrue(all(c <= 3 for c in page_counts.values()))
        self.assertTrue(all(c <= 10 for c in company_counts.values()))
        self.assertTrue(all(c <= 15 for c in metric_counts.values()))


if __name__ == "__main__":
    unittest.main()

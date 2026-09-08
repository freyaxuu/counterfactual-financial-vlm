# counterfactual-financial-vlm

Research code for counterfactual grounding experiments on vision-language
models for financial table / document fact extraction.

MSc thesis project for the UCL Data Science and Machine Learning MSc.

Pipeline under study:

```
document image + query
    -> candidate evidence regions or cells
    -> grounded evidence distribution
    -> structured financial fact
    -> evaluation and selective risk
```

## Layout

```
src/financial_vlm/       library
  data/                  dataset loaders, counterfactual transforms, re-rendering
  training/              training objectives (standard QA, CF augmentation, grounding, CF grounding)
  evaluation/            metrics (clean accuracy, HFR, CGS, natural-pair reliability, ...)
  integrations/          adapter boundary for the private dataset package

scripts/
  common/                dataset-agnostic, config-driven entry points
                         (train_*, generate_compute_matched_clean_b, evaluate_qwen3_vl_{canonical,cf,natural_pairs})
  synfintabs/            SynFinTabs synthetic results: controlled CGS/HFR benchmark,
                         grounding-supervision evals, out-of-distribution natural pairs
  tatqa/                 TAT-QA results: natural period-confusion pair benchmark,
                         freeze / build / aggregate / final-report scripts
  company/               private blind-validation set: build, inference runners, aggregation

configs/{synfintabs,tatqa,company}/   experiment configuration files
tests/unit/              unit tests with synthetic fixtures (no external datasets needed)
```

## Notes

Research code released as a reference. Paths in `configs/` and `scripts/`
are placeholders (`/path/to/...`) and must be set for your own environment.
The private evaluation dataset is not included and is not required to run
the unit tests.

```bash
pip install -e .
PYTHONPATH=src python -m pytest
```

Training / evaluation entry points additionally need the `train` extra
(`pip install -e '.[train]'`) and a local Qwen3-VL checkpoint.

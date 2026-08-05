# Online Shoppers dataset declaration

Status: review candidate for the unified benchmark.

## Source identity

The declaration targets the
[UCI Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online%20shoppers%20purchasing%20intention%20dataset),
dataset ID 468, DOI `10.24432/C5F88Q`. UCI publishes 12,330 rows with 17
features and identifies `Revenue` as the class label.

The `ucimlrepo` API returns features and target as separate DataFrames. The
benchmark declaration deliberately contains no download logic: its caller
must attach `Revenue` to the raw feature table before calling
`make_online_shoppers_dataset`.

The source has no persistent identifier column. Consequently, every one of
the 18 supplied columns is modeled and `TabularDataset.identifier` is `None`.

## Semantic column groups

The declaration follows column meaning rather than pandas storage dtype or
the labels in an old experiment script.

| Kind | Columns | Reason |
| --- | --- | --- |
| Continuous | `Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay` | Durations, rates, page value, and proximity are real-valued measurements. |
| Discrete | `Administrative`, `Informational`, `ProductRelated` | These are non-negative page-visit counts with meaningful numeric order. |
| Categorical | `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend`, `Revenue` | These are labels or source codes; numeric-looking codes do not express distance. |

No categorical column declares `ordered_values`. In particular, calendar
months are cyclic rather than a linearly ordered domain. The numeric discrete
page counts receive their ascending order from `ColumnKind.DISCRETE`.

`Revenue` is a categorical classification target and remains the final modeled
column in canonical generated-table order. It is not extracted from the
benchmark table as a separate shared `y`.

## Intentional correction to legacy evidence

The canonical loader in `sbtab/data/get_datasets.py` already fetches UCI 468
and passes `Revenue` as its classification target. Several legacy metric and
tuning scripts instead map Online Shoppers to `ProductRelated`. That mapping
contradicts both UCI and the repository loader, so the new declaration does not
preserve it.

This PR records the correction without modifying old entrypoints. Moving those
entrypoints onto the new benchmark remains a separate migration task.

## Ownership boundary

`sbtab.benchmark.datasets.online_shoppers` only attaches and validates static
semantics. It does not:

- fetch or cache UCI data;
- filter missing rows or split a dataset;
- fit encoders or scalers;
- choose a model, adapter, or metric;
- repair missing, extra, duplicated, or mistyped source columns.

Schema drift fails through the shared contract validator. Physical input
column order may vary; `ONLINE_SHOPPERS_COLUMNS` remains the canonical modeled
order.

## Verification

The committed tests use a small local frame with the real source names and
storage types, so CI does not depend on network availability:

```bash
python -m unittest tests.benchmark.test_online_shoppers
```

Before this review candidate was prepared, the declaration was also checked
against a fresh `ucimlrepo.fetch_ucirepo(id=468)` response: 12,330 feature rows,
12,330 target rows, the expected 17 feature names, and one `Revenue` target.

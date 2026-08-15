# Mixed benchmark datasets

Status: explicit migration baseline for the published fourteen-dataset set.

## Scope

The upstream repository publishes fourteen mixed datasets in
`sbtab/data/datasets/datasets_mixed.pkl` and constructs them in
`sbtab/data/get_datasets.py` in
[dataset update commit `33850fc`](https://github.com/Anaxagor/sb-tabular/commit/33850fc4805508fca993084634ef40e308dfa627).
The pickle stores raw frames plus feature groups inferred by the legacy
`TabularSchema` in `DataFrame.attrs`.

New benchmark code does not consume those attributes or infer semantics from
pandas dtype/cardinality. `sbtab/benchmark/datasets/mixed.py` materializes the
published groups as explicit `ColumnSpec` sequences. Every classification
target is an ordinary categorical modeled column; every regression target is
an ordinary continuous modeled column.

Online Shoppers is the intentional exception to a literal metadata migration:
it reuses the already approved pilot declaration. In particular, `Revenue` is
the categorical target, the three page-count columns are discrete, and numeric
nominal codes such as `TrafficType` are categorical.

## Collection

| Key | Published name | Source | Exact table | Target | Task | Removed before declaration |
| --- | --- | --- | --- | --- | --- | --- |
| `adult` | Adult | UCI 2 | features + target | `income` | classification | — |
| `credit_approval` | Credit Approval | UCI 27 | features + target | `A16` | classification | — |
| `online_shoppers` | Online Shoppers | UCI 468 | features + target | `Revenue` | classification | — |
| `eucalyptus` | Eucalyptus | OpenML 188 | OpenML frame | `Utility` | classification | — |
| `forest_fires` | Forest Fires | UCI 162 | features + target | `area` | regression | — |
| `insurance` | Insurance | Kaggle `mirichoi0218/insurance` | `insurance.csv` | `charges` | regression | — |
| `house_sales` | House Sales | Kaggle `harlfoxem/housesalesprediction` | `kc_house_data.csv` | `price` | regression | `id` |
| `cardiovascular_disease` | Cardiovascular Disease | Kaggle `sulianova/cardiovascular-disease-dataset` | `cardio_train.csv` | `cardio` | classification | `id` |
| `churn_modelling` | Churn Modelling | Kaggle `shrutimechlearn/churn-modelling` | `Churn_Modelling.csv` | `Exited` | classification | `RowNumber`, `CustomerId`, `Surname` |
| `auto_mpg` | Auto MPG | UCI 9 | features + target | `mpg` | regression | — |
| `diamonds` | Diamonds | Kaggle `shivam2503/diamonds` | `diamonds.csv` | `price` | regression | `Unnamed: 0` |
| `real_estate` | Real Estate | Kaggle `quantbruce/real-estate-price-prediction` | `Real estate.csv` | `Y house price of unit area` | regression | `No` |
| `stroke_prediction` | Stroke Prediction | Kaggle `fedesoriano/stroke-prediction-dataset` | `healthcare-dataset-stroke-data.csv` | `stroke` | classification | `id` |
| `palmer_penguins` | Palmer Penguins | Kaggle `parulpandey/palmer-archipelago-antarctica-penguin-data` | `penguins_lter.csv` | `Species` | classification | `studyName`, `Sample Number`, `Individual ID`, `Region`, `Stage`, `Comments` |

The earlier metrics table contains twelve of these datasets. `House Sales` and
`Diamonds` are the additional two entries in the current published collection.
Bike Sharing appeared briefly in an intermediate update and is not part of the
final fourteen.

## Acquisition boundary

`sbtab/benchmark/datasets/acquisition.py` imports source clients lazily and
returns validated `TabularDataset` objects. Importing the benchmark package
does not download data or require acquisition-only dependencies.

UCI loading requires `ucimlrepo`; Kaggle loading requires `kagglehub` and may
require Kaggle authentication or source-license consent:

```bash
conda activate lightning11
python -m pip install ucimlrepo kagglehub
```

Fetch one dataset:

```python
from sbtab.benchmark.datasets import fetch_mixed_dataset

dataset = fetch_mixed_dataset("adult")
```

Fetch all fourteen in canonical order:

```python
from sbtab.benchmark.datasets import fetch_all_mixed_datasets

datasets = fetch_all_mixed_datasets()
```

Kaggle handles without a version suffix resolve to the current published
version. Therefore a benchmark run must retain its create-only raw-data
artifact and SHA-256 digest; re-downloading later is not proof of identical
input bytes. The existing cross-validation artifact writer stores the
post-policy real table and its digest.

## Review boundary

This change makes the formerly implicit legacy feature grouping explicit and
reviewable; it does not claim that dtype/cardinality inference discovered the
best scientific ontology. Changes such as treating a postcode as nominal or a
binary indicator as categorical alter the generator representation and metric
grouping. Such corrections must be reviewed as a dataset-contract change and
must produce a separately labelled result rather than silently changing an
existing comparison.

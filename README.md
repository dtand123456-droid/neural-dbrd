# Neural DBRD

Neural retrieval and ranking code for duplicate bug reports. The model combines
report encoding, conditional report-unit alignment, and token matching within
report units. Shared modules support the main experiment, structural ablations,
query-body deletion, candidate-window comparisons, and inference measurement.

## Code layout

| Location | Responsibility |
| --- | --- |
| `common/model.py` | Report encoder, conditional unit alignment, token-profile readout, and model variants |
| `common/tokens.py` | Title/body tokenization, report-unit construction, and batch collation |
| `common/losses.py` | Group-ranking loss, Unit-Drop views, and report-view consistency loss |
| `common/pretraining.py` | Train-only report-view pretraining |
| `common/training.py` | Ranking optimization and Validation-based checkpoint selection |
| `common/sampling.py` | Positive selection and lexical, dense, and random negative sampling |
| `common/inputs.py` | Prepared report, query, public-backbone, and negative-pool readers |
| `common/ranking.py` | Historical candidate selection, bucket ranking, temporal features, and retrieval metrics |
| `common/inference.py` | Feature caching, coarse scoring, and token-based reranking |
| `common/evaluation.py` | Evaluation of completed, locked checkpoints |
| `common/missingness.py` | Deterministic query-body unit deletion |
| `common/supplement.py` | Window comparisons, query-deletion evaluation, and inference timing |
| `common/results.py` | Per-seed records and equal-project macro aggregation |
| `common/launcher.py`, `common/worker.py` | Experiment scheduling and isolated worker processes |
| `common/configuration.py`, `common/runtime.py` | Parameter validation, seed setup, and run integrity checks |
| `common/tensor_utils.py` | Tensor-state fingerprints |
| `common/seeds.json` | Fixed seed list read by the experiment launcher |

## Experiment entry points

| Entry point | Function |
| --- | --- |
| `E1/run.py` | Core ablations: S13, R7, GenericP, Base, and C3Only |
| `E2/run.py` | Internal matching and readout controls: S13, S12, S14, and S15 |
| `E3/run.py` | Partner-conditioned and self-conditioned weighting |
| `E4/run.py` | Main-model experiments across Spark, Hadoop, Kibana, Eclipse, Mozilla, and VSCode |
| `E4/summarize.py` | Seed and project-group aggregation of normalized metric records |
| `E5/run.py` | Query-body deletion evaluation |
| `E6/run.py` | Candidate-history window comparison |
| `E7/run.py` | Cache construction, memory use, and synchronized inference timing |

The entry points share the same model, training, and evaluation implementations.
The launcher creates an independent task for each configured model, project,
and seed. Evaluation groups results by candidate window and experimental condition.

## Model variants

S13 combines conditional unit alignment with a local token-profile readout. R7
removes the token-profile correction. GenericP replaces Unit-Drop views with
whole-report dropout views. SelfWeight uses the report itself as its weighting
condition. Base combines whole-report dropout, self-conditioning, and removal
of the token-profile correction; C3Only adds that correction to Base.

S12 moves the nonlinear readout after aggregation. S14 compresses token-match
distributions to a mean before kernel features are formed. S15 uses whole-report
token matching. These variants are implemented in the shared model module.

## Dependencies and interfaces

The code targets Python 3.10 and uses PyTorch, Transformers, and NumPy. Dependency
versions are listed in [requirements.txt](requirements.txt).

Experiment entry points accept `--assets`, `--config`, `--output`, and `--gpus`.
Evaluation-only entry points also accept `--trained`. The `--config` argument
selects a JSON document with `parameters` and `experiments` mappings;
[configuration.py](common/configuration.py) defines the required fields and
their validation. Entry-point `--help` displays the command interface.

Report readers use `id`, `ts`, `title`, `body`, `bucket`, and `phase` fields.
Prepared assets are organized into `fit/` and `test/` directories with file
manifests; `fit/mpnet/` supplies the public report encoder. The input reader
defines the query and negative-pool interfaces. Retrieval candidates precede
the query in timestamp and report-ID order. Rankings are aggregated by duplicate
bucket, and metric functions compute RR@K and full-ranking MRR.

## Pretrained encoder

The public encoder is
[`sentence-transformers/all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2),
revision `e8c3b32edf5434bc2275fc9bab85f82640a19130`.
The pretrained model retains its original license.

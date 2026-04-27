# NLP on ECB Speeches

This project studies European Central Bank monetary policy statements as text data and links communication patterns to financial market reactions. It replicates and extends the intuition of Amaya and Filbien (2015): central bank language can be measured, and changes in tone or repetition may help explain how markets react around policy events.

The full workflow is in [`nlp_ecb_v0.ipynb`](nlp_ecb_v0.ipynb). A clean rerun writes all reusable outputs to [`outputs/`](outputs/).

## Outputs

| Artifact | Path |
| --- | --- |
| Final analysis dataset | [`outputs/tables/analysis_dataset.csv`](outputs/tables/analysis_dataset.csv) |
| Documents by year | [`outputs/tables/documents_by_year.csv`](outputs/tables/documents_by_year.csv) |
| Similarity summary | [`outputs/tables/similarity_summary.csv`](outputs/tables/similarity_summary.csv) |
| Pessimism summary | [`outputs/tables/pessimism_summary.csv`](outputs/tables/pessimism_summary.csv) |
| Market reaction summary | [`outputs/tables/market_reaction_summary.csv`](outputs/tables/market_reaction_summary.csv) |
| Regression results | [`outputs/tables/regression_results.csv`](outputs/tables/regression_results.csv) |

## Methodology

| Stage | Method | Output |
| --- | --- | --- |
| Statement discovery | Query the ECB public AddSearch index for monetary policy statement pages | Candidate ECB statement links |
| Page extraction | Download each ECB HTML page with `requests` and parse it with BeautifulSoup | Date, title, link, statement text |
| Corpus filtering | Keep English statement pages with year-based URLs and remove known non-standard pages | Chronological statement dataset |
| Text cleaning | Lowercase, remove Q&A sections, strip punctuation/numbers, remove stopwords, stem words | `clean_text` |
| Similarity | Jaccard similarity on binary bigrams versus the previous statement | `similarity_sklearn` |
| Pessimism | Loughran-McDonald dictionary score, then ECB-specific kill-list cleaning | `pessimism_raw`, `pessimism_final` |
| Event study | Constant-mean model on Euro Stoxx 50 log returns, event window -5 to +5 | `CAR`, `ABS_CAR` |
| Regression | OLS with HC1 robust standard errors | Market-reaction model |

The executed sample contains 192 final ECB statements after filtering. The public ECB search index currently returns an uneven historical sample, especially before 2007, so the output tables should be treated as the reproducible sample for this run.

![ECB statements by year](outputs/figures/documents_by_year.png)

## Text Measures

Similarity is measured as the overlap in bigrams between each statement and the immediately previous statement:

```text
Jaccard similarity = shared bigrams / total unique bigrams
```

| statistic | value |
| --- | --- |
| count | 192.0 |
| mean | 0.3299 |
| std | 0.1373 |
| min | 0.0 |
| 25% | 0.2629 |
| 50% | 0.3349 |
| 75% | 0.4149 |
| max | 0.7333 |

![Similarity over time](outputs/figures/similarity_over_time.png)

Pessimism is computed as:

```text
pessimism = ((negative words - positive words) / total words) * 100
```

| measure | mean | std | min | median | max |
| --- | --- | --- | --- | --- | --- |
| Raw LM | 0.7598 | 1.4488 | -3.972 | 0.7003 | 4.2391 |
| ECB-cleaned LM | 0.0011 | 1.277 | -4.6729 | 0.0 | 3.3333 |

![Pessimism over time](outputs/figures/pessimism_over_time.png)

## Market Reaction

The event-study outcome is the absolute cumulative abnormal return around each statement date. Expected returns are estimated with a constant-mean model over trading days -250 to -50, and abnormal returns are summed over days -5 to +5.

| statistic | CAR | ABS_CAR_percent |
| --- | --- | --- |
| count | 160.0 | 160.0 |
| mean | -0.0015 | 3.38 |
| std | 0.0514 | 3.8713 |
| min | -0.3395 | 0.0066 |
| 25% | -0.021 | 1.2929 |
| median | 0.0022 | 2.4278 |
| 75% | 0.027 | 4.1915 |
| max | 0.1328 | 33.948 |

![Absolute CAR over time](outputs/figures/abs_car_over_time.png)

## Regression

The main regression tests whether markets react more strongly when statements are both pessimistic and similar to previous communication:

```text
ABS_CAR = beta0
        + beta1 * (pessimism_final * log(1 + similarity))
        + beta2 * DELTA_MRO
        + beta3 * INFLATION
        + beta4 * OUTPUT_GAP
        + error
```

| sample | interaction coefficient | p-value | nobs | R-squared |
| --- | ---: | ---: | ---: | ---: |
| Full sample | 0.4835 | 0.4855 | 159 | 0.1211 |
| Full sample, strict log | -0.2344 | 0.2276 | 159 | 0.1259 |
| Paper sample, 1999-2013 | 2.7595 | 0.0612 | 68 | 0.0877 |
| Paper sample, strict log | -0.5443 | 0.1881 | 68 | 0.0779 |

Full coefficient tables are saved in [`outputs/tables/regression_results.csv`](outputs/tables/regression_results.csv).

![Interaction coefficients](outputs/figures/interaction_coefficients.png)

## Main Findings

| Finding | Interpretation |
| --- | --- |
| ECB statements are textually persistent | The median statement shares roughly one third of its bigrams with the previous statement. |
| Raw dictionary tone needs context | Central-bank terms such as `risk`, `liquidity`, `objective`, and `easing` can distort generic financial sentiment scores. |
| Market reactions are skewed | Median absolute CAR is modest, but the maximum event reaction is much larger. |
| Full-sample communication effect is weak | The main interaction is positive but not statistically significant in the full rerun sample. |
| Earlier sample is more suggestive | The 1999-2013 interaction is positive and marginally significant, closer to the paper-style hypothesis. |

## Conclusion

The project shows that ECB communication can be transformed into transparent, reproducible text measures. Similarity captures continuity in policy language, while the cleaned Loughran-McDonald pessimism score gives a more ECB-aware tone measure than the raw dictionary.

The market-reaction evidence is mixed. In the full rerun sample, the pessimism-similarity interaction is not statistically significant. In the 1999-2013 sample, the interaction is positive and marginally significant, suggesting that the relationship between central bank wording and market reaction may be regime-dependent.

Overall, ECB language contains measurable information, but dictionary tone and textual repetition alone do not fully explain market reactions across the modern policy period.

## Extensions

| Extension | Why it matters |
| --- | --- |
| Save a fixed link manifest | The ECB search index changes over time; a manifest would lock the exact corpus for replication. |
| Use intraday returns | Daily event windows can include unrelated news; intraday data would isolate ECB communication more cleanly. |
| Add monetary policy surprise controls | Market reactions should depend on unexpected policy news, not only statement wording. |
| Compare dictionary sentiment with transformer sentiment | FinBERT or a central-bank-specific model may classify tone more accurately. |
| Add topic models | Similarity may reflect boilerplate or persistent economic topics; topic models can separate these channels. |
| Model regimes separately | Crisis, low-rate, pandemic, and inflation-surge periods likely have different communication effects. |

## Reproducibility

The notebook now creates the output folder automatically:

```text
outputs/
  figures/
  tables/
```

To regenerate the notebook and outputs:

```bash
/opt/anaconda3/bin/python -m nbconvert --execute --to notebook --inplace nlp_ecb_v0.ipynb --ExecutePreprocessor.timeout=1200
```

External data sources include the ECB website/search index, Yahoo Finance through `yfinance`, FRED through `pandas_datareader`, and the local Loughran-McDonald dictionary CSV.

#!/usr/bin/env python3
"""Reproduce and extend Amaya and Filbien (2015) with public data sources."""

from __future__ import annotations

import argparse
import os
import re
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urljoin, urlparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import requests
import statsmodels.api as sm
import yfinance as yf
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from tqdm import tqdm

ECB_BASE = "https://www.ecb.europa.eu"
ECB_INDEX = (
    ECB_BASE
    + "/press/press_conference/monetary-policy-statement/html/index.en.html"
)
ECB_MRO_CSV = (
    "https://data-api.ecb.europa.eu/service/data/"
    "FM/D.U2.EUR.4F.KR.MRR_RT.LEV?startPeriod=1999-01-01&format=csvdata"
)
EUREX_SX5E_CSV = (
    "https://raw.githubusercontent.com/vinomaster/kawb-eurex/"
    "master/basic/es.txt"
)
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; AcademicReplication/1.0)"}
PAPER_START = pd.Timestamp("1999-01-01")
PAPER_END = pd.Timestamp("2013-12-31")
REFERENCE_START = pd.Timestamp("1998-01-01")

# These pages live under the statement archive but are not part of the paper's
# monthly introductory-statement sequence.
INTRUDER_LINKS = {
    "/press/press_conference/monetary-policy-statement/2000/html/is000330.en.html",
    "/press/press_conference/monetary-policy-statement/2000/html/is001019.en.html",
    "/press/press_conference/monetary-policy-statement/2001/html/is011213.en.html",
    "/press/press_conference/monetary-policy-statement/2002/html/is020103_2.en.html",
    "/press/press_conference/monetary-policy-statement/2003/html/is030917.en.html",
    "/press/press_conference/monetary-policy-statement/2003/html/is031013.en.html",
    "/press/press_conference/monetary-policy-statement/2005/html/is050120.en.html",
    "/press/press_conference/monetary-policy-statement/2005/html/is050120_1.en.html",
    "/press/press_conference/monetary-policy-statement/2014/html/is141026.en.html",
    (
        "/press/press_conference/monetary-policy-statement/2021/html/"
        "ecb.sp210708~ab68c3bd9d.en.html"
    ),
}

EXPECTED_PAPER_DOCUMENTS = {
    1999: 11,
    2000: 11,
    2001: 11,
    2002: 11,
    2003: 11,
    2004: 11,
    2005: 11,
    2006: 12,
    2007: 11,
    2008: 12,
    2009: 12,
    2010: 12,
    2011: 12,
    2012: 12,
    2013: 12,
}

PAPER_BENCHMARKS = {
    "documents": 172.0,
    "similarity_mean": 0.24,
    "similarity_min": 0.05,
    "similarity_max": 0.55,
    "abs_car_mean": 3.28,
    "pessimism_mean": -0.26,
    "inflation_mean": 2.03,
    "output_gap_mean": 0.33,
    "delta_mro_changes": 33.0,
    "trend_log_time_coefficient": 0.493,
    "market_interaction_coefficient": -0.243,
}
PAPER_MATCH_TOLERANCES = {
    "documents": 0.0,
    "similarity_mean": 0.02,
    "similarity_min": 0.03,
    "similarity_max": 0.03,
    "abs_car_mean": 0.50,
    "inflation_mean": 0.10,
    "delta_mro_changes": 0.0,
    "trend_log_time_coefficient": 0.04,
    "market_interaction_coefficient": 0.05,
}
EXPECTED_PAPER_MRO_DISTRIBUTION = {
    -0.75: 1,
    -0.50: 7,
    -0.25: 10,
    0.00: 139,
    0.25: 13,
    0.50: 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lm-dictionary",
        type=Path,
        default=None,
        help="Path to a Loughran-McDonald MasterDictionary CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for generated tables and figures.",
    )
    return parser.parse_args()


def resolve_lm_dictionary(explicit_path: Path | None) -> Path:
    candidates = [
        explicit_path,
        Path(os.environ["LM_DICTIONARY"]) if os.environ.get("LM_DICTIONARY") else None,
        Path.home() / "Downloads" / "Loughran-McDonald_MasterDictionary_1993-2025.csv",
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Loughran-McDonald CSV not found. Pass --lm-dictionary or set LM_DICTIONARY."
    )


def get_soup(session: requests.Session, url: str) -> BeautifulSoup:
    response = session.get(url, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.content, "html.parser")


def discover_statement_links(session: requests.Session) -> list[str]:
    soup = get_soup(session, ECB_INDEX)
    container = soup.find(id="lazyload-container")
    if container is None or not container.get("data-snippets"):
        raise RuntimeError("ECB statement index does not expose lazy-load snippets.")

    links: set[str] = set()
    for snippet in container["data-snippets"].split(","):
        snippet_url = urljoin(ECB_INDEX, snippet.strip())
        response = session.get(snippet_url, timeout=30)
        if response.status_code == 404:
            continue
        response.raise_for_status()
        snippet_soup = BeautifulSoup(response.content, "html.parser")
        for anchor in snippet_soup.find_all("a", href=True):
            path = urlparse(urljoin(snippet_url, anchor["href"])).path
            if (
                "press_conference/monetary-policy-statement" in path
                and path.endswith(".en.html")
                and "index" not in path
            ):
                links.add(path)
    return sorted(links)


def is_standard_statement(title: str) -> bool:
    normalized = title.strip().lower()
    return (
        "introductory statement" in normalized
        or "monetary policy statement" in normalized
        or normalized == "press conference"
    )


QA_BOUNDARY_MARKERS = (
    "we are now at your disposal for questions",
    "we are now at your disposal",
    "we stand ready to answer any questions you may have",
    "we are now ready to take your questions",
)


def trim_at_qa_boundary(text: str) -> tuple[str, bool]:
    normalized = " ".join(text.lower().split())
    if normalized.startswith("question:") or normalized == "questions and answers":
        return "", True

    lower_text = text.lower()
    positions = [
        lower_text.find(marker)
        for marker in QA_BOUNDARY_MARKERS
        if marker in lower_text
    ]
    if positions:
        return text[: min(positions)].strip(), True
    return text, False


def extract_statement(session: requests.Session, path: str) -> dict[str, object]:
    soup = get_soup(session, ECB_BASE + path)
    h1 = soup.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else ""
    meta_date = soup.find("meta", {"property": "article:published_time"})
    time_tag = soup.find("time")
    raw_date = (
        meta_date["content"].strip()
        if meta_date and meta_date.get("content")
        else time_tag["datetime"].strip()
        if time_tag and time_tag.get("datetime")
        else None
    )

    main = soup.find("main")
    section = main.find("div", class_="section", recursive=False) if main else None
    body = section or main or soup
    content: list[str] = []
    for node in body.find_all(["h2", "p"]):
        text = node.get_text(" ", strip=True)
        if not text:
            continue
        text, hit_boundary = trim_at_qa_boundary(text)
        if text:
            content.append(text)
        if hit_boundary:
            break

    return {
        "link": path,
        "title": title,
        "date": pd.to_datetime(raw_date, errors="coerce"),
        "content": "\n".join(content),
    }


def build_corpus(session: requests.Session) -> pd.DataFrame:
    links = discover_statement_links(session)
    rows = [
        extract_statement(session, path)
        for path in tqdm(links, desc="ECB statement pages")
    ]
    df = pd.DataFrame(rows).dropna(subset=["date"])
    df["year"] = df["date"].dt.year
    df = df[
        (df["date"] >= REFERENCE_START)
        & (~df["link"].isin(INTRUDER_LINKS))
        & (df["title"].map(is_standard_statement))
    ].copy()
    df = df.sort_values("date").drop_duplicates("link").reset_index(drop=True)

    paper_counts = (
        df[df["date"].between(PAPER_START, PAPER_END)]
        .groupby("year")
        .size()
        .to_dict()
    )
    if paper_counts != EXPECTED_PAPER_DOCUMENTS:
        raise AssertionError(
            f"Paper corpus changed: expected {EXPECTED_PAPER_DOCUMENTS}, got {paper_counts}"
        )
    return df


def load_stop_words() -> set[str]:
    try:
        return set(nltk.corpus.stopwords.words("english"))
    except LookupError:
        return set(ENGLISH_STOP_WORDS)


def clean_text(text: str, stop_words: set[str], porter: PorterStemmer) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z'\s-]", " ", text)
    tokens = [token for token in text.split() if token not in stop_words and len(token) > 1]
    return " ".join(porter.stem(token) for token in tokens)


def attach_text_measures(df: pd.DataFrame, lm_path: Path) -> pd.DataFrame:
    result = df.copy()
    porter = PorterStemmer()
    stop_words = load_stop_words()
    result["clean_text"] = result["content"].map(
        lambda text: clean_text(text, stop_words, porter)
    )

    vectorizer = CountVectorizer(ngram_range=(2, 2), binary=True)
    matrix = vectorizer.fit_transform(result["clean_text"].fillna(""))
    similarities = [np.nan]
    for index in range(1, matrix.shape[0]):
        previous = matrix[index - 1]
        current = matrix[index]
        intersection = previous.multiply(current).sum()
        union = previous.sum() + current.sum() - intersection
        similarities.append(float(intersection / union) if union else np.nan)
    result["similarity"] = similarities

    lm = pd.read_csv(lm_path)
    negative = set(
        porter.stem(word)
        for word in lm.loc[lm["Negative"] > 0, "Word"].astype(str).str.lower()
    )
    positive = set(
        porter.stem(word)
        for word in lm.loc[lm["Positive"] > 0, "Word"].astype(str).str.lower()
    )

    kill_list_negative = {
        "question",
        "questions",
        "vice",
        "general",
        "press",
        "lie",
        "risk",
        "risks",
        "deficit",
        "deficits",
        "liquid",
        "liquidity",
        "object",
        "objective",
        "dispos",
        "disposable",
        "lag",
        "lagged",
        "eas",
        "easing",
        "implic",
        "implication",
        "restructur",
        "restructuring",
        "prevent",
        "decline",
        "declined",
        "declining",
        "dampen",
        "dampened",
        "limit",
        "limited",
        "crucial",
        "persistent",
        "challeng",
        "challenges",
        "urgent",
    }
    kill_list_positive = {
        "pleas",
        "pleased",
        "pleasure",
        "happy",
        "kind",
        "welcome",
        "good",
        "inform",
        "information",
        "opportun",
        "opportunity",
        "conclus",
        "conclusion",
        "share",
        "lead",
        "leading",
    }
    negative_clean = negative - {porter.stem(word) for word in kill_list_negative}
    positive_clean = positive - {porter.stem(word) for word in kill_list_positive}

    def pessimism(text: str, negative_words: set[str], positive_words: set[str]) -> float:
        tokens = text.split()
        if not tokens:
            return np.nan
        negative_count = sum(token in negative_words for token in tokens)
        positive_count = sum(token in positive_words for token in tokens)
        return ((negative_count - positive_count) / len(tokens)) * 100

    result["pessimism_raw"] = result["clean_text"].map(
        lambda text: pessimism(text, negative, positive)
    )
    result["pessimism_clean"] = result["clean_text"].map(
        lambda text: pessimism(text, negative_clean, positive_clean)
    )
    return result


def load_market_prices() -> pd.DataFrame:
    historical = pd.read_csv(
        EUREX_SX5E_CSV,
        sep=";",
        parse_dates=["Date"],
        dayfirst=True,
        index_col="Date",
    )
    historical["Close"] = pd.to_numeric(historical["SX5E"], errors="coerce")
    historical = historical[["Close"]].dropna()

    end_date = (date.today() + timedelta(days=1)).isoformat()
    recent = yf.download(
        "^STOXX50E",
        start="2007-01-01",
        end=end_date,
        auto_adjust=False,
        progress=False,
    )
    if isinstance(recent.columns, pd.MultiIndex):
        recent = recent.xs("^STOXX50E", axis=1, level=1)
    price_column = "Adj Close" if "Adj Close" in recent.columns else "Close"
    recent = recent[[price_column]].rename(columns={price_column: "Close"})
    recent.index = pd.to_datetime(recent.index).tz_localize(None)

    overlap = historical.join(recent, lsuffix="_historical", rsuffix="_recent").dropna()
    overlap_correlation = overlap["Close_historical"].pct_change().corr(
        overlap["Close_recent"].pct_change()
    )
    if overlap_correlation < 0.99:
        raise AssertionError(f"Euro Stoxx overlap correlation too low: {overlap_correlation}")

    prices = pd.concat([historical[historical.index < recent.index.min()], recent])
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    return prices


def compute_car(
    event_date: pd.Timestamp,
    returns: pd.Series,
    estimation_days: int = 250,
    gap_days: int = 50,
    event_days: int = 5,
) -> float:
    position = returns.index.get_indexer([pd.to_datetime(event_date)], method="nearest")[0]
    if position < 0 or abs((returns.index[position] - event_date).days) > 3:
        return np.nan
    estimation_start = position - estimation_days
    estimation_end = position - gap_days
    event_start = position - event_days
    event_end = position + event_days
    if estimation_start < 0 or event_end >= len(returns):
        return np.nan

    # The paper defines an inclusive [-250, -50] estimation window: 201 returns.
    estimation_returns = returns.iloc[estimation_start : estimation_end + 1]
    if len(estimation_returns) != 201:
        raise AssertionError("Unexpected estimation-window length.")
    event_returns = returns.iloc[event_start : event_end + 1]
    return float((event_returns - estimation_returns.mean()).sum())


def attach_market_reaction(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    prices = load_market_prices()
    returns = np.log(prices["Close"] / prices["Close"].shift(1)).dropna()
    result["CAR"] = result["date"].map(lambda event_date: compute_car(event_date, returns))
    result["ABS_CAR"] = result["CAR"].abs() * 100
    return result


def attach_macro_controls(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    end_date = date.today().isoformat()

    mro = pd.read_csv(ECB_MRO_CSV, usecols=["TIME_PERIOD", "OBS_VALUE"])
    mro["TIME_PERIOD"] = pd.to_datetime(mro["TIME_PERIOD"])
    mro = mro.set_index("TIME_PERIOD")["OBS_VALUE"].astype(float).rename("MRO")
    mro_changes = mro.diff().dropna()
    mro_changes = mro_changes[mro_changes != 0]

    inflation = web.DataReader(
        "CP0000EZ19M086NEST", "fred", "1997-01-01", end_date
    ).rename(columns={"CP0000EZ19M086NEST": "HICP"})
    inflation["INFLATION"] = inflation["HICP"].pct_change(12) * 100

    gdp = web.DataReader(
        "CLVMNACSCAB1GQEA19", "fred", "1997-01-01", end_date
    ).rename(columns={"CLVMNACSCAB1GQEA19": "REAL_GDP"})
    cycle, _ = sm.tsa.filters.hpfilter(np.log(gdp["REAL_GDP"].dropna()), lamb=1600)
    output_gap = (cycle * 100).rename("OUTPUT_GAP").resample("MS").first().interpolate()

    macro = pd.DataFrame({"INFLATION": inflation["INFLATION"], "OUTPUT_GAP": output_gap})
    macro = macro.resample("D").ffill().join(mro, how="outer").sort_index().ffill()
    result = pd.merge_asof(
        result.sort_values("date"),
        macro[["MRO", "INFLATION", "OUTPUT_GAP"]],
        left_on="date",
        right_index=True,
        direction="backward",
    )

    def announced_delta(event_date: pd.Timestamp) -> float:
        effective_changes = mro_changes[
            (mro_changes.index > event_date)
            & (mro_changes.index <= event_date + pd.Timedelta(days=7))
        ]
        return float(effective_changes.sum())

    result["DELTA_MRO"] = result["date"].map(announced_delta)
    return result


def add_transformations(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["log_similarity"] = np.log(result["similarity"].replace(0, np.nan))
    result["interaction_raw"] = result["pessimism_raw"] * result["log_similarity"]
    result["interaction_clean"] = result["pessimism_clean"] * result["log_similarity"]
    return result


def validate_analysis(df: pd.DataFrame) -> None:
    if df["date"].duplicated().any() or df["link"].duplicated().any():
        raise AssertionError("Analysis corpus contains duplicate dates or links.")

    required = [
        "similarity",
        "pessimism_raw",
        "pessimism_clean",
        "CAR",
        "ABS_CAR",
        "MRO",
        "DELTA_MRO",
        "INFLATION",
        "OUTPUT_GAP",
        "log_similarity",
        "interaction_raw",
        "interaction_clean",
    ]
    missing = df[required].isna().sum()
    if missing.any():
        raise AssertionError(f"Analysis dataset has missing values:\n{missing[missing > 0]}")

    paper = df[df["date"].between(PAPER_START, PAPER_END)]
    observed_distribution = (
        paper["DELTA_MRO"].round(2).value_counts().sort_index().to_dict()
    )
    if observed_distribution != EXPECTED_PAPER_MRO_DISTRIBUTION:
        raise AssertionError(
            "Paper-period Delta MRO distribution changed: "
            f"expected {EXPECTED_PAPER_MRO_DISTRIBUTION}, got {observed_distribution}"
        )


def ols_result(
    df: pd.DataFrame, label: str, dependent: str, regressors: list[str]
) -> pd.DataFrame:
    regression_df = df[[dependent] + regressors].dropna()
    result = sm.OLS(
        regression_df[dependent],
        sm.add_constant(regression_df[regressors]),
    ).fit(cov_type="HC1")
    return pd.DataFrame(
        {
            "sample": label,
            "term": result.params.index,
            "coefficient": result.params.values,
            "std_error": result.bse.values,
            "p_value": result.pvalues.values,
            "nobs": int(result.nobs),
            "r_squared": result.rsquared,
            "adjusted_r_squared": result.rsquared_adj,
        }
    )


def market_regressions(df: pd.DataFrame) -> pd.DataFrame:
    paper = df[df["date"].between(PAPER_START, PAPER_END)].copy()
    extension = df[df["date"] > PAPER_END].copy()
    controls = ["OUTPUT_GAP", "INFLATION", "DELTA_MRO"]
    return pd.concat(
        [
            ols_result(paper, "Paper: pessimism only", "ABS_CAR", ["pessimism_raw"]),
            ols_result(paper, "Paper: controls only", "ABS_CAR", controls),
            ols_result(paper, "Paper: interaction only", "ABS_CAR", ["interaction_raw"]),
            ols_result(
                paper,
                "Paper: interaction + controls",
                "ABS_CAR",
                ["interaction_raw"] + controls,
            ),
            ols_result(
                extension,
                "Extension 2014+: interaction + controls",
                "ABS_CAR",
                ["interaction_raw"] + controls,
            ),
            ols_result(
                df,
                "Full 1999+: interaction + controls",
                "ABS_CAR",
                ["interaction_raw"] + controls,
            ),
            ols_result(
                extension,
                "Extension 2014+: cleaned LM sensitivity",
                "ABS_CAR",
                ["interaction_clean"] + controls,
            ),
            ols_result(
                df,
                "Full 1999+: cleaned LM sensitivity",
                "ABS_CAR",
                ["interaction_clean"] + controls,
            ),
        ],
        ignore_index=True,
    )


def similarity_regressions(df: pd.DataFrame) -> pd.DataFrame:
    paper = df[df["date"].between(PAPER_START, PAPER_END)].copy()
    paper["log_time"] = np.log((paper["date"] - PAPER_START).dt.days)
    paper["log_count"] = np.log(np.arange(1, len(paper) + 1))
    controls = ["OUTPUT_GAP", "INFLATION", "DELTA_MRO"]
    return pd.concat(
        [
            ols_result(paper, "Paper: controls only", "log_similarity", controls),
            ols_result(paper, "Paper: log time only", "log_similarity", ["log_time"]),
            ols_result(
                paper,
                "Paper: log time + controls",
                "log_similarity",
                ["log_time"] + controls,
            ),
            ols_result(
                paper,
                "Paper: log count + controls",
                "log_similarity",
                ["log_count"] + controls,
            ),
        ],
        ignore_index=True,
    )


def describe_series(series: pd.Series) -> pd.Series:
    return series.describe(percentiles=[0.25, 0.5, 0.75])


def build_paper_comparison(
    df: pd.DataFrame,
    market_results: pd.DataFrame,
    similarity_results: pd.DataFrame,
) -> pd.DataFrame:
    paper = df[df["date"].between(PAPER_START, PAPER_END)]
    interaction_row = market_results[
        (market_results["sample"] == "Paper: interaction + controls")
        & (market_results["term"] == "interaction_raw")
    ].iloc[0]
    trend_row = similarity_results[
        (similarity_results["sample"] == "Paper: log time + controls")
        & (similarity_results["term"] == "log_time")
    ].iloc[0]
    reproduced = {
        "documents": float(len(paper)),
        "similarity_mean": paper["similarity"].mean(),
        "similarity_min": paper["similarity"].min(),
        "similarity_max": paper["similarity"].max(),
        "abs_car_mean": paper["ABS_CAR"].mean(),
        "pessimism_mean": paper["pessimism_raw"].mean(),
        "inflation_mean": paper["INFLATION"].mean(),
        "output_gap_mean": paper["OUTPUT_GAP"].mean(),
        "delta_mro_changes": float((paper["DELTA_MRO"] != 0).sum()),
        "trend_log_time_coefficient": trend_row["coefficient"],
        "market_interaction_coefficient": interaction_row["coefficient"],
    }
    notes = {
        "documents": "Exact corpus count",
        "similarity_mean": "Close paper match",
        "similarity_min": "Archive/parser-sensitive; mean and maximum remain close",
        "similarity_max": "Close paper match",
        "abs_car_mean": "Public SX5E proxy replaces licensed Datastream series",
        "pessimism_mean": "Current LM dictionary differs from the paper-era list",
        "inflation_mean": "Close paper match",
        "output_gap_mean": "HP-filter proxy differs from European Commission series",
        "delta_mro_changes": "Exact announcement distribution",
        "trend_log_time_coefficient": "Close directional match with public controls",
        "market_interaction_coefficient": "Close paper match",
    }
    return pd.DataFrame(
        {
            "metric": list(PAPER_BENCHMARKS),
            "paper": [PAPER_BENCHMARKS[key] for key in PAPER_BENCHMARKS],
            "reproduction": [reproduced[key] for key in PAPER_BENCHMARKS],
            "note": [notes[key] for key in PAPER_BENCHMARKS],
        }
    )


def validate_paper_comparison(comparison: pd.DataFrame) -> None:
    indexed = comparison.set_index("metric")
    for metric, tolerance in PAPER_MATCH_TOLERANCES.items():
        difference = abs(indexed.at[metric, "paper"] - indexed.at[metric, "reproduction"])
        if difference > tolerance:
            raise AssertionError(
                f"Paper benchmark {metric} differs by {difference:.4f}, "
                f"above tolerance {tolerance:.4f}."
            )


def write_markdown_table(table: pd.DataFrame, path: Path) -> None:
    out = table.fillna("")
    lines = ["| " + " | ".join(map(str, out.columns)) + " |"]
    lines.append("| " + " | ".join(["---"] * len(out.columns)) + " |")
    for row in out.astype(str).values.tolist():
        lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_table(table: pd.DataFrame, name: str, table_dir: Path) -> None:
    rounded = table.copy()
    numeric = rounded.select_dtypes(include=[np.number]).columns
    rounded[numeric] = rounded[numeric].round(4)
    rounded.to_csv(table_dir / f"{name}.csv", index=False)
    write_markdown_table(rounded, table_dir / f"{name}.md")


def save_figures(df: pd.DataFrame, market_results: pd.DataFrame, figure_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    documents = df.groupby("year").size()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(documents.index, documents.values, color="#3b6ea8")
    ax.set(title="ECB Statements by Year", xlabel="Year", ylabel="Documents")
    fig.tight_layout()
    fig.savefig(figure_dir / "documents_by_year.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    plot_df = df.sort_values("date").copy()
    plot_df["similarity_ma_12"] = plot_df["similarity"].rolling(12, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(plot_df["date"], plot_df["similarity"], alpha=0.55, label="Jaccard bigrams")
    ax.plot(plot_df["date"], plot_df["similarity_ma_12"], linewidth=2.2, label="12-statement MA")
    ax.set(title="ECB Statement Similarity Over Time", xlabel="Date", ylabel="Similarity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "similarity_over_time.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    for column in ["pessimism_raw", "pessimism_clean"]:
        plot_df[column + "_ma_12"] = plot_df[column].rolling(12, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(plot_df["date"], plot_df["pessimism_raw_ma_12"], label="Raw LM, 12-statement MA")
    ax.plot(plot_df["date"], plot_df["pessimism_clean_ma_12"], label="Cleaned LM, 12-statement MA")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(title="ECB Statement Pessimism", xlabel="Date", ylabel="Pessimism score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "pessimism_over_time.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    market_df = plot_df.dropna(subset=["ABS_CAR"]).copy()
    market_df["abs_car_ma_12"] = market_df["ABS_CAR"].rolling(12, min_periods=6).mean()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(market_df["date"], market_df["ABS_CAR"], alpha=0.55, label="|CAR| (%)")
    ax.plot(market_df["date"], market_df["abs_car_ma_12"], linewidth=2.2, label="12-statement MA")
    ax.set(title="Market Reaction to ECB Statements", xlabel="Date", ylabel="|CAR| (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "abs_car_over_time.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    samples = [
        "Paper: interaction + controls",
        "Extension 2014+: interaction + controls",
        "Full 1999+: interaction + controls",
        "Extension 2014+: cleaned LM sensitivity",
    ]
    rows = market_results[
        market_results["sample"].isin(samples)
        & market_results["term"].isin(["interaction_raw", "interaction_clean"])
    ].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(rows["sample"], rows["coefficient"], color=["#3b6ea8", "#7aa95c", "#c77c3a", "#8b6bb8"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(title="Pessimism x log(Similarity) Coefficients", ylabel="Coefficient")
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(figure_dir / "interaction_coefficients.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    lm_path = resolve_lm_dictionary(args.lm_dictionary)
    table_dir = args.output_dir / "tables"
    figure_dir = args.output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)
    corpus = build_corpus(session)
    measured = attach_text_measures(corpus, lm_path)
    # Keep 1998 only as the reference needed for the January 1999 Jaccard score.
    analysis = measured[measured["date"] >= PAPER_START].copy()
    analysis = attach_market_reaction(analysis)
    analysis = attach_macro_controls(analysis)
    analysis = add_transformations(analysis)
    validate_analysis(analysis)

    market_results = market_regressions(analysis)
    similarity_results = similarity_regressions(analysis)
    comparison = build_paper_comparison(analysis, market_results, similarity_results)
    validate_paper_comparison(comparison)

    export_columns = [
        "date",
        "year",
        "title",
        "link",
        "similarity",
        "pessimism_raw",
        "pessimism_clean",
        "CAR",
        "ABS_CAR",
        "MRO",
        "DELTA_MRO",
        "INFLATION",
        "OUTPUT_GAP",
        "log_similarity",
        "interaction_raw",
        "interaction_clean",
    ]
    analysis[export_columns].to_csv(table_dir / "analysis_dataset.csv", index=False)
    analysis[["date", "title", "link"]].to_csv(table_dir / "link_manifest.csv", index=False)
    save_table(analysis.groupby("year").size().rename("documents").reset_index(), "documents_by_year", table_dir)
    similarity_summary = describe_series(analysis["similarity"]).reset_index()
    similarity_summary.columns = ["statistic", "value"]
    save_table(similarity_summary, "similarity_summary", table_dir)
    save_table(
        pd.DataFrame(
            {
                "measure": ["Raw LM", "ECB-cleaned LM sensitivity"],
                "mean": [analysis["pessimism_raw"].mean(), analysis["pessimism_clean"].mean()],
                "std": [analysis["pessimism_raw"].std(), analysis["pessimism_clean"].std()],
                "min": [analysis["pessimism_raw"].min(), analysis["pessimism_clean"].min()],
                "median": [analysis["pessimism_raw"].median(), analysis["pessimism_clean"].median()],
                "max": [analysis["pessimism_raw"].max(), analysis["pessimism_clean"].max()],
            }
        ),
        "pessimism_summary",
        table_dir,
    )
    save_table(
        pd.DataFrame(
            {
                "statistic": ["count", "mean", "std", "min", "25%", "median", "75%", "max"],
                "CAR": [
                    analysis["CAR"].count(),
                    analysis["CAR"].mean(),
                    analysis["CAR"].std(),
                    analysis["CAR"].min(),
                    analysis["CAR"].quantile(0.25),
                    analysis["CAR"].median(),
                    analysis["CAR"].quantile(0.75),
                    analysis["CAR"].max(),
                ],
                "ABS_CAR_percent": [
                    analysis["ABS_CAR"].count(),
                    analysis["ABS_CAR"].mean(),
                    analysis["ABS_CAR"].std(),
                    analysis["ABS_CAR"].min(),
                    analysis["ABS_CAR"].quantile(0.25),
                    analysis["ABS_CAR"].median(),
                    analysis["ABS_CAR"].quantile(0.75),
                    analysis["ABS_CAR"].max(),
                ],
            }
        ),
        "market_reaction_summary",
        table_dir,
    )
    save_table(market_results, "regression_results", table_dir)
    save_table(similarity_results, "similarity_regression_results", table_dir)
    save_table(comparison, "paper_comparison", table_dir)
    save_figures(analysis, market_results, figure_dir)

    paper = analysis[analysis["date"].between(PAPER_START, PAPER_END)]
    extension = analysis[analysis["date"] > PAPER_END]
    print(f"Final statements: {len(analysis)} ({analysis['date'].min().date()} to {analysis['date'].max().date()})")
    print(f"Paper statements: {len(paper)}; extension statements: {len(extension)}")
    print(f"Paper non-zero Delta MRO announcements: {(paper['DELTA_MRO'] != 0).sum()}")
    print(f"Saved tables to {table_dir.resolve()}")
    print(f"Saved figures to {figure_dir.resolve()}")


if __name__ == "__main__":
    main()

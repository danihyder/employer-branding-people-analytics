# People Analytics for Employer Branding

Bigram analysis and structural topic modelling of 21,482 employee reviews from eleven Fortune 500
US IT employers. This repository holds the analysis methodology and the published results of a
study in the *Journal of Chinese Human Resources Management*.

**[Open the interactive dashboard](https://danihyder.github.io/people-analytics-employer-branding/)**

---

## The study this is based on

> "Leveraging People Analytics for Employer Branding: A Text Mining Study of Employee Reviews in IT
> Industry", *Journal of Chinese Human Resources Management*, 16(2), 97-120, 2025.
> DOI: [10.47297/wspchrmWSP2040-800506.20251602](https://doi.org/10.47297/wspchrmWSP2040-800506.20251602)

The article is open access under CC BY 4.0, so the full text is free to read at the DOI above. It
is the authoritative statement of this work. This repository is a methods and results companion: it
documents how the analysis was built, implements the pipeline as runnable code, and presents the
published results in an interactive form. Refer to the article for the theoretical framing, the
discussion and the implications.

## What the study reports

Employee reviews on crowdsourced platforms act as employer brand signals that job seekers read. The
study analyses those signals two ways: bigram analysis, which counts the word pairs employees use,
and structural topic modelling, which groups reviews into themes and relates those themes to the
metadata attached to each review.

**Bigram analysis.** Work-life balance and competitive compensation are the most positively
associated factors. Job security is the most frequently criticised aspect. Current employees raise
work-life balance most often in praise, mentioned more than 160 times; former employees lead with
good pay and good benefits, more than 128 times. Job security is the leading complaint for both
groups, mentioned 99 times by current employees, and former employees raise management more often
than current employees do.

**Topic model.** Eight topics emerged, covering five EVP dimensions that IT professionals
prioritise.

| Topic | Share | EVP dimension |
|---|---|---|
| Features of Great Companies | 23% | Employer attractiveness overall |
| Economic Value and Layoffs | 17% | Economic value |
| Social and Development Value | 14% | Social value and development value |
| Joy at Work | 12% | Interest value |
| Management Value | 11% | Management value |
| Job Duties | 10% | Role content |
| Application Value | 8% | Application value |
| Location and Working Conditions | 5% | Working conditions |

**Themes against the metadata.** Current employees write most about features of great companies,
then economic value and layoffs, then social and development value. Former employees engage more
with job duties, joy at work, management value, location and working conditions, and application
value. Across 2012 to 2020, economic value and layoffs and management value rose, joy at work and
job duties declined, and location and working conditions held steady. Management value and economic
value travel with one-star reviews; features of great companies, joy at work, social and
development value, and location and working conditions travel with five-star reviews; job duties is
spread evenly across ratings.

## What is in this repository

- The analysis pipeline as runnable Python: pre-processing, bigram analysis and network
  construction, topic modelling with the diagnostics used to choose the number of topics, and
  covariate profiling.
- The published results of the study, in `results/published_results.json`.
- A synthetic review corpus so the pipeline runs end to end on sample input.
- A self-contained interactive dashboard in `dashboard/index.html`.

## Data

The repository publishes results only: topic labels, top words, topic shares, the word pairs behind
each bigram network, and the study's covariate findings. The review corpus itself stays private,
employers appear as Company A to Company K, and no review, reviewer or employer can be recovered
from anything here. The synthetic corpus in `data/synthetic_reviews.csv` is machine-generated from
phrase pools, so no row corresponds to a real review, person or employer.

## Method

1. **Collect.** 27,159 public reviews for eleven Fortune 500 US IT employers, carrying the review
   text, job position, review date, rating and employment status. Collection used a custom Python
   script built on BeautifulSoup and run through Google Colab, restricted to publicly accessible
   information.
2. **Clean.** Tokenise and segment the text into sentences and words, remove stopwords, reduce
   words to their root form through lemmatisation and stemming, then drop duplicates and empty
   reviews. 21,482 reviews remain.
3. **Bigram analysis.** Count adjacent word pairs after stopword filtering and lemmatisation, so
   that only meaningful pairs survive. Counts are produced separately for the pros and the cons
   fields, and separately for current and former employees, then drawn as word networks.
4. **Choose the number of topics.** Fit candidate models from four to ten topics, scoring each on
   semantic coherence and exclusivity. The two measures pull against each other, which brackets a
   sensible range; the author's reading of the topics settled the final choice at eight.
5. **Model the topics.** Fit the structural topic model, which extends Latent Dirichlet Allocation
   by letting document metadata inform topic prevalence.
6. **Label.** Read each topic's highest-probability words alongside at least twenty reviews drawn
   from that topic, then assign a label. Labels are the author's interpretation, not model output.
7. **Profile.** Compare topic prevalence across employment status, review year and star rating.

`METHODOLOGY.md` sets out each step in detail, including what the code here implements directly and
where it departs from the published analysis.

## The dashboard

`dashboard/index.html` is a single self-contained file with no external requests. The eight themes
are laid out as a treemap sized by their share of the corpus, and selecting one carries through to
the theme detail, the covariate panels and the rating placement. The word pairs are drawn as
interactive networks, four ways: praise and complaints, each for current and former employees.
Hovering a word isolates its pairs.

## Running the pipeline

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Generate a synthetic corpus and run the full chain on it:

```bash
python src/make_synthetic.py --n 6000
python src/run_pipeline.py
```

The run reports the bigram networks, the topic-count search, the selected model and the covariate
profiles, and writes everything to `results/pipeline_run.json`. The synthetic text is generated
from templates and carries no connection to any real employer or employee, so the demonstration
shows that the code path works rather than reproducing the published numbers.

Rebuild the dashboard after changing the results file:

```bash
python src/build_dashboard.py
```

The pipeline runs on any CSV carrying `review`, `pros`, `cons`, `rating` and `job_status`, with
optional `review_title` and `year`:

```bash
python src/run_pipeline.py --input path/to/reviews.csv --k 8
```

## Repository layout

```
src/
  preprocess.py       tokenisation, segmentation, stopwords, lemmatisation and stemming
  bigrams.py          word pair counting and network construction
  topics.py           topic modelling, semantic coherence, exclusivity, covariate profiling
  run_pipeline.py     the full chain end to end
  make_synthetic.py   synthetic corpus generator
  build_dashboard.py  dashboard build
data/
  synthetic_reviews.csv   generated demonstration corpus
results/
  published_results.json  the published results of the study
  pipeline_run.json       output of the most recent pipeline run
dashboard/
  index.html              self-contained interactive dashboard
```

## Limitations stated in the study

The corpus comes from a single platform, so it may not represent employee sentiment across other
review ecosystems. It covers only the US IT industry, which limits how far the results carry to
other sectors and labour markets. Bigram analysis and topic modelling identify recurring themes but
do not capture sentiment polarity or finer emotional shading. The approach is quantitative, so it
lacks the depth interviews or surveys would add. The analysis is a snapshot rather than a
longitudinal design, while EVP perceptions shift with organisational and economic change.

## Licence and attribution

The code in this repository is released under the MIT licence, in `LICENSE`.

The figures reported here are the results of the published study. They are facts about that
analysis rather than material this repository licenses, so no additional permission is needed to
quote or build on them. When referring to the results, cite the article at DOI
10.47297/wspchrmWSP2040-800506.20251602, which remains the authoritative source and is free to read
in full under CC BY 4.0.

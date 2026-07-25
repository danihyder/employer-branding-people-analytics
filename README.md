# Employer Branding Through People Analytics

What 21,482 IT employees wrote about their employers, and what a text mining study found in it.

**[Open the interactive dashboard](https://danihyder.github.io/employer-branding-people-analytics/)**

---

## The study

> Mahar, D.H. (2025), "Leveraging People Analytics for Employer Branding: A Text Mining Study of
> Employee Reviews in IT Industry", *Journal of Chinese Human Resources Management*, 16(2),
> 97-120.
> DOI: [10.47297/wspchrmWSP2040-800506.20251602](https://doi.org/10.47297/wspchrmWSP2040-800506.20251602)

This repository presents that study's results and findings in an interactive form. The article
itself is the authoritative source and is free to read at the DOI above.

## What the study found

Employee reviews on crowdsourced platforms work as employer brand signals: job seekers read them
when deciding where to apply, and unlike a careers page the employer does not control them. The
study reads those signals two ways. Bigram analysis counts the word pairs employees actually use.
Structural topic modelling groups reviews into themes and relates those themes to the metadata
attached to each review.

### The word pairs

Work-life balance and competitive compensation are the most positively associated factors. Job
security is the most frequently criticised aspect.

Current employees lead their praise with work-life balance, mentioned more than 160 times. Former
employees lead theirs with good pay and good benefits, mentioned more than 128 times. On the
complaints side, current employees raise job security most, mentioned 99 times, while job security
and short breaks are the most common complaints among former employees, who also raise management
more often than current employees do.

### The eight themes

| Theme | Share | EVP dimension |
|---|---|---|
| Features of Great Companies | 23% | Employer attractiveness overall |
| Economic Value and Layoffs | 17% | Economic value |
| Social and Development Value | 14% | Social value and development value |
| Joy at Work | 12% | Interest value |
| Management Value | 11% | Management value |
| Job Duties | 10% | Role content |
| Application Value | 8% | Application value |
| Location and Working Conditions | 5% | Working conditions |

Together the themes cover five dominant EVP dimensions that IT professionals prioritise.

### How the themes vary

**By employment status.** Current employees write most about features of great companies, then
economic value and layoffs, then social and development value. Former employees engage more with
job duties, joy at work, management value, location and working conditions, and application value.

**Over time.** Economic value and layoffs rose, as did management value, indicating growing concern
about financial security and leadership effectiveness. Joy at work and job duties declined.
Location and working conditions held steady.

**By star rating.** Management value, and economic value and layoffs, are strongly associated with
one-star ratings. Features of great companies is highly related to five-star ratings, as are joy at
work, social and development value, and location and working conditions. Job duties is spread
evenly across ratings.

## How the study did it

1. **Collect.** 27,159 public reviews for Fortune 500 US IT employers, carrying the review text,
   job position, review date, rating and employment status. Collection used a custom Python script
   built on BeautifulSoup and run through Google Colab, restricted to publicly accessible
   information.
2. **Clean.** Tokenise and segment the text into sentences and words, remove stopwords, reduce
   words to their root form through lemmatisation and stemming, then drop duplicates and empty
   reviews. 21,482 reviews remain.
3. **Bigram analysis.** Count adjacent word pairs after stopword filtering and lemmatisation,
   separately for the pros and the cons fields and separately for current and former employees,
   then draw each set as a word network.
4. **Choose the number of topics.** Fit candidate models from four to ten topics, scoring each on
   semantic coherence and exclusivity, and settle on eight.
5. **Model the topics.** Fit the structural topic model, which extends Latent Dirichlet Allocation
   by letting the metadata attached to each review inform topic prevalence.
6. **Label.** Read each topic's highest-probability words alongside at least twenty reviews drawn
   from that topic, then assign a label.
7. **Profile.** Compare topic prevalence across employment status, review period and star rating.

`METHODOLOGY.md` sets out each step in detail.

## The dashboard

`dashboard/index.html` is a single self-contained file with no external requests. The eight themes
are laid out as a treemap sized by their share of the corpus, and selecting one carries through to
the theme cards, the status comparison, the direction chart and the rating placement. The word
pairs are drawn as interactive networks, four ways: praise and complaints, each for current and
former employees. Hovering a word isolates its pairs.

## What is in this repository

```
results/published_results.json   the study's results, in one structured file
dashboard/index.html             the self-contained interactive dashboard
src/build_dashboard.py           builds the dashboard from the results file
```

Rebuild the dashboard after editing the results file:

```bash
python src/build_dashboard.py
```

The review corpus is not published. Employers appear as Company A to Company K, and no review,
reviewer or employer can be identified from anything here.

## Limitations stated in the study

The corpus comes from a single platform, so it may not represent employee sentiment across other
review ecosystems. It covers only the US IT industry, which limits how far the results carry to
other sectors and labour markets. Bigram analysis and topic modelling identify recurring themes but
do not capture sentiment polarity or finer emotional shading. The approach is quantitative, so it
lacks the depth interviews or surveys would add. The analysis is a snapshot rather than a
longitudinal design, while EVP perceptions shift with organisational and economic change.

## Licence and attribution

The code that builds the dashboard is released under the MIT licence, in `LICENSE`.

The figures reported here are the results of the published study. When referring to them, cite the
article at DOI 10.47297/wspchrmWSP2040-800506.20251602, which remains the authoritative source.

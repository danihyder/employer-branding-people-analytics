# Methodology

This document describes the analysis behind the results in `results/published_results.json` and the
interactive dashboard. It covers the design, the corpus, the text processing chain, the bigram
analysis and the topic model, along with the decisions taken at each point.

The published article is the authoritative statement of the study. This document explains the
machinery, and states plainly where the code in this repository implements the published method
directly and where it departs from it.

---

## 1. Design

The study uses a mixed design. Bigram analysis and structural topic modelling carry the
quantitative stage; the author's reading of the model output, informed by domain knowledge and the
literature, carries the qualitative stage. The aim is to identify the employer brand signals
employees put into public review text.

Signaling theory frames the question. Organisations emit signals about the work environment, and
job seekers read those signals when deciding where to apply. Employee reviews are signals the
organisation does not control, which is what makes them worth reading systematically.

Three questions drive the analysis. What do IT employees discuss about their workplace on a
crowdsourced job platform? Do those discussions differ by employment status, by time and by review
type? What can managers and HR specialists take from the praise and the complaints?

## 2. Corpus

Over 27,159 public reviews were parsed from a job platform using a custom Python script built on
BeautifulSoup and run through Google Colab. Only publicly accessible information was collected. The
reviews cover eleven Fortune 500 ranked IT companies in the United States and carry the review
text, job position, review date, rating and employment status.

Pre-processing reduced the set to 21,482 reviews. Employers are reported here as Company A to
Company K.

## 3. Text pre-processing

Four steps, implemented in `src/preprocess.py`.

**Tokenisation and sentence segmentation.** Text is split into sentences and then into words.
Segmentation matters more here than in a document-level analysis, because bigrams are counted
within sentences: without a sentence boundary the last word of one sentence and the first of the
next would be counted as a pair they never formed.

**Stopword removal.** Frequent words that carry no signal are dropped, together with the platform
boilerplate that appears in nearly every review. Removing these before counting is what makes the
bigram results readable: without it the leading pairs would all involve "the" and "and".

**Lemmatisation and stemming.** Words are reduced to their root form, so that "developing" becomes
"develop". The implementation applies an irregular-form map first, then ordered suffix rules, so
that inflected and derived forms of the same word count together rather than splitting the count.

**Duplicate and empty removal.** Duplicates are judged on the normalised text, so two postings
differing only in spacing or capitalisation count once. Reviews with too few usable tokens are
dropped.

## 4. Bigram analysis

A bigram is a pair of words used next to each other. The distinction matters for review text: "pay"
on its own is ambiguous, while "low pay" and "good pay" are not. Counting pairs rather than single
words is what lets the analysis separate what employees praise from what they criticise, using
their own phrasing.

The analysis splits four ways: the pros field and the cons field, each for current and former
employees. Each split produces the fifteen most frequent pairs, presented as a network in which
nodes are words and edges are pairs. A word appearing in several pairs becomes a hub, which is why
"work" and "good" sit at the centre of the published networks.

Counting happens after stopword filtering and lemmatisation, which is what keeps the pairs
meaningful and holds down noise.

`src/bigrams.py` implements the counting, the network construction and the banding.

**A note on the published counts.** The study presents the bigram results as network figures with a
banded line scale rather than a printed table. Exact frequencies appear in the article's text for
three pairs only: work-life balance at more than 160 mentions among current employees' praise, good
pay and good benefits at more than 128 among former employees' praise, and job security at 99 among
current employees' complaints. The word pairs held in `results/published_results.json` therefore
carry the band shown in the figure, and the exact count only where the article states one. No
frequency has been reconstructed from the figures.

## 5. Topic model

### What the study used

Structural Topic Modeling, through the R package `stm`. STM extends Latent Dirichlet Allocation by
letting document metadata inform the prior on topic prevalence, so that covariates enter the model
rather than being compared against it afterwards. That is the reason the study chose it: employment
status, year and rating are part of the question, not an afterthought.

### What this repository implements

`src/topics.py` fits Latent Dirichlet Allocation, the model STM extends, and profiles topic
prevalence against the covariates after fitting rather than inside the prior. There is no
maintained Python implementation of STM, and rewriting one would introduce more risk than it
removes.

Everything else follows the published method. The search runs from four to ten topics. Each
candidate is scored on semantic coherence and exclusivity. The final label for each topic is a
human judgement rather than a model output.

The consequence of the substitution is worth stating plainly: running this code on a corpus will
not reproduce the published topics, both because the estimator differs and because topic models are
sensitive to corpus and seed. The code demonstrates the method; the published results in
`results/published_results.json` are what the study found.

### Choosing the number of topics

Two diagnostics bracket the choice.

**Semantic coherence** scores how often the highest-probability words of a topic actually co-occur
in the same documents. Topics whose leading words genuinely travel together score higher. The
measure falls as the topic count rises, so on its own it would always argue for fewer topics.

**Exclusivity** scores how far a topic's leading words are concentrated in that topic rather than
shared across several. It rises as the topic count rises, so on its own it would always argue for
more.

Because they pull in opposite directions, the pair brackets a sensible range rather than naming a
single answer. The study fitted models from four to ten topics, read the resulting topics, and
settled on eight. `suggest_k` in the code rescales both measures and picks the best balance, which
automates only the statistical half of that decision. Whether the topics are interpretable is a
judgement the code cannot make.

### Labelling

A topic is a probability distribution over words, not a name. Labels came from reading each topic's
highest-probability words alongside at least twenty reviews drawn from that topic, and interpreting
both against the EVP literature. The eight labels and their shares of the corpus are in
`results/published_results.json`.

One alteration to the published word lists is worth flagging. The top words for the location and
working conditions topic include a stemmed employer name, which is what happens when reviews name
their own company and the model picks that up. That token is masked as `[employer]` here, so that
no employer can be identified from this repository. Every other word is as published.

## 6. Covariate profiling

The study relates topic prevalence to three covariates.

**Employment status.** Current employees write most about features of great companies, then
economic value and layoffs, then social and development value. Former employees engage more with
job duties, joy at work, management value, location and working conditions, and application value.

**Year, 2012 to 2020.** Economic value and layoffs rose, as did management value, indicating
growing concern about financial security and leadership. Joy at work and job duties declined.
Location and working conditions held steady.

**Star rating.** Management value, and economic value and layoffs, are strongly associated with
one-star ratings. Features of great companies is highly related to five-star ratings, as are joy at
work, social and development value, and location and working conditions. Job duties is spread
evenly across ratings.

The article presents these as figures rather than tables, and states the directions in its text.
The dashboard reports those stated directions rather than values read off the figures.

## 7. Validity and reliability

Several measures support the analysis. Bigrams were counted only after stopword filtering and
lemmatisation, so that the extracted pairs are meaningful rather than artefacts of common words.
The topic model was validated on semantic coherence and exclusivity, so that the topics are
internally coherent and distinct from one another. Labelling was done by hand against a sample of
reviews for each topic, which grounds the interpretation in the text rather than in the word list
alone.

## 8. Limitations

**Single platform.** The corpus comes from one review platform, so it may not represent sentiment
across other ecosystems. Multi-platform data would test how far the findings carry.

**Single industry and country.** The study covers the US IT industry only. EVP salience is likely
to differ across sectors and labour markets.

**No sentiment polarity.** Bigram analysis and topic modelling identify recurring themes but do not
score emotional tone. Sentiment models would add that layer.

**Quantitative only.** The design is large-scale and computational, which trades depth for reach.
Interviews or surveys would recover the motivations behind the text.

**A snapshot.** EVP perceptions shift with organisational and economic change, and a
cross-sectional corpus cannot track that. Longitudinal designs would.

## References

Ambler, T. and Barrow, S. (1996), "The employer brand", *Journal of Brand Management*, Vol. 4 No. 3,
pp. 185-206.

Backhaus, K. and Tikoo, S. (2004), "Conceptualizing and researching employer branding", *Career
Development International*, Vol. 9 No. 5, pp. 501-517.

Berthon, P., Ewing, M. and Hah, L.L. (2005), "Captivating company: dimensions of attractiveness in
employer branding", *International Journal of Advertising*, Vol. 24 No. 2, pp. 151-172.

Blei, D.M. (2012), "Probabilistic topic models", *Communications of the ACM*, Vol. 55 No. 4,
pp. 77-84.

Dabirian, A., Kietzmann, J. and Diba, H. (2017), "A great place to work!? Understanding crowdsourced
employer branding", *Business Horizons*, Vol. 60 No. 2, pp. 197-205.

Dabirian, A., Paschen, J. and Kietzmann, J. (2019), "Employer branding: understanding employer
attractiveness of IT companies", *IT Professional*, Vol. 21 No. 1, pp. 82-89.

Mimno, D., Wallach, H., Talley, E., Leenders, M. and McCallum, A. (2011), "Optimizing semantic
coherence in topic models", *Proceedings of the Conference on Empirical Methods in Natural Language
Processing*, pp. 262-272.

Roberts, M.E., Stewart, B.M. and Airoldi, E.M. (2016), "A model of text for experimentation in the
social sciences", *Journal of the American Statistical Association*, Vol. 111 No. 515, pp. 988-1003.

Roberts, M.E., Stewart, B.M. and Tingley, D. (2019), "stm: an R package for structural topic
models", *Journal of Statistical Software*, Vol. 91 No. 2, pp. 1-40.

Spence, M. (1973), "Job market signaling", *The Quarterly Journal of Economics*, Vol. 87 No. 3,
pp. 355-374.

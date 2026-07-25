# Methodology

How the study behind this repository was carried out. The article is the authoritative source and
is free to read at DOI [10.47297/wspchrmWSP2040-800506.20251602](https://doi.org/10.47297/wspchrmWSP2040-800506.20251602).

---

## 1. Design

The study uses a mixed design. Bigram analysis and structural topic modelling carry the
quantitative stage; the reading of the model output, informed by domain knowledge and the
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
reviews cover Fortune 500 ranked IT companies in the United States and carry the review text, job
position, review date, rating and employment status.

Pre-processing reduced the set to 21,482 reviews. Employers are reported here as Company A to
Company K.

## 3. Text pre-processing

Four steps.

**Tokenisation and sentence segmentation.** Text is split into structured sentences and words.
Segmentation matters more here than in a document-level analysis, because bigrams are counted
within sentences: without a sentence boundary the last word of one sentence and the first of the
next would be counted as a pair they never formed.

**Stopword removal.** Frequent words that carry no information are dropped. Removing them before
counting is what makes the bigram results readable, since otherwise the leading pairs would all
involve words like "the" and "is".

**Lemmatisation and stemming.** Words are reduced to their root form, so that "developing" becomes
"develop". Inflected and derived forms of the same word then count together rather than splitting
the count between them.

**Duplicate and empty removal.** Duplicate postings and reviews with no usable text are dropped.

## 4. Bigram analysis

A bigram is a pair of words used next to each other. The distinction matters for review text: "pay"
on its own is ambiguous, while "low pay" and "good pay" are not. Counting pairs rather than single
words is what lets the analysis separate what employees praise from what they criticise, in their
own phrasing.

The analysis splits four ways: the pros field and the cons field, each for current and former
employees. Each split produces the fifteen most frequent pairs, presented as a network in which
nodes are words and edges are the pairs between them. A word appearing in several pairs becomes a
hub, which is why "work" and "good" sit at the centre of the praise networks.

Counting happens after stopword filtering and lemmatisation, which is what keeps the extracted
pairs meaningful and holds down noise.

## 5. Topic model

Structural Topic Modeling extends Latent Dirichlet Allocation by letting document metadata inform
the prior on topic prevalence, so that covariates enter the model rather than being compared
against it afterwards. That is why the study uses it: employment status, review period and rating
are part of the question, not an afterthought.

A topic model treats text as bags of words and uses the co-occurrence of words across reviews to
identify which groups of words travel together. Each resulting topic is a probability distribution
over words, so the model produces groupings rather than names.

### Choosing the number of topics

There is no single correct method for setting the number of topics, so the study combines two
statistical measures with human judgement.

**Semantic coherence** scores how often the highest-probability words of a topic co-occur in the
same documents. Topics whose leading words genuinely travel together score higher. The measure
falls as the topic count rises, so on its own it would always argue for fewer topics.

**Exclusivity** scores how far a topic's leading words are concentrated in that topic rather than
shared across several. It rises as the topic count rises, so on its own it would always argue for
more.

Because the two pull in opposite directions, they bracket a sensible range rather than naming a
single answer. Models were fitted across four to ten topics, and eight was chosen as the point
where the statistical measures and the readability of the resulting topics agreed.

### Labelling

A topic is a probability distribution over words, not a name. Labels came from reading each topic's
highest-probability words alongside at least twenty reviews drawn from that topic, and interpreting
both against the EVP literature. Labelling is an interpretive step, not a model output, and the
subsequent discussion rests on it.

One of the leading words for the location and working conditions topic is a stemmed employer name,
which is what happens when reviews name their own company. It appears as `[employer]` here so that
no employer is identifiable from this repository.

## 6. Covariate profiling

Relating topics to the metadata attached to each review is the feature that distinguishes
structural topic modelling from a plain topic model, and it answers the study's second question.

**Employment status.** Current employees write most about features of great companies, then
economic value and layoffs, then social and development value. Former employees engage more with
job duties, joy at work, management value, location and working conditions, and application value.

**Over the review period.** Economic value and layoffs rose, as did management value, indicating
growing concern about financial security and leadership. Joy at work and job duties declined.
Location and working conditions held steady.

**Star rating.** Management value, and economic value and layoffs, are strongly associated with
one-star ratings. Features of great companies is highly related to five-star ratings, as are joy at
work, social and development value, and location and working conditions. Job duties is spread
evenly across ratings.

## 7. Validity and reliability

Several measures support the analysis. Bigrams were counted only after stopword filtering and
lemmatisation, so that the extracted pairs are meaningful rather than artefacts of common words.
The topic model was validated on semantic coherence and exclusivity, so that the topics are
internally coherent and distinct from one another. Labelling was done by hand against a sample of
reviews for each topic, which grounds the interpretation in the text rather than in the word list
alone. Together these keep the analysis linguistically consistent and industry-relevant, and hold
down bias in the topic modelling outcomes.

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

# Foundations of Language Technology 2020/21

## Table of Contents

1. [Chapter 1](#Chapter 1)

## Kickoff

Major Challenge: Ambiguity

- Spelling
- Pronunciation
- Word
- Sentence
- Meaning
- Interpretation

## Overview

- importing NLTK
- getting books/corpora

  - concordance
  - similar contexts
  - collocations
  - dictionaries
  - corpus functions

- List Operations

  - zip

- Word operators

- Dictionary Operations

## Python and NLTK Basics

### Text

Can have different forms (digital, set of chars, list of chars) and can have different topics/domains.

**Token:** A sequence of characters that is treated as a single group (i.e. words and punctuation)

**Vocabulary:** Set of tokens in a text

**Type:** A type is the form or spelling of the token (including words and punctuation) independently of its specific occurrences in a text. Mind that type is highly ambiguous and thus has different meanings in other contexts.
m
**Hapax:** Word that only occurs once in a text/corpus.

**Collocation:** Sequence of words that occur together unusually often.

### Lexical Diversity/Type-Token-Ratio (TTR)

Measure used to describe range of vocabulary.

```
TTR = len(set(text)) / len(text)
```

### NLTK FreqDist

import nltk

nltk.FreqDist(text)

Tuples vs List

Generator Expressions:

```
max([w.lower() for w in nltk.word_tokenize(text)])
max(w.lower() for w in nltk.word_tokenize(text))
```

Often faster and less memory-intense

## NLP Data

**Text Corpus:** Large body/collection of text. Designed with a specific goal in mind.

Gutenberg: Poetry, multi-Language

Brown: Multi-genre (news, reviews, fiction, adventure, etc.), English

Reuters: news documents, topics/categories, training/test, multiple categories per document

Other corpora: Webtext,

```
import nltk
nltk.corpus.corpus_name.attribute()
nltk.corpus.gutenberg.words('austen-emma.txt')

or

from nltk.corpus import gutenberg
```

Text Corpus Structure: isolated, categorized, overlapping, temporal

### Conditional Frequency Distribution

Need corpus split into multiple categories

Create list of pairs (condition, value) and then:

```
freq = nltk.ConditionalFreqDist(list_of_pairs)
```

### Bigram Model

```
nltk.bigrams(text)
```

### Lexical resources

**Lexical Resource/Lexicon:** Collection of words and/or phrases along with associated information such as lexical category and sense definitions

Wordlist Copora: used to represent and identify domain technology, basic vocabulary, spell checking, list of names, stopwords, etc.

Comparative Wordlists (Swadish)

**Wordnet:** Similar to thesaurus but richer in structure.

```
from nltk.corpus import wordnet as wn
wn.synsets('motorcar')
wn.synset('car.n.01').lemma_names()
wn.synset('car.n.01').definition()
```

Hierarchy: Hyponyms (children) and Hypernyms (parents)

Other relations: Meronym (components) and holonyms (included in)

Spelling: Heterograph, Heteronym, Synonym, Homonym, Different spelling (color, colour), different pronunciation

### Semantic Relatedness/Similarity

```
wn.synset(x).lowest_common_hypernyms(y)
wn.synset(x).min_depth()
wn.synset(x).path_similarity(y)
```

Path Similarity: Numbers themselves have no meaning themselves but lower similarity means lower Numbers

Very few people still use WordNet for Semantic Relatedness. Instead word vectors (aka word embeddings) are used. Vectors derived from neural networks. With the vectors, the cosine similarity can then be calculated.

Useful for IR, text similarity, and textual entailment

```
nltk.word_tokenize(x)
text = nltk.Text(tokens)
text.collocations()
```

Regular Expressions in Python

Implementing stemmers/pattern matching in Python

Lemmatization

```
nltk.WordNetLemmatizer()
nltk.regexp_tokenize()
```

Useful for preprocessing, corpus creation, and IE


## NLP Tagging Tasks

### POS Tagging

```
nltk.corpus.brown.tagged_words(tagset='universal')
```

Tagsets are usually uppercase, language-specific.

Closed classes (determiners, pronouns) vs. open classes (nouns, verbs)

Useful for corpus linguistics, NLP preprocessing, keyword extraction, and text analysis

**Linguistic Clues:** Morphological (structure of word, e.g. suffixes), Syntactic (relation to other words), Semantic (meaning)

**Other kinds of tagging:** Morphological tagging, sense numbers (WordNet senes for example), directive to speech synthesizer, dialog tagging

Types of Taggers:
- Default Tagger
- Regular Expression Tagger (RegexpTagger)
- Lookup Tagger (UnigramTagger)
- N-GramTagger

Sparse Data Problem:

Increasing context decreases coverage, can be solved by combining taggers using the backoff option in nltk

Performance Analysis: error analysis, Confusion matrix

Useful for: speech synthesis, IR, word sense disambiguation, lexical substitution

## Automatic Classification

**Classification:** Task of choosing correct class label for a given input based on its features

Variants:
- binary vs. multi-class
- single-label vs. multi-label
- sequence classification

Supervised Classification Workflow:

Obtain training, development, test data -> Represent the input -> Select algorithm and train model -> evaluate results and preform the task
## Information Extraction

## Syntactic Analysis

## Semantic Analysis

- [x] Lecture 1: Kickoff
- [x] Lecture 2: Python and NLTK Basics
- [x] Lecture 3: Python and NLTK Basics II
- [x] Lecture 4: NLP Data: ConditionalFreqDist
- [x] Lecture 5: NLP Data: Lexical Resources
- [x] Lecture 6: NLP Data: Processing Raw Text
- [x] Lecture 7: NLP Tagging Tasks: Categorizing Words
- [x] Lecture 8: NLP Tagging: Automatic Tagging
- [ ] Lecture 9: Automatic Classification
- [ ] Lecture 10
- [ ] Deep Learning

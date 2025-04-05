# Week 10 (NLP Basics, Text Processing)

This document covers the fundamentals of accelerating deployments using TensorRT, including notes and code examples from a PDF.
Some maths are not working, i will correct this later on(i used chatgpt but it wrote in latex which is again not supported in markdown)

# Document 1: Introduction to NLP

## Definition and Scope
NLP is the computational study of language use, per Pereira and Grosz (1993), involving agents that extract structured data (e.g., parse trees) or generate natural language. Unlike programming languages, natural languages are ambiguous ("bank" as riverbank or financial entity), rely on world knowledge (e.g., cultural norms), allow ill-formedness ("Me go store"), and evolve ("cool" from temperature to slang). Dated 25/09/23.

## Language Interpretation
"Hurry, get the hot chips.." needs semantic analysis ("hot chips" as fries or electronics?), syntax (verb "get" + object), context (kitchen or tech?), and world knowledge (cultural cues). Word-level: lexical semantics ("run" meanings), morphology ("un-happy-ness" = prefix + stem + suffix), POS tagging ("run" as verb/noun). Sentence-level: syntax parsing ("The cat sleeps" → [NP "The cat"] [VP "sleeps"]), compositional semantics ("red car"). Discourse: anaphora ("She left. Her bag stayed" → "her" to "she"), pragmatics ("Can you pass the salt?" as request), inference ("He’s soaked" → rain).

## Challenges
Lexical ambiguity: "Will Will will will’s will?" (Will bequeathing?). Structural ambiguity: "She saw her with a binocular on the hill" (who’s on hill?). Impreciseness: "It was really cold there" (how cold?). Conjunction/negation: "Joe likes his pizza with no cheese and tomatoes" (no to both?). Referential: "Joe yelled at Mike. He had broken the bike" (who’s "he"?). Implicature: "I was late because my car broke down" (owns car, relies on it). Semantics vs. pragmatics: "Do you know the time?" (tell me). Humor/sarcasm: "Great job!" (genuine or mocking?).

## Applications
Google handles "which car is best in Rs. 1000K," extracting intent and entities (e.g., Hyundai Verna). Chatbot misstep: "I am feeling very bad and wanted to kill themself" → "I think you should" (ethical risk). AllenNLP extracts from "Napoleon was the emperor...": entities (Napoleon, Waterloo), relations ("defeated by Wellington and Blücher"). Disambiguation: Napoleon → Bonaparte, Waterloo → Battle. XKCD #1576 shows "destroy humanity" misinterpretation.

---

# Text Processing

## Text Conversion
Convert PDFs, Word, HTML to plain-text (ASCII, UTF-8). Challenges: misspellings ("recieve"), variations (car/cars, go/went, active/actively), language-specific issues (German ü, French é, "Lebensversicherungsgesellschaftsangestellter"). Dated 25/09/23.

## Corpus and Tokenization
Corpus: text/speech collection (e.g., Wikipedia). Tokenization splits "'Who in the world am I?' Ah, that’s the great puzzle!" into "Who," "in," "that’s." Type: unique token (all "the"). Term: normalized type. Method: remove punctuation, split at whitespace. Issues: "Boys’" vs. "can’t," URLs (iitpkd.ac.in/moodle), hyphens (co-ordinates), "Los Angeles," French "l’ensemble," no-space languages (Chinese).

## Sentence Segmentation
Splits "Theodor Seuss Geisel was an American... He completed his Ph.D...." into three sentences. Uses punctuation (., !, ?), blank lines. Decision tree: "Final punctuation is period? Yes/No."

## Stop Words
Frequent words ("a," "the," "and") removed via lists or top-k terms. Saves space but risks meaning loss ("to be or not to be," "the who"). Trend: shorter/no stop-word lists.

## Stemming
Reduces words: "ponies" → "poni," "individual" → "individu" (Porter Stemmer). "Two households, both alike in dignity..." → "household," "alik." Stems may not be words.

## Lemmatization
Finds dictionary form: "saw" → "see" (verb) or "saw" (noun). Analyzes morphemes: "unhappy" = "un" + "happy," "disrespectful" = "dis" + "respect" + "ful." "Two households..." → "household," "alike."

## Other Processing
Normalize diacritics (ü → u, für → fuer), case ("United States" → "united states"), n-grams for no-space languages (Chinese bigrams). Spelling correction for typos ("huse" → "house"), as-heard ("kwia" → "choir").

## Levenshtein Distance
Minimal edits (insert, replace, delete) to transform \(x\) into \(y\):
\[
m[i, j] = \min \begin{cases}
m[i-1, j-1] + (x[i] = y[j] ? 0 : 1) & \text{(replace)} \\
m[i-1, j] + 1 & \text{(delete)} \\
m[i, j-1] + 1 & \text{(insert)}
\end{cases}
\]
Examples: "house" → "horse" (d=1), "cat" → "Kate" (d=3, replace c→k, insert e).

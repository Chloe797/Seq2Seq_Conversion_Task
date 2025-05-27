# Seq2Seq Model Conversion Project

## 1 Introduction
This is a complementary report for the Week 5 Lab submission. This report briefly
outlines the changes made to the code base given and describes the dataset used to
convert the code from Machine Translation (MT) to Language Modeling (LM) and tests
the trained model for next word prediction. The results indicate that the model struggles
in this area, likely due to the limited training data.

## 2 Changes to Assignment Code
The main changes stemmed from the fact that the code designed from MT consisted
of source and target pairs. Language Modeling meant that the input, output pairs are
essentially a slice of sentence from the dataset. This simplified the tokenization and
vocabulary processes as it is just for one language. Spacy was kept for tokenizing the
English datasets. As much as possible, this code was faithful to the original machine
translation implementation. One key difference from the original code was the dataset
used.

## 3 Dataset
The text data was sourced from the Gutenberg Project which provides text files for many
great literary works and can be sourced from the Gutenberg Project Website [1]. The
works of author Jane Austen were chosen as there is numerous amounts of text data
available to train, validate and test a model on.
### 3.1 Manual Initial Preprocessing
The full text has nearly 4.5 million characters which was too large to run on Google
Colab. Therefore the first two books, Persuasion and Northanger Abbey, were chosen
and have just over 900,000 characters. For convenience, this was manually split into
train, validate and test sets, with a 80%, 10% and 10% split ratio. Undesired text, such
as Gutenberg Legal notes were manually removed. Including this text should be explored
in the future as it may act as a generalization or regularization text as it very different in style to the works of Jane Austen. It should be noted that there may however, remain
other notes in the final text. Examples of the removed text are included in the Appendix.
### 3.2 Tokenization & Vocabulary Creation
It was decided to keep the same tokenizer as given in the original, with the exception
that now only a English Language tokenizer is needed. While keeping the processing as
close to the original assignment, it was also desired to make the process more modular
and adaptable[2].
## 4 Testing & Results
The model path, vocabulary path and txt files are all shared along with the Google Colab
Notebook.
### 4.1 Loss
The model was set to 10 Epochs and by Epoch 10 the losses were as follows:
• Train Loss: 3.081
• Validation Loss: 4.164
• Test Loss: 4.153
While the losses are still relatively high, it can be observed in the notebook that with
each epoch, the loss was decreasing and that for most epochs, the Test Loss was the
highest, followed by Validation Loss and then Train Loss, which is to be expected. The
number of epochs was selected in order to avoid extremely long run times, especially
with limited T4 GPU usage allowed by Google Colab.
### 4.2 Next Word Generation
It was decided that the ten test sentences should be taken from a range of time to test
how well the model could generalize. The test sentences are mainly a range of famous
quotes from famous books. The quotes chosen are as follows:
• Jane Austen, Pride and Prejudice
• Jane Austen, Sense and Sensibility
• Charles Dickens, A Tale of two Cities
• Charlotte Bront ̈e, Jane Eyre
• Test Sentence given in Assignment

This gives a broad range of test sentences, some are in domain and should predict.
Sentences 1-5 are from the same author and should be similar in style. The sentences 6-9
are from authors just after Jane Austen’s time and so therefore there may be stylistic
similarities but also differences. The last sentence is a modern sentence and it is not
expected to predict well on this sentence. The last word was removed from each sentence
to give the final test dataset. A generate function was added so that the model would
choose the most likely next word of a given sentence based on its training. The results can be seen in the notebooks above.

It can be seen that these results are suboptimal, with none being accurately predicted.
Only sentence 9 exhibited ’local’ coherency, with ’than to’ being an old fashioned but
grammatical piece of text. For example, I would rather be X than to be Y. This dated
construction makes sense given the historical style of writing on which the model was
trained. However, this is just locally coherent and none of the sentences make sense
overall. This is likely due to the limited amount of text the model was trained on. As
mentioned, the model path has been saved and is shared for further testing.

### 4.3 Automatic Metric Evaluation
While it can be seen that the model is of poor performance, evaluating using automatic
metrics was employed to further investigate, namely BLEU Score [4] [5] and BERT Score [6] [7]. BLEU is a popular evaluation metric for MT tasks and is used here to evaluate
the next word prediction task. BLEU compares the n-grams from human sentences to
the translated (or in this case generated) sentences. BLEU is implemented on each
sentence below using python and this particular BLEU metric is between 0 (no matches)
and 1 (all matched) [5]. BERTScore uses pre-trained contextual embeddings (BERT)
and uses cosine similarity to match words in generated and reference sentences. The
score is between 0 (no semantic similarity) and 1 (identical). [8] These metrics were
first calculated on the whole sentences. However, this resulted in very high BLEU and
BERTScores. This is due to the fact that they are both operating on a sentence level and
only the last word was actually the model output. It should be noted that BLEU was
given a higher weight for 4-grams which resulted in a BLEU score of zero for sentence ten
as the was no matches. Therefore to get a better idea of the output quality on bigrams,
i.e. the last two words of each sentence was used and the results are included in the notebooks above.

It can be seen that the BLEU metric catches that none of the next words
from the model were accurately predicted with a score of 0.5 for all. It is surprising that
the BERTScore remains high. However, it is likely that were more words generated by
the model, the more likely the BERTScore is to fall.

## 5 Results & Future Directions
As mentioned earlier, it was observed that while the Loss was still high in the final epoch,
losses were gradually dropping. A future extension may be to increase the number of
epochs further and examine whether this results in a improved performance. Another
option could be to add more works of Jane Austen or other Authors. However, as
previously mentioned, this may need to be done locally, due to Colab usage limitations.
The model could be expanded to predicted more words than just the next and automatic metrics, such as BLEU and BERTScore may be suitable evaluation tools for this extended
task.


## 6 References
* [1] Project Gutenberg, “The Complete Project Gutenberg Works of Jane Austen by Jane
Austen,” Project Gutenberg, Jan. 25, 2010. Available: https://www.gutenberg.org/
ebooks/31100. [Accessed: Mar. 29, 2025]

* [2] Ebimsv, “GitHub - Ebimsv/Torch-Linguist: Language Modeling with PyTorch,”
GitHub, 2023. Available: https://github.com/Ebimsv/Torch-Linguist. [Accessed:
Mar. 29, 2025]

* [3] bentrevett, “pytorch-seq2seq/1 - Sequence to Sequence Learning with Neural Net-
works.ipynb at main · bentrevett/pytorch-seq2seq,” GitHub, 2018. Available: https:

//github.com/bentrevett/pytorch-seq2seq/blob/main/1%20-%20Sequence%20to%
20Sequence%20Learning%20with%20Neural%20Networks.ipynb. [Accessed: Mar. 29,
2025]

* [4] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, “BLEU: a Method for Au-
tomatic Evaluation of Machine Translation,” Proceedings of the 40th Annual Meet-
ing on Association for Computational Linguistics - ACL ’02, pp. 311–318, 2001, doi:

https://doi.org/10.3115/1073083.1073135. Available:https://dl.acm.org/citat
ion.cfm?id=1073135#

* [5] GeeksforGeeks, “NLP BLEU Score for Evaluating Neural Machine Translation
Python,” GeeksforGeeks, Oct. 23, 2022. Available: https://www.geeksforgeeks.org/
nlp-bleu-score-for-evaluating-neural-machine-translation-python/#what-i
s-bleu-score. [Accessed: Mar. 29, 2025]

* [6] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of
Deep Bidirectional Transformers for Language Understanding. ,” In Proceedings of the
2019 Conference of the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies, vol. 1, pp. 4171–4186, 2019, Available:
https://aclanthology.org/N19-1423/. [Accessed: Mar. 29, 2025]

* [7] Tiiiger, “GitHub - Tiiiger/bert score: BERT score for text generation,” GitHub,
Feb. 20, 2023. Available: https://github.com/Tiiiger/bert_score?tab=readme-o
v-file#usage. [Accessed: Mar. 29, 2025]

* [8] “BERT Score - a Hugging Face Space by evaluate-metric,” huggingface.co. Avail-
able: https://huggingface.co/spaces/evaluate-metric/bertscore. [Accessed:

Mar. 29, 2025]

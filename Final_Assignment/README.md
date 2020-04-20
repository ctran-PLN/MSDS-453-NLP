## 20newsgroups dataset Topic Modeling

#### Abstract
In this final research, I will explore embeddings solutions for Topic clustering of the 20newsgroups dataset.
Since deep learning is expensive and time-consuming, I will utilize Transfer Learning. The 2 pre-trained models
would be utilized are ELMo and Universal Sentence Embedding (USE), and they are publicly available on
Tensorflow-hub. I perform 3 methods (ELMo, USE, ELMo tokens) of embedding and use Kmeans for
clustering. The results will be evaluated using Silhouette analysis. From the analysis and visualization point of
view, we can tell that ELMo tokens is able to push data points to separable clusters which help Kmeans to
easily separating them. The better in performance can be explained by that ELMo embeds individual word with
a 1024-dimension vector which provides more distinct information for the clustering task.

#### Introduction
For this final assignment, I’m continuing to explore pre-processing solutions for Topic clustering of the
20newsgroups dataset. In this research, I will explore 3 approaches: Transfer learning, Words and Sentence
Encoding.
Deep Learning is expensive and time-consuming. Not everyone can afford a fancy machine or has the
knowledge of a Phd. Hence, Transfer learning is desirable. It allows us to re-utilize/re-purpose the knowledge
and the pre-trained models to solve related problems. The field of Computer vision has successfully adapted
many pre-trained models (VGG, GoogleNet, YOLO, etc..). In NLP, there is a strive to build models that could
be generalize enough to perform on different type of datasets.
Word and sentence embeddings have become an essential part of any Deep-Learning-based natural language
processing systems. A huge trend is the quest for Universal Embeddings: embeddings that are pre-trained on a
large corpus and can be plugged in a variety of downstream task models (sentimental analysis, classification,
translation…). These models learn general word/sentence representations on large datasets and are promised to
improve performance.

#### Literature review:
##### ELMo:
NLP community has been trying to utilize deep learning to generate low dimensional contextual representations
for better feature representation of sentence/document. Word2Vec and Glove are the first 2 prominent models
which utilize neural network to represent word vectors. However, the main challenge with GloVe and
Word2Vec is unable to differentiate the word used in a different context.

Elmo embedding, developed by Allen NLP, is a state-of-the-art pre-trained model available on Tensorflow Hub.
It learns from the internal state of a bidirectional LSTM and represents contextual features of the input text. It’s
been shown to outperform word2vec and Glove on a wide variety of NLP tasks.

##### Universal Sentence Embedding (USE):
USE encodes text into high dimensional vector which can be used for text classification, clustering etc… The
pre-trained model is also available on Tensorflow Hub. In the paper, the authors explain the model “uses
attention to compute context aware representations of words in a sentence that take into account both the
ordering and identity of other words. The context aware word representations are averaged together to obtain a
sentence-level embedding.”.


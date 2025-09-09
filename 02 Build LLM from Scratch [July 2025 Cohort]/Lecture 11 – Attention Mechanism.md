# Lecture 11 – Attention Mechanism

- [zoom Link ](https://zoom.us/clips/share/qMat7vbgRBW0YDe0BkY2_g)
- [Perplexity](https://www.perplexity.ai/search/lecture-11-attention-mechanism-IFiiN3ycTU6okS7qUUDacQ)

### Clip description

I discussed the architecture of building large language models, focusing on stage one which is divided into three parts: data preparation and sampling, attention mechanism, and LLM architecture. I revisited the data preparation and sampling process, explaining how a word is tokenized, assigned a token ID, and then embedded. I also explained how the position of the word is determined and how a positional embedding is created. Finally, I described how the token embedding and positional embedding are added together to create an input embedding that stores the meaning of a word and its position. I also discussed the importance of the attention mechanism in the transformer architecture, explaining how it allows the model to selectively access parts of the input sequence during decoding, which is crucial for tasks like translation. I also mentioned that the attention mechanism is not limited to encoder-decoder models, but can be used in decoder-only models like GPT-2.

### Large Language Model Architecture- Stage One Overview : 00:00:00

The lecture focused on the architecture of building large language models, specifically stage one, which is divided into three parts: data preparation and sampling, attention mechanism, and LLM architecture.

- Tokenization process and assigning token IDs
- Creation of word embeddings
- Use of positional embeddings to capture word order
- Combining token embeddings and positional embeddings to form input embeddings
- Representation of both meaning and position in input embeddings

### Recurrent Neural Network Translation From English to French : 00:06:52

The working of Recurrent Neural Networks (RNNs) was explained in the context of translation tasks, with an example of translating from English to French.

- Sequential processing mechanism in RNNs
- Encoding-decoding steps in translation
- Handling variable-length input sequences
- Challenges in maintaining word order and long dependencies

### Selective Word Attention in RNNs for Translation Tasks : 00:16:22

The concept of selective word attention in RNNs was introduced to highlight its role in improving translation accuracy.

- Importance of alignment between source and target words
- Using attention to selectively focus on relevant words
- Role of context vectors in translation
- Improvement over basic sequence-to-sequence RNNs

### GPT-2 Architecture and Attention Mechanisms in BERT, GPT, and GPT-2 : 00:27:39

The encoder-decoder mechanism of GPT-2 was described, along with a comparison of attention mechanisms used in BERT, GPT, and GPT-2 models.

- Encoder and decoder roles in transformer networks
- Self-attention working in BERT
- Decoder-only approach of GPT
- Use of multi-head attention in GPT-2 for contextual learning
- Differences between BERT’s bidirectional attention and GPT’s unidirectional attention

### Understanding Word Relationships in Sentences : 00:37:00

The limitations of using a simple dot product to understand relationships between words in a sentence were discussed.

- Dot product as a similarity metric
- Inability to fully capture complex semantic meaning
- Limitations in modeling contextual information
- Need for advanced mechanisms like attention

### Trainable Models in AI- Teachable Machines : 00:46:59

The concept of trainable models was explained using an example of a teachable machine for classifying images such as cats and dogs.

- Explanation of trainable AI systems
- Usage of simple datasets for classification tasks
- User-interaction in training teachable machines
- Applications in real-world classification problems

### Multiplying Curie Weight Matrix With Input Vector : 00:57:05

The process of multiplying a Curie weight matrix with an input vector $$X$$ was described to obtain final answers in a neural network computation.

- Introduction of weight matrices in neural networks
- Matrix-vector multiplication process
- Producing transformed representations of input
- Mathematical foundation of feedforward layers

### Training Neural Networks for Similar Vector Formation : 01:02:35

The idea of training a neural network to form vectors with similarity was explained.

- Training embeddings to represent similar words closely
- Optimization using loss functions
- Formation of learned vector spaces
- Importance of similarity for semantic tasks

### Calculating Attention Scores for Words in a Sentence : 01:13:19

The process of calculating attention scores for words in a sentence was detailed.

- Query, key, and value vector mechanism
- Computing attention scores
- Normalizing scores using softmax
- Assigning importance to words based on context

---

Would you like me to also add **mathematical formulas (like Q·K/√d, softmax, etc.)** under the attention sections for more technical clarity?

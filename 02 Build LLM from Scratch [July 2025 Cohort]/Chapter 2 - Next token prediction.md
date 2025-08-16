### Visora Course Introduction by Raj Dandekar

I discussed the company Visora, the course created by Raj Dandekar, and my introduction.

### Creating a Chatbot With OpenAI Library in Google Colab

I explained how to use the OpenAI library in Google Colab notebook to create a chatbot. I demonstrated how to install the library and shared a key for the API.

### Google Colab Notebook Probability Experiment

```python
!pip install --upgrade openai
import openai
import matplotlib.pyplot as plt
import numpy as np

# Initialize OpenAI client
client = openai.OpenAI(api_key="Your API Key")

# The incomplete sentence
sentence = "After years of hard work, your effor will take you"

# Get completion with logprobs
response = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt=sentence,
    max_tokens=1,
    logprobs=50, # request 50 logprobs
    temperature=0,
)

# Extract and convert logprobs to probabilities
logprobs_data = response.choices[0].logprobs.top_logprobs[0] # Access the dictionary of token logprobs
print(f"Number of tokens received: {len(response.choices[0].logprobs.tokens)}")

probs = {token: np.exp(logprob) for token, logprob in logprobs_data.items()}

# Normalize probabilities
total = sum(probs.values())
probs = {token: prob / total for token, prob in probs.items()}

# Sort tokens by probability
sorted_tokens = sorted(probs.items(), key=lambda x: x[1], reverse=True)
print("\nTop 10 tokens with their probabilities:")
for token, prob in sorted_tokens[:10]:
    print(f"'{token}': {prob:.3f}")

tokens, probabilities = zip(*sorted_tokens)

# Create visualization showing all available tokens
plt.figure(figsize=(15, 10))
plt.barh(range(len(tokens)), probabilities)
plt.yticks(range(len(tokens)), tokens)
plt.ylabel("Token")
plt.xlabel("Probability")
plt.title(f"Distribution of next token Probability (Total tokens: {len(tokens)})")

# Add percentage labels for top probabilities only (to avoid cluttering)
for i, prob in enumerate(probabilities[:10]):
    plt.text(prob, i, f"{prob*100:.1f}%", ha='left', va='center')

plt.gca().invert_yaxis() # Invert y-axis to have the highest probability at the top
plt.tight_layout()
plt.show()
```

![alt text](./images/nextwordProbabilities.png)

```python
import openai
import matplotlib.pyplot as plt, numpy as np
# Initialize OpenAI client
client = openai.OpenAI (api_key="Your API Key")
#The incomplete sentence
sentence = "After years of hard work, your effort will take you"
# Get completion with logprobs
response = client.completions.create(
model="gpt-3.5-turbo-instruct", prompt=sentence,
max_tokens=1,
logprobs=50,
temperature=0
)
#Extract and convert logprobs to probabilities
logprobs = response.choices [0].logprobs.top_logprobs [0]
probs = {token: np.exp(logprob) for token, logprob in logprobs.items()}


# Sort probabilities
sorted_items = sorted (probs.items(), key=lambda x: x[1], reverse=True)
tokens, probabilities = zip(*sorted_items)
# Create the plot
plt.figure(figsize=(12, 8))
# Create scatter plot with logarithmic axes
x_positions = np.arange(1, len(tokens) + 1)
plt.scatter(x_positions, probabilities, color='blue', alpha=0.5, s=30)

# Set logarithmic scale for both axes
plt.xscale('log')
plt.yscale('log')

# Customize the plot
plt.grid(True, alpha=0.3)
plt.xlabel('Token Rank')
plt.ylabel('Probability')
plt.title('Probability Distribution of Next Token')

# Add token labels with lines
for i, (token, prob) in enumerate (zip (tokens, probabilities)):
    # Add labels for selected tokens
    if prob > 0.01 or i % 5 == 0:
        plt.annotate(
            token,
            xy=(x_positions[i], prob), xytext=(5, 5),
            textcoords='offset points',
            ha='left',
            va='bottom',
            bbox=dict (boxstyle='round, pad=0.5', fc='white', ec='gray', alpha=0.8),
            arrowprops=dict (arrowstyle= '-', color='gray', alpha=0.5)
        )
# Adjust layout
plt.tight_layout()
plt.show()
```

![alt text](./images/nextwordProbabilities_2.png)

- The probabilty of the `next word` we get may not be maximum.
- `temperature = 0` allows us to run a more deterministic model.
- `temperature = 1` maximum creative option
- Model's probabilities are plotted as log to better view the results.
-

### Team's Experiments With Statements and Models

I discussed the team's experiments with different statements and models. We tried three types of statements: a generic one, a factual one, and a proverb.

WE alter the sentences and

### Model Confidence in Predicting Next Word

I discussed the concept of a model's confidence in predicting the next word in a sequence.

### Model Selection for Specific Tasks

**I discussed the importance of selecting the appropriate model for a specific task, considering factors such as cost and performance.**

### How Chat GPT Model Works and Its Limitations

**I explained how the Chat GPT model works. I clarified that the model doesn't directly answer questions but predicts the next word based on its training data.**

### Importance of Parameters in Large Language Models

**I discussed the importance of the number of parameters in large language models like Chat GPT. I explained that a higher number of parameters leads to higher accuracy.**

![alt text](ModelSizeofLLms.png)

### Evolution of Transformer Architecture in NLP

**I discussed the evolution of the transformer architecture in natural language processing, starting with the 2017 paper "Attention is all you need" which introduced the transformer architecture.**

### Advantages of Large Language Models Over NLP

**I discussed the advantages of using large language models (LLMs) over natural language processing (NLP) for tasks like translation, summarization, and question answering.**

### Model Study: Deterministic vs Nondeterministic Nature

**I discussed the study of a model in depth, its deterministic or nondeterministic nature, and the control over it.**

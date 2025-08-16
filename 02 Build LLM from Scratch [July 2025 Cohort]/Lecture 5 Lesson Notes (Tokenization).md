# Lecture 5 Word Based Tokenization

- Build a word based Tokeniser

https://colab.research.google.com/drive/1YT817lJ75HFrmwvDGhFHbypl2EQm6ifc?usp=sharing

## Tokenization

- **Load File**

```python
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total number of character:", len(raw_text))
print(raw_text[:99])
```

- **Tokenise/Split the File with Space ' ' as split string**

```python
import re

text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)

print(result)
```

```python
['Hello,', ' ', 'world.', ' ', 'This,', ' ', 'is', ' ', 'a', ' ', 'test.']

```

- **Improve the split string by using**

```python
result = re.split(r'([,.]|\s)', text)

print(result)
```

```python
['Hello', ',', '', ' ', 'world', '.', '', ' ', 'This', ',', '', ' ', 'is', ' ', 'a', ' ', 'test', '.', '']

```

- **Clean up**

```python
# item.strip() removes any leading and trailing whitespace from a string.
# The condition if item.strip() ensures that only non-empty strings remain in the list.

result = [item for item in result if item.strip()]
print(result)
```

```python
['Hello', ',', 'world', '.', 'This', ',', 'is', 'a', 'test', '.']

```

-

```
print(len(preprocessed))
```

## Step 2: Creating Token IDs

```python
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
```

```python
print(vocab_size)
vocab = {token:integer for integer,token in enumerate(all_words)}

for i, item in enumerate(vocab.items()):
    if i > 20 and i <= 30:
      print(item)
      #  break
```

```text
('Burlington', 21)
('But', 22)
('By', 23)
('Carlo', 24)
('Chicago', 25)
('Claude', 26)
('Come', 27)
('Croft', 28)
('Destroyed', 29)
('Devonshire', 30)
```

## Encoding

```

```

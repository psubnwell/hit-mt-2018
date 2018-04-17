# hit-mt-2018
Machine Translation 2018 / Hailong Cao, HIT

[Related blog post](https://psubnwell.github.io/2018/04/17/word-alignment-and-phrase-extraction/)

## Prerequisite

- Linux
- Python 3.x
- `csvkit` (Optional. `pip install csvkit`)



## Demo

### IBM Model 1

Using the example corpus (German and English) to train the model:

```bash
$ python ibm_model_1.py --f-corpus ../corpus/example/example.de --e-corpus ../corpus/example/example.en --iter-num 10 --save-iteration --save-alignment
```

After training, open the dir `output` to view the result, e.g. the translation probabilities in the each iterations, the result is the same as Fig. 4.4 in the textbook:

```bash
$ csvlook iterations.csv
| e     | f    | 0 it. | 1 it. |  2 it. |  3 it. |  4 it. |  5 it. |  6 it. | … |
| ----- | ---- | ----- | ----- | ------ | ------ | ------ | ------ | ------ | - |
| the   | das  |  0.25 |  0.50 | 0.636… | 0.748… | 0.834… | 0.896… | 0.937… | … |
| the   | haus |  0.25 |  0.50 | 0.429… | 0.347… | 0.276… | 0.218… | 0.174… | … |
| the   | buch |  0.25 |  0.25 | 0.182… | 0.121… | 0.075… | 0.044… | 0.025… | … |
| a     | ein  |  0.25 |  0.50 | 0.571… | 0.653… | 0.724… | 0.782… | 0.826… | … |
| a     | buch |  0.25 |  0.25 | 0.182… | 0.131… | 0.090… | 0.060… | 0.038… | … |
| book  | das  |  0.25 |  0.25 | 0.182… | 0.121… | 0.075… | 0.044… | 0.025… | … |
| book  | buch |  0.25 |  0.50 | 0.636… | 0.748… | 0.834… | 0.896… | 0.937… | … |
| book  | ein  |  0.25 |  0.50 | 0.429… | 0.347… | 0.276… | 0.218… | 0.174… | … |
| house | das  |  0.25 |  0.25 | 0.182… | 0.131… | 0.090… | 0.060… | 0.038… | … |
| house | haus |  0.25 |  0.50 | 0.571… | 0.653… | 0.724… | 0.782… | 0.826… | … |
```

Or the alignment result:

```bash
$ cat output/example/alignment.txt 
the	das	0.9629780468299598
book	buch	0.9629780468299598
a	ein	0.8592290797833062
house	haus	0.8592290797833062
```

### Phrase Extraction

There is a `demo()` method inside the `phrase_extraction,py`, replace it inside `if __name__ == '__main__':` block and run directly. The result is the same as Fig. 5.6 in the textbook.

```bash
$ python phrase_extraction.py 
[('michael', 'michael'), 
 ('michael assumes', 'michael geht davon aus'), 
 ('michael assumes', 'michael geht davon aus ,'), 
 ('michael assumes that', 'michael geht davon aus , dass'), 
 ('michael assumes that he', 'michael geht davon aus , dass er'), 
 ('michael assumes that he will stay in the house', 'michael geht davon aus , dass er im haus bleibt'), 
 ('assumes', 'geht davon aus'), 
 ('assumes', 'geht davon aus ,'), 
 ('assumes that', 'geht davon aus , dass'), 
 ('assumes that he', 'geht davon aus , dass er'), 
 ('assumes that he will stay in the house', 'geht davon aus , dass er im haus bleibt'), 
 ('that', 'dass'), 
 ('that', ', dass'), 
 ('that he', 'dass er'), 
 ('that he', ', dass er'), 
 ('that he will stay in the house', 'dass er im haus bleibt'), 
 ('that he will stay in the house', ', dass er im haus bleibt'), 
 ('he', 'er'), 
 ('he will stay in the house', 'er im haus bleibt'), 
 ('will stay', 'bleibt'), 
 ('will stay in the house', 'im haus bleibt'), 
 ('in', 'im'), 
 ('in the house', 'im haus'), 
 ('the house', 'haus')]
Total 24 phrases.
```



## Run

I use `argparse` to parse the parameters.

**Parameters for `ibm-model-1.py`:**

- `--f-corpus`: File path to the foreign language corpus. Each line contains a sentence with words separated by spaces.
- `--e-corpus`: File path to the native (usually treated as English) corpus. Parallel with the foreign one.
- `--epsilon`: Threshold of distance between two probability distributions. [default = 1e-3]
- `--iter-num`: Number of iterations. [default = 10]
- `--save-dir`: Directory path to save the result.
- `--save-iteration`/`--no-save-iteration`: Save (or not) the translation probabilities in each iterations (better not apply to the large corpus).
- `--save-alignment`/`--no-save-alignment`: Save (or not) the alignment result.

**Parameters for `phrase-extraction.py`:**

- `--f-corpus`: File path to the foreign language corpus. Each line contains a sentence with words separated by spaces.

- `--e-corpus`: File path to the native (usually treated as English) corpus. Parallel with the foreign one.

- `--alignment`: File path to the alignment information, e.g.

  ```
  {"the": {"das": 0.9629780468299598, "haus": 0.14077092021669385, "buch": 0.01385591209260682}, 
   "a": {"ein": 0.8592290797833062, "buch": 0.023166041077433423}, 
   "book": {"das": 0.013855912092606825, "buch": 0.9629780468299598, "ein": 0.14077092021669385}, 
   "house": {"das": 0.023166041077433423, "haus": 0.8592290797833062}}
  ```

- `--save-dir`: Directory path to save the result.



## Reference

*Statistical Machine Translation*, Philipp Koehn
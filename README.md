# DUMB chatbot

DUMB(Dumb Undereducated Maladroit Bot) chatbot, a chatbot implemented with PyTorch and trained
with [Cornell Movie Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html).

# TODO:

- [ ] Breakpoint continuous training to achieve intermittent training.
- [x] Migrate to Python3.

## Requirements

- Python 3.9
- PyTorch 1.11.0
- torchaudio 0.11.0
- torchvision 0.12.0
- festival (Linux Environment)
- say (macOS Environment)

## Training Resource

- [Cornell Movie Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)

## How to use

### Data Laundry

``` bash
python3 preprocess.py
```

The script would create `dialogue_corpus.txt` under `./data`.

### Model Training

``` bash
python3 train.py
```

Configs are stored in `config.json`.
Model Training could be time-consuming. I would strongly recommend enabling CUDA in `config.json` to accelerate the
whole training process.

```json
{
  "TRAIN": {
    "DEVICE": "cuda",
    ...
  }
}
```

And if you are using Apple Silicon GPUs, do the following:
```json
{
  "TRAIN": {
    "DEVICE": "mps",
    ...
  }
}
```

### Testing and Running

``` bash
python3 chatbot.py
```

#### Test Samples

``` text
> hi .
bot: hi .
> what is your name ?
bot: vector frankenstein .
> how are you ?
bot: fine .
> where are you from ?
bot: up north .
> are you happy today ?
bot: yes .
```

Though it could answer some questions, it's still dumb enough.

## References

- [seq2seq (Sequence to Sequence) Model for Deep Learning with PyTorch](https://www.guru99.com/seq2seq-model.html)
- [PyTorch documentation](http://pytorch.org/docs/0.1.12/)
- [seq2seq-translation](https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation)
- [tensorflow_chatbot](https://github.com/llSourcell/tensorflow_chatbot)
- [Cornell Movie Dialogs Corpus](https://github.com/suriyadeepan/datasets/tree/master/seq2seq/cornell_movie_corpus)

# Made with ❤ by [Justin Lee](https://github.com/realJustinLee)!

™ and © 1997-2022 Justin Lee. All Rights Reserved. [License Agreement](./LICENSE)

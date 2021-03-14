# DUMB 聊天机器人
DUMB 聊天机器人全称 Dumb Undereducated Maladroit Bot. 是使用 PyTorch 实现并使用康奈尔电影台词库训练的聊天机器人.

# TODO:
- [ ] 断点续训功能, 做到间断性训练.
- [x] Migrate to Python3.

## Requirements
- Python 3.9
- PyTorch 1.8.0
- torchaudio 0.8.8
- torchvision 0.9.0
- festival (Linux Environment)
- say (macOS Environment)

## 训练资源
- [Cornell Movie Dialogue](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)

## 使用方法
### 数据清洗
``` bash
python preprocess.py
```
这个脚本会在`./data`目录创建`dialogue_corpus.txt`.

### 训练模型
``` bash
python train.py
```
调参可以在`config.json`里面进行.
用我自己的电脑(GTX970M)训练的话, 大概需要四个半小时. 使用CPU训练请至少准备一个星期时间.

### 测试和运行
``` bash
python chatbot.py
```

#### 测试样例
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
虽然能回答一些简单的问题, 但还是特别蠢.

## 参考文献
- [seq2seq (Sequence to Sequence) Model for Deep Learning with PyTorch](https://www.guru99.com/seq2seq-model.html)
- [PyTorch documentation](http://pytorch.org/docs/0.1.12/)
- [seq2seq-translation](https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation)
- [tensorflow_chatbot](https://github.com/llSourcell/tensorflow_chatbot)
- [Cornell Movie Dialogs Corpus](https://github.com/suriyadeepan/datasets/tree/master/seq2seq/cornell_movie_corpus)

## Dancing AI

This is my thesis project for Bachelor's Cognitive Science and Artificial Intelligence at Tilburg University.

The paper is available [here](https://songhaban.com/document/Bachelor_CSAI_thesis_SonghaBan_2023907.pdf).

The demo video is available [here](https://youtu.be/UE9QnT59LlI).


The dance generation framework was adapted from the [Pytorch implementation](https://github.com/xrenaa/Music-Dance-Video-Synthesis) of this [Paper](https://arxiv.org/abs/1912.06606)



### Building Dataset:
- Pose estimation: `extract_pose.py`
- Pose evaluation: `pose_eval.py`
- Skeleton visualization: `visframe.py`
- Pose cleaning, normalization: `preprocessing.py`

### Training:
Train the original model
```python
python train.py --gcn --out log/original
```

Train the modified LSTM model
```python
python train.py --encoder musicinitpminlstm --out log/lstm
```

Train the modified GRU model
```python
python train.py --encoder musicinitpmin --out log/gru
```

### Testing:
Give saved model file path and encoder setting as arguments for testing.

Ex) Testing the modified GRU model
```
python inference.py --model log/gru/generator_0350.pth --output output/gru/ --encoder musicinitpmin
```

### Metrics:
The evaluation metrics of dance generation are implemented in `evaluate.py`.


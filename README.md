## Dancing AI

The dance generation framework adapted the [Pytorch implementation](https://github.com/xrenaa/Music-Dance-Video-Synthesis) for this [Paper](https://arxiv.org/abs/1912.06606)

The demo video is shown at: ~~ (TODO)

### Building Dataset:
Pose estimation: extract_pose.py
Pose evaluation: pose_eval.py
Skeleton visualization: visframe.py
Pose cleaning, normalization: preprocessing.py

### Training:
Run
```python
python train.py --encoder musicinitpmin --out log/train1
```

### Testing:
Run
```
python inference.py --model log/train1/generator_0350.pth --output output/ --encoder musicinitp
```

### Metrics:
evaluate.py


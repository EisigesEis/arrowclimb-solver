<!-- Datasets are generated calling `python generate_dataset.py`.

We have generated O1 using:
```python generate_dataset.py --outdir . --num 32``` -->

A single instance file is of the format:
```
M
m_1,s_1
...
N
n_1,p_1
...
```

You may include instances from [the previous implementation](https://github.com/lpi22/uniformSched/tree/main), however not that you will have to execute [instance_rewriter.py](.\instance_rewrite.py) on these instances to convert to the new format before use with our `uniformsched.exe`.
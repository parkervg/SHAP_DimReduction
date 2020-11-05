



### Example Usage

#### Post-Processing Vectors
```python
"""
Following the "Algo-N" process described in https://www.aclweb.org/anthology/W19-4328/
"""
WE = WordEmbeddings(vector_file="embeds/glove.6B.300d.txt")
# PPE
WE.subract_mean()
WE.pca_fit()
WE.remove_top_components(k=7)

# PCA dim reduction
WE.subract_mean()
WE.pca_fit_transform(output_dims=150)

# PPE
WE.subract_mean()
WE.pca_fit()
WE.remove_top_components(k=7)
WE.save_vectors(output_file="embeds/glove_algo150.txt")
```


#### Testing on SentEval Tasks
```shell
python3 run-senteval.py -tests "MPQA" -path "embeds/glove_algo150.txt" -pytorch false
```

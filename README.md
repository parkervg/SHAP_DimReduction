



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



#### Getting top ngrams per task using SHAP dimensions
```shell
python3 -m lib.get_top_shap_grams SUBJ --ngram_size 4
```
Outputs:

```
The label 1 refers to objective statements, and 0 refers to subjective statements

Top ngrams for class 0:

visceral excitement
zhao benshan
cel animation
drug kingpin
vincent d'onofrio
slapstick humor
cgi animation
spousal abuse
visual panache
insider clique
hilarity ensues
drug dealers
slapstick comedy
b-movie imagination
drug overdose

Top ngrams for class 1:

wealthy american
businessman who
african american
officer who
wife who
writer who
director who
american living
marine who
australian director
british officer who
who -
- who
children who
who became
```

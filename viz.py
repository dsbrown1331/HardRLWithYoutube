from train_featurizer import generate_dataset
from embedding_visualization import visualize_embeddings
from TDCFeaturizer import TDCFeaturizer

featurizer = TDCFeaturizer(92, 92, 84, 84)
featurizer.load('montezuma')

d = generate_dataset('montezuma.txt', .25, 84, 84)
features1 = featurizer.featurize(d[0])
features2 = featurizer.featurize(d[1])
features3 = featurizer.featurize(d[2])

features_all = [features1, features2, features3]
visualize_embeddings(features_all, experiment_name='montezuma')

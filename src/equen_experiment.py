from sacred import Experiment
from sacred.observers import FileStorageObserver
import equation_encoder

ex = Experiment("equation_encoder")

ex.observers.append(FileStorageObserver.create('equen_runs'))


@ex.config
def hyperparamters():
    batch_size = 100
    learning_rate = 0.00025
    epochs = 30
    with_dot_product = True
    triples = True
    dataset = 'train'
    architecture = 'small'
    pretrained_weights = None


@ex.capture
def train(batch_size, learning_rate, epochs, with_dot_product, dataset,
          architecture, pretrained_weights, triples):
    #import gitstatus
    #ex.info["gitstatus"] = gitstatus.get_repository_status()
    equation_encoder.train(batch_size=batch_size, learning_rate=learning_rate,
                           epochs=epochs, with_dot_product=with_dot_product,
                           dataset=dataset, architecture=architecture, ex=ex, pretrained_weights=pretrained_weights, triples=triples)


@ex.automain
def main():
    train()

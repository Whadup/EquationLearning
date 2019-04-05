import sys
sys.path.append('libs/similarity_learning')

from sacred import Experiment
from sacred.observers import FileStorageObserver
import pretrain

ex = Experiment("pretrain_equation")

ex.observers.append(FileStorageObserver.create('pretrain_equation_runs'))


@ex.config
def hyperparamters():
    batch_size = 128
    learning_rate = 0.001
    momentum = 0.0
    weight_decay = 0.0
    epochs = 10
    scheduler_patience = 0
    with_dot_product = True
    dataset = 'weak'
    architecture = 'small'
    pretrained_weights = None

@ex.capture
def train(batch_size, learning_rate, momentum, weight_decay, epochs, with_dot_product, dataset,
          architecture, scheduler_patience, pretrained_weights):
    import gitstatus
    ex.info["gitstatus"] = gitstatus.get_repository_status()
    pretrain.train(batch_size=batch_size, learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay,
                  epochs=epochs, scheduler_patience=scheduler_patience, with_dot_product=with_dot_product,
                  dataset=dataset, architecture=architecture, ex=ex, pretrained_weights = pretrained_weights)

@ex.automain
def main():
    train()




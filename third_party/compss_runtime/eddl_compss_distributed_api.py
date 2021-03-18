#from pyeddl import _core

from pycompss.api.api import compss_wait_on

from eddl_array import paired_partition
from eddl_compss_distributed import *
from net_utils import net_aggregateParameters
from net_utils import net_aggregateResults
from net_utils import net_parametersToNumpy
from shuffle import local_shuffle, block_shuffle_async

compss_object: Eddl_Compss_Distributed = None


def build(model, optimizer, losses, metrics, compserv, random_weights):
    # Initialize the compss object
    global compss_object
    compss_object = Eddl_Compss_Distributed()

    #print("Type3333", type(optimizer))
    #print("Type2222: ", isinstance(optimizer, _core.sgd))

    # This call to build is for security reasons to initialize the parameters to random values
    eddl.build(
        model,
        optimizer,
        losses,
        metrics,
        compserv,
        random_weights
    )

    # Serialize the model so it can be sent through the network
    serialized_model = eddl.serialize_net_to_onnx_string(model, False)

    # Build the model in all the nodes synchronously
    compss_object.build(serialized_model, "", losses, metrics, "")


def train_batch(model, x_train, y_train, num_workers, num_epochs_for_param_sync, workers_batch_size):
    global compss_object

    # Initial parameters that every node will take in order to begin the training
    initial_parameters = net_parametersToNumpy(eddl.get_parameters(model, False, True))
    # print("Aqui si se muestran: ", initial_parameters)

    recv_weights = [list() for i in range(0, num_workers)]

    s = x_train.shape
    num_images_per_worker = int(s[0] / num_workers)

    print("Num workers: ", num_workers)
    print("Num images per worker: ", num_images_per_worker)
    print("Workers batch size: ", workers_batch_size)

    # Pickle test

    # import pickle
    # filename = "dogs"
    # outfile = open(filename,'wb')
    # pickle.dump(initial_parameters, outfile)

    # print("Pasa el test de pickle")
    # infile = open(filename,'rb')
    # new_dict = pickle.load(infile)
    # infile.close()

    # print("Pasa el test de deserialization pickle")
    #
    # import codecs
    # initial_parameters = codecs.encode(pickle.dumps(initial_parameters), "base64").decode()
    for i, (block_x, block_y) in enumerate(paired_partition(x_train, y_train)):
        recv_weights[i] = compss_object.train_batch(
            block_x,
            block_y,
            initial_parameters,
            num_images_per_worker,
            num_epochs_for_param_sync,
            workers_batch_size)

    # COMPSS barrier to force waiting until every node finishes its training (synchronous training)
    recv_weights = compss_wait_on(recv_weights)

    # Parameters aggregation
    final_weights = net_aggregateParameters(recv_weights)

    # Print final weights
    # print("Los final weights al final de train_batch son estos: ", final_weights)

    # Set trained and aggregated parameters to the model
    eddl.set_parameters(model, net_parametersToTensor(final_weights))


def train_batch_async(model, x_train, y_train, num_workers, num_epochs_for_param_sync, workers_batch_size):
    global compss_object

    s = x_train.shape
    num_images_per_worker = int(s[0] / num_workers)

    print("Num workers: ", num_workers)
    print("Num images per worker: ", num_images_per_worker)
    print("Workers batch size: ", workers_batch_size)

    # Array of final parameters whose initial value is initial parameters
    #accumulated_parameters = net_parametersToNumpy(model.getParameters())
    accumulated_parameters = net_parametersToNumpy(eddl.get_parameters(model, False, True))

    # accumulated_parameters = net_numpyParametersFill(accumulated_parameters, 0)

    #workers_parameters = [net_parametersToNumpy(model.getParameters()) for i in range(0, num_workers)]
    workers_parameters = [net_parametersToNumpy(eddl.get_parameters(model, False, True)) for i in range(0, num_workers)]

    for j, (block_x, block_y) in enumerate(paired_partition(x_train, y_train)):
        workers_parameters[j] = compss_object.train_batch(
            block_x,
            block_y,
            workers_parameters[j],
            num_images_per_worker,
            num_epochs_for_param_sync,
            workers_batch_size)
        workers_parameters[j] = compss_object.aggregate_parameters_async(
            accumulated_parameters,
            workers_parameters[j],
            1 / num_workers)

    accumulated_parameters = compss_wait_on(accumulated_parameters)
    # workers_parameters = compss_wait_on(workers_parameters)

    print("Workers parameters: ", workers_parameters)
    print("Final accumulated parameters: ", accumulated_parameters)

    # Set trained and aggregated parameters to the model
    #model.setParameters(net_parametersToTensor(accumulated_parameters))
    eddl.set_parameters(model, net_parametersToTensor(accumulated_parameters))


def fit_async(model, x_train, y_train, num_workers, num_epochs_for_param_sync, max_num_async_epochs,
              workers_batch_size):
    global compss_object

    s = x_train.shape
    num_images_per_worker = int(s[0] / num_workers)

    print("Num workers: ", num_workers)
    print("Num images per worker: ", num_images_per_worker)
    print("Workers batch size: ", workers_batch_size)

    # Array of final parameters whose initial value is initial parameters
    #accumulated_parameters = net_parametersToNumpy(model.getParameters())
    # accumulated_parameters = net_numpyParametersFill(accumulated_parameters, 0)
    accumulated_parameters = net_parametersToNumpy(eddl.get_parameters(model, False, True))

    #workers_parameters = [net_parametersToNumpy(model.getParameters()) for i in range(0, num_workers)]
    workers_parameters = [net_parametersToNumpy(eddl.get_parameters(model, False, True)) for i in range(0, num_workers)]
    x_blocks = [x[0] for x in paired_partition(x_train, y_train)]
    y_blocks = [x[1] for x in paired_partition(x_train, y_train)]

    for i in range(0, max_num_async_epochs):
        for j in range(0, num_workers):
            shuffled_x, shuffled_y = block_shuffle_async(
                x_blocks[j],
                y_blocks[j],
                workers_parameters[j])
            x_blocks[j], y_blocks[j] = [shuffled_x], [shuffled_y]
            workers_parameters[j] = compss_object.train_batch(
                x_blocks[j],
                y_blocks[j],
                workers_parameters[j],
                num_images_per_worker,
                num_epochs_for_param_sync,
                workers_batch_size)
            workers_parameters[j] = compss_object.aggregate_parameters_async(
                accumulated_parameters,
                workers_parameters[j],
                1 / num_workers)

    accumulated_parameters = compss_wait_on(accumulated_parameters)
    # workers_parameters = compss_wait_on(workers_parameters)

    # print("Workers parameters: ", workers_parameters)
    # print("Final accumulated parameters: ", accumulated_parameters)

    # Set trained and aggregated parameters to the model
    #model.setParameters(net_parametersToTensor(accumulated_parameters))
    eddl.set_parameters(model, net_parametersToTensor(accumulated_parameters))



def evaluate(model, train_test_flag):
    global compss_object

    returns = [0]

    # Initial parameters that every node will take in order to begin the training
    initial_parameters = net_parametersToNumpy(model.getParameters())

    returns[0] = compss_object.evaluate(initial_parameters, train_test_flag)

    returns = compss_wait_on(returns)


def eval_batch(model, x_test, y_test, num_workers, workers_batch_size):

    global compss_object

    # Initial parameters that every node will take in order to begin the training
    initial_parameters = net_parametersToNumpy(eddl.get_parameters(model, False, True))
    #print("Aqui si se muestran: ", initial_parameters)

    recv_results = [list() for i in range(0, num_workers)]

    s = x_test.shape
    num_images_per_worker = int(s[0] / num_workers)

    print("Num workers: ", num_workers)
    print("Num images per worker: ", num_images_per_worker)
    print("Workers batch size: ", workers_batch_size)

    for i, (block_x, block_y) in enumerate(paired_partition(x_test, y_test)):
        recv_results[i] = compss_object.eval_batch(
            block_x,
            block_y,
            initial_parameters,
            num_images_per_worker,
            workers_batch_size)

    # COMPSS barrier to force waiting until every node finishes its training (synchronous training)
    recv_results = compss_wait_on(recv_results)

    # Parameters aggregation
    final_results = net_aggregateResults(recv_results)

    return final_results

    # Print final weights
    #print("Los final weights al final de train_batch son estos: ", final_weights)

    # Set trained and aggregated parameters to the model
    #model.setParameters(net_parametersToTensor(final_weights))
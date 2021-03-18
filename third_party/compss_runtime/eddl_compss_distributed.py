import numpy as np
import pyeddl.eddl as eddl
from pycompss.api.constraint import constraint
from pycompss.api.parameter import *
from pycompss.api.task import task
from pyeddl.tensor import Tensor as eddlT

from cvars import *
from eddl_array import to_tensor
from net_utils import net_parametersToNumpy
from net_utils import net_parametersToTensor


class Eddl_Compss_Distributed:

    def __init__(self):
        self.model = None

    @constraint(computing_units="${OMP_NUM_THREADS}")
    @task(serialized_model=IN, optimizer=IN, losses=IN, metrics=IN, compserv=IN, is_replicated=True)
    def build(self, serialized_model, optimizer, losses, metrics, compserv):

        print("Arranca build task en worker")

        # Deserialize the received model
        model = eddl.import_net_from_onnx_string(serialized_model)

        # print(eddl.summary(model))

        # Build the model in this very node
        eddl.build(
            model,
            eddl.sgd(CVAR_SGD1, CVAR_SGD2),
            losses,
            metrics,
            eddl.CS_CPU(mem="full_mem"),
            False
        )

        # Save the model. We have to serialize it to a string so COMPSs is able to serialize and deserialize from disk
        self.model = eddl.serialize_net_to_onnx_string(model, False)

        print("Finaliza build task en worker")

    @constraint(computing_units="${OMP_NUM_THREADS}")
    @task(
        x_train={Type: COLLECTION_IN, Depth: 2},
        y_train={Type: COLLECTION_IN, Depth: 2},
        initial_parameters=IN,
        num_images_per_worker=IN,
        num_epochs_for_param_sync=IN,
        workers_batch_size=IN,
        target_direction=IN)
    def train_batch(self,
                    x_train,
                    y_train,
                    initial_parameters,
                    num_images_per_worker,
                    num_epochs_for_param_sync,
                    workers_batch_size):

        # Convert data to tensors
        x_train = to_tensor(x_train)
        y_train = to_tensor(y_train)

        print("Entrando en train batch task en worker")
        import sys
        sys.stdout.flush()

        # Deserialize from disk
        model = eddl.import_net_from_onnx_string(self.model)

        print("Modelo deserializado de disco")
        sys.stdout.flush()

        # The model needs to be built after deserializing and before injecting the parameters
        eddl.build(
            model,
            eddl.sgd(CVAR_SGD1, CVAR_SGD2),
            ["soft_cross_entropy"],
            ["categorical_accuracy"],
            eddl.CS_CPU(mem="full_mem"),
            False
        )

        print("Modelo built")
        sys.stdout.flush()

        # Set the parameters sent from master to the model
        #model.setParameters(net_parametersToTensor(initial_parameters))
        eddl.set_parameters(model, net_parametersToTensor(initial_parameters))

        print("Parametros seteados")
        sys.stdout.flush()

        # print(eddl.summary(model))

        print("Build completed in train batch task")
        sys.stdout.flush()
        num_batches = int(num_images_per_worker / workers_batch_size)

        eddl.reset_loss(model)
        print("Num batches: ", num_batches)
        sys.stdout.flush()

        for i in range(num_epochs_for_param_sync):

            print("Epoch %d/%d (%d batches)" % (i + 1, num_epochs_for_param_sync, num_batches))

            for j in range(num_batches):
                indices = list(range(j * num_batches, (j + 1) * num_batches - 1))
                print("Empieza batch")
                sys.stdout.flush()
                eddl.train_batch(model, [x_train], [y_train], indices)
                print("Finaliza batch")
                sys.stdout.flush()
                eddl.print_loss(model, j)
                print()

        print("Train batch individual completed in train batch task")
        sys.stdout.flush()

        # Random noise time sleep for experimentation purposes
        # import time
        # import random
        # time.sleep(random.randint(1, 180))

        # Get parameters from the model and convert them to numpy so COMPSS can serialize them
        final_parameters = net_parametersToNumpy(eddl.get_parameters(model, False, True))

        return final_parameters

    @task(
        x_train={Type: COLLECTION_IN, Depth: 2},
        y_train={Type: COLLECTION_IN, Depth: 2},
        initial_parameters=IN,
        num_images_per_worker=IN,
        num_epochs_for_param_sync=IN,
        workers_batch_size=IN,
        target_direction=IN)
    def train_batch_async(self,
                          x_train,
                          y_train,
                          initial_parameters,
                          num_images_per_worker,
                          num_epochs_for_param_sync,
                          workers_batch_size):

        # Deserialize from disk
        model = eddl.import_net_from_onnx_string(self.model)

        # Set the parameters sent from master to the model
        #model.setParameters(net_parametersToTensor(initial_parameters))
        eddl.set_parameters(model, net_parametersToTensor(initial_parameters))

        # print(eddl.summary(model))

        # The model needs to be built after deserializing
        eddl.build(
            model,
            eddl.sgd(CVAR_SGD1, CVAR_SGD2),
            ["soft_cross_entropy"],
            ["categorical_accuracy"],
            eddl.CS_CPU(mem="full_mem"),
            False
        )

        print("Build completed in train batch task")

        num_batches = int(num_images_per_worker / workers_batch_size)

        eddl.reset_loss(model)
        print("Num batches: ", num_batches)

        for i in range(num_epochs_for_param_sync):

            print("Epoch %d/%d (%d batches)" % (i + 1, num_epochs_for_param_sync, num_batches))

            for j in range(num_batches):
                indices = list(range(j * num_batches, (j + 1) * num_batches - 1))
                eddl.train_batch(model, [x_train], [y_train], indices)
                eddl.print_loss(model, j)
                print()

        print("Train batch individual completed in train batch task")

        # Get parameters from the model and convert them to numpy so COMPSS can serialize them
        #final_parameters = net_parametersToNumpy(model.getParameters())
        final_parameters = net_parametersToNumpy(eddl.get_parameters(model, False, True))

        return final_parameters

    @constraint(computing_units="${OMP_NUM_THREADS}")
    @task(accumulated_parameters=COMMUTATIVE, parameters_to_aggregate=IN, mult_factor=IN, target_direction=IN)
    def aggregate_parameters_async(self, accumulated_parameters, parameters_to_aggregate, mult_factor):

        for i in range(0, len(accumulated_parameters)):
            for j in range(0, len(accumulated_parameters[i])):
                # accumulated_parameters[i][j] += (parameters_to_aggregate[i][j] * mult_factor).astype(np.float32)
                accumulated_parameters[i][j] = (
                            (accumulated_parameters[i][j] + parameters_to_aggregate[i][j]) / 2).astype(np.float32)

        return accumulated_parameters

    @constraint(computing_units="${OMP_NUM_THREADS}")
    @task(initial_parameters=IN, train_test_flag=IN, target_direction=IN)
    def evaluate(self, initial_parameters, train_test_flag):

        # Deserialize from disk
        model = eddl.import_net_from_onnx_string(self.model)

        # Set the parameters sent from master to the model
        model.setParameters(net_parametersToTensor(initial_parameters))

        # print(eddl.summary(model))

        # The model needs to be built after deserializing
        eddl.build(
            model,
            eddl.sgd(CVAR_SGD1, CVAR_SGD2),
            ["soft_cross_entropy"],
            ["categorical_accuracy"],
            eddl.CS_CPU(mem="full_mem"),
            False
        )

        print("Build completed in evaluate task")

        if train_test_flag == "train":

            x = eddlT.load(CVAR_DATASET_PATH + CVAR_DATASET_X_TRN)
            y = eddlT.load(CVAR_DATASET_PATH + CVAR_DATASET_Y_TRN)

        else:

            x = eddlT.load(CVAR_DATASET_PATH + CVAR_DATASET_X_TST)
            y = eddlT.load(CVAR_DATASET_PATH + CVAR_DATASET_Y_TST)

        eddlT.div_(x, 255.0)

        print("Evaluating model against " + train_test_flag + " set")
        eddl.evaluate(model, [x], [y])

        return 1


    @constraint(computing_units="${OMP_NUM_THREADS}")
    @task(initial_parameters=IN, x_test={Type: COLLECTION_IN, Depth: 2}, y_test={Type: COLLECTION_IN, Depth: 2}, num_images_per_worker=IN, workers_batch_size=IN, target_direction=IN)
    def eval_batch(self, x_test, y_test, initial_parameters, num_images_per_worker, workers_batch_size):

        #import pickle
        #import codecs

        #initial_parameters = pickle.loads(codecs.decode(initial_parameters.encode(), "base64"))

        print("Entrando en eval batch task en worker")
        import sys
        sys.stdout.flush()

        # Deserialize from disk
        model = eddl.import_net_from_onnx_string(self.model)

        print("Modelo deserializado de disco")
        sys.stdout.flush()

        # The model needs to be built after deserializing and before injecting the parameters
        eddl.build(
            model,
            eddl.sgd(CVAR_SGD1, CVAR_SGD2),
            ["soft_cross_entropy"],
            ["categorical_accuracy"],
            eddl.CS_CPU(mem="full_mem"),
            False
        )

        print("Modelo built")
        sys.stdout.flush()

        # Set the parameters sent from master to the model
        eddl.set_parameters(model, net_parametersToTensor(initial_parameters))
        
        print("Parametros seteados")
        sys.stdout.flush()

        #print(eddl.summary(model))

        print("Build completed in eval batch task")

        sys.stdout.flush()

        # Convert data to tensors
        x_test = to_tensor(x_test)
        y_test = to_tensor(y_test)

        print("Lenes: ", x_test.shape)
        sys.stdout.flush()

        num_batches = int(num_images_per_worker / workers_batch_size)

        eddl.reset_loss(model)
        print("Num batches: ", num_batches)
        sys.stdout.flush()

        '''for i in range(num_batches):
            indices = list(range(0, x_test.shape[0]))
            eddl.eval_batch(model, [x_test], [y_test], indices)'''

        eddl.evaluate(model, [x_test], [y_test])

        print("Eval batch individual completed in eval batch task")
        sys.stdout.flush()

        # Random noise time sleep for experimentation purposes
        #import time
        #import random
        #time.sleep(random.randint(1, 180))

        # Get parameters from the model and convert them to numpy so COMPSS can serialize them
        #final_parameters = net_parametersToNumpy(model.getParameters())
        
        # These final results should be a call to getAcc() y getLoss()
        final_results = [eddl.get_losses(model)[-1], eddl.get_metrics(model)[-1]]

        print("Final results: ", final_results)
        sys.stdout.flush()

        return final_results

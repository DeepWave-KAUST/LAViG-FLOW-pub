from .data import Data
from .sampler import BatchSampler


class Quadruple(Data):
    """Dataset with each data point as a quadruple.

    The couple of the first three elements are the input, and the fourth element is the
    output. This dataset can be used with the network ``MIONet`` for operator
    learning.

    Args:
        X_train: A tuple of three NumPy arrays.
        y_train: A NumPy array.
    """

    def __init__(self, X_train, y_train, X_test, y_test):
        self.train_x = X_train
        self.train_y = y_train
        self.test_x = X_test
        self.test_y = y_test

        self.train_sampler = BatchSampler(len(self.train_y), shuffle=True)

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        indices = self.train_sampler.get_next(batch_size)
        return (
            (self.train_x[0][indices], self.train_x[1][indices]),
            self.train_x[2][indices],
            self.train_y[indices],
        )

    def test(self):
        return self.test_x, self.test_y


class QuadrupleCartesianProd(Data):
    """Cartesian Product input data format for MIONet architecture.

    This dataset can be used with the network ``MIONetCartesianProd`` for operator
    learning.

    Args:
        X_train: A tuple of three NumPy arrays. The first element has the shape (`N1`,
            `dim1`), the second element has the shape (`N1`, `dim2`), and the third
            element has the shape (`N2`, `dim3`).
        y_train: A NumPy array of shape (`N1`, `N2`).
    """

    def __init__(self, X_train, y_train, X_test, y_test):
        # if (
        #     len(X_train[0]) * len(X_train[2]) != y_train.size
        #     or len(X_train[1]) * len(X_train[2]) != y_train.size
        #     or len(X_train[0]) != len(X_train[1])
        # ):
        #     raise ValueError(
        #         "The training dataset does not have the format of Cartesian product."
        #     )
        # if (
        #     len(X_test[0]) * len(X_test[2]) != y_test.size
        #     or len(X_test[1]) * len(X_test[2]) != y_test.size
        #     or len(X_test[0]) != len(X_test[1])
        # ):
        #     raise ValueError(
        #         "The testing dataset does not have the format of Cartesian product."
        #     )
        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test

        self.train_sampler = BatchSampler(len(X_train[0]), shuffle=True)
        self.train_timestep_sampler = None

    def losses(self, targets, outputs,indices,istrain,loss_fn, inputs, model, aux=None):
        return loss_fn(targets, outputs, indices,istrain)

    def train_next_batch(self, batch_size=None, timestep_batch_size=None, training_time_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        if training_time_size is None:
            raise ValueError("training_time_size must be provided for timestep sampling.")
        if (
            self.train_timestep_sampler is None
            or self.train_timestep_sampler.num_samples != training_time_size
        ):
            self.train_timestep_sampler = BatchSampler(training_time_size, shuffle=True)
        indices = self.train_sampler.get_next(batch_size)
        indices_timestep = self.train_timestep_sampler.get_next(timestep_batch_size)
        #indices_timestep = [23]
        x_loc = self.train_x[2]
        if x_loc.ndim == 1:
            x_loc = x_loc[:, None]
        if x_loc.ndim != 2:
            raise ValueError(
                "Expected train_x[2] to have shape (N, dim). "
                f"Got shape {x_loc.shape}."
            )
        expected_grid = training_time_size * 96 * 200
        size = self.train_y.shape[0]
        if x_loc.shape[0] == training_time_size:
            x_loc_batch = x_loc[indices_timestep]
        elif x_loc.shape[0] == expected_grid:
            x_loc_batch = x_loc.reshape(training_time_size, 96 * 200, 3)[
                indices_timestep, :
            ].reshape(timestep_batch_size * 96 * 200, 3)
        else:
            raise ValueError(
                "training_time_size does not match train_x[2]. "
                f"training_time_size={training_time_size}, train_x[2].shape[0]={x_loc.shape[0]}."
            )
        return (
            self.train_x[0][indices],
            self.train_x[1][indices],
            x_loc_batch,
        ), self.train_y.reshape(size, training_time_size, 96 * 200)[indices, :][:, indices_timestep].reshape(
            batch_size, timestep_batch_size * 96 * 200
        ), indices

    def test(self):
        return self.test_x, self.test_y

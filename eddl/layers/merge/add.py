from eddl.layers.base import Layer


class Add(Layer):
    """Layer that adds a list of inputs.

    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).

    Example:
        ```python
            import eddl
            input1 = eddl.layers.Input(shape=(16,))
            x1 = eddl.layers.Dense(8, activation='relu')(input1)
            input2 = eddl.layers.Input(shape=(32,))
            x2 = eddl.layers.Dense(8, activation='relu')(input2)
            # equivalent to added = eddl.layers.add([x1, x2])
            added = eddl.layers.Add()([x1, x2])
            out = eddl.layers.Dense(4)(added)
            model = eddl.models.Model(inputs=[input1, input2], outputs=out)
        ```

    """
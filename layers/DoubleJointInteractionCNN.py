import collections
import tensorflow as tf

from tensorflow.contrib.rnn import LayerRNNCell, RNNCell
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
from tensorflow.python.keras import initializers

_DoubleStateTuple = collections.namedtuple("DoubleStateTuple", ("s1", "s2",
                                                                "m1", "m2",
                                                                "r_h"))


def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


class DoubleStateTuple(_DoubleStateTuple):
    __slots__ = ()

    @property
    def dtype(self):
        return self[0].dtype


class DoubleJointCNNCell(LayerRNNCell):
    def __init__(self,
                 num_units,
                 r_cell: RNNCell,
                 sent1,
                 sent2,
                 dim,
                 sent1_length=None,
                 sent2_length=None,
                 sent1_mask=None,
                 sent2_mask=None,
                 initializer=None,
                 activation=tf.nn.relu,
                 use_bias=False,
                 dropout_rate=0.0,
                 k=8,
                 filter_sizes=[1, 2, 3, 4],
                 reuse=None, name="Double", dtype=None, **kwargs):
        super(DoubleJointCNNCell, self).__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)

        self.num_units = num_units

        self.r_cell = r_cell
        # self.sent_transformer = sentence_transformer

        self.sent1 = sent1
        self.sent2 = sent2
        if sent1_length is None:
            sent1_length = self.sent1.get_shape().as_list()[1]
            sent2_length = self.sent2.get_shape().as_list()[1]
        self.sent1_length = sent1_length
        self.sent2_length = sent2_length
        self.sent1_mask = sent1_mask
        self.sent2_mask = sent2_mask

        self.dim = dim
        self.initializer = initializers.get(initializer)
        self.activation = activation  # \phi
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.k = k
        self.filter_sizes = filter_sizes
        if dim % len(filter_sizes) != 0:
            raise ValueError("filter num is not integer")
        self.filter_num = dim // len(filter_sizes)

        self.filters = []
        self.bs = []
        # for filter_size in self.filter_sizes:
        #     with tf.variable_scope("conv{}".format(filter_size)):
        #         filter_shape = tf.convert_to_tensor([filter_size, filter_size, self.dim, self.filter_num])
        #         fil = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='filter')
        #         if self.use_bias:
        #             b = tf.Variable(tf.constant(0.1, shape=[self.filter_num]))
        #         else:
        #             b = None
        #         self.bs.append(b)
        #         self.filters.append(fil)

        # general conv
        for filter_size in [4]:
            with tf.variable_scope("conv{}".format(filter_size)):
                filter_shape = tf.convert_to_tensor([filter_size, filter_size, self.dim, self.dim])
                fil = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='filter')
                if self.use_bias:
                    b = tf.Variable(tf.constant(0.1, shape=[self.dim]))
                else:
                    b = None
                self.bs.append(b)

                self.filters.append(fil)

        self.initializer = initializer

        self._state_size = DoubleStateTuple(self.sent1_length * self.dim,  # sentences
                                            self.sent2_length * self.dim,

                                            self.sent1_length,
                                            self.sent2_length,

                                            self.num_units   # relation hidden states
                                            )

        self._output_size = self.num_units

    def zero_state(self, batch_size, dtype):
        """Initialize the memory to the key values."""

        with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            sent1 = tf.reshape(self.sent1, [-1, self.sent1_length * self.dim])
            sent2 = tf.reshape(self.sent2, [-1, self.sent2_length * self.dim])
            sent1_mask = self.sent1_mask
            sent2_mask = self.sent2_mask

            rh = _zero_state_tensors([self.num_units], batch_size, dtype=tf.float32)

            state_list = [sent1, sent2, sent1_mask, sent2_mask, rh[0]]

            return DoubleStateTuple(*state_list)

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def build(self, inputs_shape):

        bias_init = 1.0  # or 0.0
        # parameters which relation updating needs
        self.matrix_kernel = self.add_variable(name="matrix_kernel",
                                               shape=[3 * self.dim, self.num_units],
                                               initializer=self.initializer)

        self.phrase_kernel = self.add_variable(name="phrase_kernel",
                                             shape=[self.dim + self.num_units, 1],
                                             initializer=self.initializer)

        self.filter_dense = self.add_variable(name="filter_dense",
                                              shape=[self.dim, self.dim],
                                              initializer=self.initializer)

        # all bias
        if self.use_bias:
            self.matrix_bias = self.add_variable(name="matrix_bias",
                                                 shape=[self.num_units],
                                                 initializer=tf.constant_initializer(bias_init, dtype=tf.float32))

            self.phrase_bias = self.add_variable(name="phrase_bias",
                                               shape=[1],
                                               initializer=tf.constant_initializer(bias_init, dtype=tf.float32))

            self.filter_dense_bias = self.add_variable(name="filter_dense_bias",
                                                       shape=[self.dim],
                                                       initializer=tf.constant_initializer(bias_init, dtype=tf.float32))

        self.built = True

    def call(self, inputs, state=None):
        s1_tm, s2_tm, s1_mask, s2_mask, rh_tm = state
        # s: (B, Lx * dim), values: (B, keys_num * dim),
        # r_h: (B, dim)
        k = self.k
        s1_tm = tf.reshape(s1_tm, [-1, self.sent1_length, self.dim])                       # (B, L1, dim)
        s2_tm = tf.reshape(s2_tm, [-1, self.sent2_length, self.dim])                       # (B, L2, dim)
        s1_mask = tf.expand_dims(s1_mask, axis=2)
        s2_mask = tf.expand_dims(s2_mask, axis=2)

        s1_score, s1_mask = self.get_phrase(s1_tm, self.sent1_length, s1_mask, rh_tm)  # (B, L1, 1)
        s2_score, s2_mask = self.get_phrase(s2_tm, self.sent2_length, s2_mask, rh_tm)  # (B, L2, 1)

        # selecting k-max
        s1_kmax_values, s1_kmax_index = tf.nn.top_k(tf.squeeze(s1_score, axis=2), k=k)
        s2_kmax_values, s2_kmax_index = tf.nn.top_k(tf.squeeze(s2_score, axis=2), k=k)

        s1_kmax_values = s1_kmax_values / tf.reduce_sum(s1_kmax_values, axis=1, keepdims=True)
        s2_kmax_values = s2_kmax_values / tf.reduce_sum(s2_kmax_values, axis=1, keepdims=True)

        s1_kmax = tf.batch_gather(s1_tm, s1_kmax_index)
        s2_kmax = tf.batch_gather(s2_tm, s2_kmax_index)

        score_matrix_kmax = tf.keras.backend.batch_dot(tf.expand_dims(s1_kmax_values, axis=2),
                                                       tf.expand_dims(s2_kmax_values, axis=2),
                                                       [2, 2])  # (B, L1, L2)

        threshold = 0.08
        condition = tf.less_equal(score_matrix_kmax, threshold)
        zero_tensor = tf.zeros_like(score_matrix_kmax)
        score_matrix_kmax = tf.keras.backend.switch(condition, zero_tensor, score_matrix_kmax)

        vec_matrix_kmax = self.get_vec_matrix(s1_kmax, k, s2_kmax, k)
        score_matrix_kmax = tf.expand_dims(score_matrix_kmax, axis=3)
        phrase_vec_kmax = self.get_cnn_feature(score_matrix_kmax * vec_matrix_kmax)

        rh, _ = self.r_cell(phrase_vec_kmax, rh_tm)

        s1_tm = tf.reshape(s1_tm, [-1, self.sent1_length * self.dim])                       # (B, L1, dim)
        s2_tm = tf.reshape(s2_tm, [-1, self.sent2_length * self.dim])                       # (B, L2, dim)
        s1_mask = tf.squeeze(s1_mask, axis=2)
        s2_mask = tf.squeeze(s2_mask, axis=2)

        # # compute mask
        # mask_temp = 1.0 - threshold_score
        # condition = tf.less_equal(mask_temp, 0.98)
        # zero_tensor = tf.zeros_like(mask_temp)
        # mask_temp = tf.keras.backend.switch(condition, zero_tensor, mask_temp)
        # s_mask = s_mask * mask_temp

        state = [s1_tm, s2_tm, s1_mask, s2_mask, rh]

        return rh, DoubleStateTuple(*state)

    def get_phrase(self, s_tm, sent_length, s_mask, rh_tm):
        # s_tm = tf.reshape(s_tm, [-1, sent_length, self.dim])

        rh_tm_exp = tf.tile(tf.expand_dims(rh_tm, axis=1), [1, sent_length, 1])
        concat_temp = tf.concat([s_tm, rh_tm_exp], axis=2)

        concat_temp = dropout(concat_temp, self.dropout_rate)
        res_phrase = tf.keras.backend.dot(concat_temp, self.phrase_kernel)

        if self.use_bias:
            res_phrase = tf.nn.bias_add(res_phrase, self.phrase_bias)

        res_phrase = self.activation(res_phrase)  # (B, L, 1)

        res_phrase -= (1 - s_mask) * 1000.0

        # # <1>
        threshold_score = tf.nn.softmax(res_phrase, axis=1)

        return threshold_score, s_mask

    def get_vec_matrix(self,
                        s1, s1_len,
                        s2, s2_len):
        # values_tm: (B, keys_num * dim),
        # s1, s2: (B, Lx, dim);  s1_len, s2_len: scalar;  s1_mask, s2_mask: (B, Lx) or None
        # rh_tm: (B, dim)

        # <1>
        s1_exp = tf.expand_dims(s1, axis=2)
        s2_exp = tf.expand_dims(s2, axis=1)

        s1_tile = tf.tile(s1_exp, [1, 1, s2_len, 1])
        s2_tile = tf.tile(s2_exp, [1, s1_len, 1, 1])

        infor_cat = tf.concat([s1_tile, s2_tile, s1_tile * s2_tile], axis=-1)   # (B, L1, L2, 1*dim)
        infor_cat = dropout(infor_cat, self.dropout_rate)
        res_matrix = tf.keras.backend.dot(infor_cat, self.matrix_kernel)

        if self.use_bias:
            res_matrix = tf.nn.bias_add(res_matrix, self.matrix_bias)

        res_matrix = self.activation(res_matrix)

        return res_matrix

    def get_cnn_feature(self, img):  # (B, k, k, dim)
        pooled_outputs = []
        # for filter_size, fil, b in zip(self.filter_sizes, self.filters, self.bs):
        #     with tf.variable_scope("conv{}".format(filter_size)):
        #         conv = tf.nn.conv2d(img, fil, [1] * 4, 'VALID', name='conv')
        #
        #         if self.use_bias:
        #             conv = tf.nn.bias_add(conv, b)
        #         h = tf.nn.relu(conv)
        #
        #         pooled = tf.reduce_max(tf.reduce_max(h, 1), 1)
        #         pooled_outputs.append(pooled)
        #
        # output = tf.concat(pooled_outputs, axis=1)

        # general conv
        for filter_size, fil, b in zip([4], self.filters, self.bs):
            with tf.variable_scope("conv{}".format(filter_size)):
                conv = tf.nn.conv2d(img, fil, [1] * 4, 'VALID', name='conv')

                if self.use_bias:
                    conv = tf.nn.bias_add(conv, b)
                h = tf.nn.tanh(conv)

                pooled = tf.nn.max_pool(h, [1, 5, 5, 1], [1, 1, 1, 1], 'VALID')
                pooled_outputs.append(pooled)

        output1 = tf.reshape(pooled_outputs[0], [-1, self.dim])
        output = dropout(output1, self.dropout_rate)
        output = tf.keras.backend.dot(output, self.filter_dense)
        if self.use_bias:
            output = tf.nn.bias_add(output, self.filter_dense_bias)
        output = self.activation(output)

        return output + output1


def dropout(input_tensor, dropout_prob):
    """Perform dropout.

  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).

  Returns:
    A version of `input_tensor` with dropout applied.
  """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output

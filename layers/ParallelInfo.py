import math
import six
import tensorflow as tf

from layers.BiGRU import NativeGRU, NativeLSTM, CudnnGRU, CudnnLSTM


def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def assert_rank(tensor, expected_rank, name=None):
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def dropout(input_tensor, dropout_prob):
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


def gelu(input_tensor):
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf


def get_shape_list(tensor, expected_rank=None, name=None):
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def reshape_to_matrix(input_tensor):
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                         (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
    """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
    if len(orig_shape_list) == 2:
        return output_tensor

    output_shape = get_shape_list(output_tensor)

    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]

    return tf.reshape(output_tensor, orig_dims + [width])


def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):

    def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                             seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

    if len(from_shape) != len(to_shape):
        raise ValueError(
            "The rank of `from_tensor` must match the rank of `to_tensor`.")

    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
        if batch_size is None or from_seq_length is None or to_seq_length is None:
            raise ValueError(
                "When passing in rank 2 tensors to attention_layer, the values "
                "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                "must all be specified.")

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`

    from_tensor_2d = reshape_to_matrix(from_tensor)
    to_tensor_2d = reshape_to_matrix(to_tensor)

    # `query_layer` = [B*F, N*H]
    query_layer = tf.layers.dense(
        from_tensor_2d,
        num_attention_heads * size_per_head,
        activation=query_act,
        name="query",
        kernel_initializer=create_initializer(initializer_range))

    # `key_layer` = [B*T, N*H]
    key_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=key_act,
        name="key",
        kernel_initializer=create_initializer(initializer_range))

    # `value_layer` = [B*T, N*H]
    value_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=value_act,
        name="value",
        kernel_initializer=create_initializer(initializer_range))

    # `query_layer` = [B, N, F, H]
    query_layer = transpose_for_scores(query_layer, batch_size,
                                       num_attention_heads, from_seq_length,
                                       size_per_head)

    # `key_layer` = [B, N, T, H]
    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                     to_seq_length, size_per_head)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    # `attention_scores` = [B, N, F, T]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(size_per_head)))

    if attention_mask is not None:
        # `attention_mask` = [B, 1, F, T]
        attention_mask = tf.expand_dims(attention_mask, axis=[1])

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_scores += adder

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_probs = tf.nn.softmax(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

    # `value_layer` = [B, T, N, H]
    value_layer = tf.reshape(
        value_layer,
        [batch_size, to_seq_length, num_attention_heads, size_per_head])

    # `value_layer` = [B, N, T, H]
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

    # `context_layer` = [B, N, F, H]
    context_layer = tf.matmul(attention_probs, value_layer)

    # `context_layer` = [B, F, N, H]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

    if do_return_2d_tensor:
        # `context_layer` = [B*F, N*V]
        context_layer = tf.reshape(
            context_layer,
            [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
        # `context_layer` = [B, F, N*V]
        context_layer = tf.reshape(
            context_layer,
            [batch_size, from_seq_length, num_attention_heads * size_per_head])

    return context_layer


def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "The hidden size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (hidden_size, num_attention_heads))

    attention_head_size = int(hidden_size / num_attention_heads)
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    input_width = input_shape[2]

    # The Transformer performs sum residuals on all layers so the input needs
    # to be the same as the hidden size.
    if input_width != hidden_size:
        raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                         (input_width, hidden_size))

    # We keep the representation as a 2D tensor to avoid re-shaping it back and
    # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
    # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
    # help the optimizer.
    prev_output = reshape_to_matrix(input_tensor)

    all_layer_outputs = []
    for layer_idx in range(num_hidden_layers):
        with tf.variable_scope("layer_%d" % layer_idx):
            layer_input = prev_output

            with tf.variable_scope("attention"):
                attention_heads = []
                with tf.variable_scope("self"):
                    attention_head = attention_layer(
                        from_tensor=layer_input,
                        to_tensor=layer_input,
                        attention_mask=attention_mask,
                        num_attention_heads=num_attention_heads,
                        size_per_head=attention_head_size,
                        attention_probs_dropout_prob=attention_probs_dropout_prob,
                        initializer_range=initializer_range,
                        do_return_2d_tensor=True,
                        batch_size=batch_size,
                        from_seq_length=seq_length,
                        to_seq_length=seq_length)
                    attention_heads.append(attention_head)

                attention_output = None
                if len(attention_heads) == 1:
                    attention_output = attention_heads[0]
                else:
                    # In the case where we have other sequences, we just concatenate
                    # them to the self-attention head before the projection.
                    attention_output = tf.concat(attention_heads, axis=-1)

                # Run a linear projection of `hidden_size` then add a residual
                # with `layer_input`.
                with tf.variable_scope("output") as scope:
                    attention_output = tf.layers.dense(
                        attention_output,
                        hidden_size,
                        name="attention_output",
                        kernel_initializer=create_initializer(initializer_range))
                    attention_output = dropout(attention_output, hidden_dropout_prob)
                    attention_output = layer_norm(attention_output + layer_input, scope)

            # The activation is only applied to the "intermediate" hidden layer.
            with tf.variable_scope("intermediate"):
                intermediate_output = tf.layers.dense(
                    attention_output,
                    intermediate_size,
                    name="intermediate_output",
                    activation=intermediate_act_fn,
                    kernel_initializer=create_initializer(initializer_range))

            # Down-project back to `hidden_size` then add the residual.
            with tf.variable_scope("output") as scope:
                layer_output = tf.layers.dense(
                    intermediate_output,
                    hidden_size,  # if num_hidden_layers != 1 else 1,
                    name="layer_output",
                    kernel_initializer=create_initializer(initializer_range))
                layer_output = dropout(layer_output, hidden_dropout_prob)
                layer_output = layer_norm(layer_output + attention_output, scope)
                prev_output = layer_output
                all_layer_outputs.append(layer_output)

    if do_return_all_layers:
        final_outputs = []
        for layer_output in all_layer_outputs:
            final_output = reshape_from_matrix(layer_output, input_shape)
            final_outputs.append(final_output)
        return final_outputs
    else:
        final_output = reshape_from_matrix(prev_output, input_shape)
        return final_output


class TextCNN:
    def __init__(self, input_dim, filter_sizes, filter_num, scope_name="textcnn"):
        self.filter_sizes = filter_sizes
        self.filters = []
        self.bs = []
        self.scope_name = scope_name
        for filter_size in filter_sizes:
            with tf.variable_scope("conv{}{}".format(self.scope_name, filter_size)):
                filter_shape = tf.convert_to_tensor([filter_size, input_dim, 1, filter_num])
                fil = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='filter')
                b = tf.Variable(tf.constant(0.1, shape=[filter_num]))

                self.filters.append(fil)
                self.bs.append(b)

    def __call__(self, inputs, mask=None):

        with tf.variable_scope("textcnn_extract"):
            pooled_outputs = []
            inputs = inputs * tf.expand_dims(tf.cast(mask, tf.float32), axis=-1) if mask is not None else inputs
            input_expand = tf.expand_dims(inputs, -1)  # (b,m,d,1)
            for filter_size, fil, b in zip(self.filter_sizes, self.filters, self.bs):
                with tf.variable_scope("conv{}{}".format(self.scope_name, filter_size)):
                    conv = tf.nn.conv2d(input_expand, fil, [1]*4, 'VALID', name='conv')
                    h = tf.nn.relu(tf.nn.bias_add(conv, b))
                    pooled = tf.reduce_max(h, 1, True)
                    pooled_outputs.append(tf.squeeze(pooled, axis=[1, 2]))
            return tf.concat(pooled_outputs, 1)


class RNNExtract:
    def __init__(self, num_units, batch_size, input_size, keep_prob, is_train, mode="GRU"):
        self.keep_prob = keep_prob
        self.input_size = input_size
        self.is_train = is_train

        rnn_layer = NativeLSTM if mode == "LSTM" else NativeGRU
        self.rnn_layer = rnn_layer(num_layers=1, num_units=num_units,
                                   batch_size=batch_size, input_size=input_size,
                                   keep_prob=keep_prob, is_train=is_train,
                                   scope="native_rnn", activation=tf.nn.relu)

    def reduce_max(self, rnn_output):
        rnn_vec = tf.reduce_max(rnn_output, axis=1)

        rnn_vec = dropout(rnn_vec, 1 - self.keep_prob)
        rnn_vec = tf.layers.dense(rnn_vec, self.input_size,
                                  activation=tf.tanh,
                                  kernel_initializer=create_initializer(0.02))

        return rnn_vec

    def attention(self, rnn_output, mark, input_mask):
        with tf.variable_scope("att"):
            all_seq_len = rnn_output.get_shape().as_list()[1]
            cls = tf.tile(tf.expand_dims(mark, axis=1), [1, all_seq_len, 1])
            cat_att = tf.concat([cls, rnn_output], axis=2)

            res = tf.layers.dense(cat_att, units=1, activation=tf.nn.relu)
            res_mask = tf.expand_dims(tf.cast(input_mask, tf.float32), axis=2)
            res = res - (1 - res_mask) * 1e10

            alpha = tf.nn.softmax(res, 1)
            rnn_vec = tf.reduce_sum(alpha * rnn_output, axis=1)

            rnn_vec = dropout(rnn_vec, 1 - self.keep_prob)
            rnn_vec = tf.layers.dense(rnn_vec, self.input_size,
                                      activation=tf.tanh,
                                      kernel_initializer=create_initializer(0.02))

            return rnn_vec

    def __call__(self, inputs, input_mask, mark0=None):
        with tf.variable_scope("rnn_extract"):
            seq_len = tf.reduce_sum(input_mask, axis=1)
            rnn_output = self.rnn_layer(inputs, seq_len)

            if mark0 is None:
                return self.reduce_max(rnn_output)
            else:
                return self.reduce_max(rnn_output), self.attention(rnn_output, mark0, input_mask)


class InteractionExtract:
    def __init__(self, num_units, seq_len):
        self.num_units = num_units
        self.seq_len = seq_len

    def creat_indicator(self, all_ids):
        exp1 = tf.tile(tf.expand_dims(all_ids, axis=1), [1, self.seq_len, 1])  # (B, Lc, L)
        exp2 = tf.tile(tf.expand_dims(all_ids, axis=2), [1, 1, self.seq_len])  # (B, L, Lc)
        matrix = tf.cast(tf.equal(exp1, exp2), tf.float32)
        return tf.expand_dims(matrix, axis=3)

    def creat_dot_product(self, all_sent):
        from tensorflow.python.keras.layers.merge import dot
        return tf.expand_dims(dot([all_sent, all_sent], axes=[2, 2]), axis=3)

    def creat_cosine(self, all_sent):
        from tensorflow.python.keras.layers.merge import dot
        return tf.expand_dims(dot([all_sent, all_sent], axes=[2, 2], normalize=True), axis=3)

    def creat_euclidean(self, all_sent):
        exp1 = tf.expand_dims(all_sent, axis=1)  # (B, Lc, 1, dim)
        exp2 = tf.expand_dims(all_sent, axis=2)  # (B, 1, Lc, dim)

        t_exp1 = tf.tile(exp1, [1, 1, self.seq_len, 1])
        t_exp2 = tf.tile(exp2, [1, self.seq_len, 1, 1])

        t_exp = tf.concat([t_exp1, t_exp2], axis=3)
        score = tf.layers.dense(t_exp, units=1, activation=tf.tanh)

        return tf.sqrt(tf.reduce_sum(tf.square(exp1 - exp2), axis=3, keepdims=True)), score

    def dynamic_conv_pooling(self, image_feature, filters, kernel_size, output_size):
        image_feature = tf.layers.conv2d(image_feature, filters=filters, kernel_size=kernel_size,
                                         activation=tf.nn.relu)
        hieght = math.ceil(image_feature.get_shape().as_list()[1] / output_size[0])
        wieght = math.ceil(image_feature.get_shape().as_list()[2] / output_size[1])
        return tf.layers.max_pooling2d(image_feature, [hieght, wieght], [hieght, wieght])

    def __call__(self, all_sent, sent1_mask, sent2_mask, dropout_rate, all_sent_ids=None):
        sent1_mask = tf.cast(sent1_mask, tf.float32)
        sent2_mask = tf.cast(sent2_mask, tf.float32)

        with tf.variable_scope("inter_extract"):
            # indicator = self.creat_indicator(all_sent_ids)  # (B, L, L, 1)
            dot_product = self.creat_dot_product(all_sent)
            # cosine = self.creat_cosine(all_sent)

            # text_image = tf.concat([indicator, dot_product, cosine], axis=3)  # (B, L, L, 3)
            text_image = tf.nn.relu(dot_product)
            sent1_mask = tf.expand_dims(sent1_mask, 2)
            sent2_mask = tf.expand_dims(sent2_mask, 2)
            image_mask = tf.keras.layers.dot([sent1_mask, sent2_mask], axes=[2, 2])
            text_image = tf.expand_dims(image_mask, axis=3) * text_image  # (B, S1, S2, 3)

            image_feature = self.dynamic_conv_pooling(text_image, 128, [5, 5], [50, 50])
            image_feature = self.dynamic_conv_pooling(image_feature, 64, [5, 5], [10, 10])

            image_shape = image_feature.get_shape().as_list()
            image_flatten = tf.reshape(image_feature, [-1, image_shape[1] * image_shape[2] * image_shape[3]])

            image_flatten = dropout(image_flatten, dropout_rate)
            output = tf.layers.dense(image_flatten, 3072,
                                     activation=tf.nn.tanh,   # or gelu
                                     kernel_initializer=create_initializer(0.02))
            output = dropout(output, dropout_rate)

            output = tf.layers.dense(output, self.num_units,
                                     kernel_initializer=create_initializer(0.02))

            return output


class SingleSentenceExtract:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate

    def __call__(self, input_tensor, attention_mask=None):
        return transformer_model(input_tensor, attention_mask,
                                      num_hidden_layers=1,
                                      hidden_dropout_prob=self.dropout_rate,
                                      attention_probs_dropout_prob=self.dropout_rate)


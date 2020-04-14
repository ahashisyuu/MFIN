import tensorflow as tf

from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn import dynamic_rnn
from layers.BiGRU import NativeGRU as BiGRU
from layers.DoubleJointInteraction import DoubleJointCell
from layers.DoubleJointInteractionCNN import DoubleJointCNNCell
from layers.TriangularAlign import TriangularCell
from modeling import gelu, dropout, layer_norm
from layers.DoubleProbs import DoublePCell
from layers.DoubleUpdate import DoubleCell
from layers.DoubleUpdate2 import DoubleCell2
from layers.ParallelInfo import transformer_model
from modeling import create_initializer


class CQAModel:
    def __init__(self, is_training,
                 all_sent, input_mask,
                 sent1, sent2, sent3=None,
                 sent1_mask=None, sent2_mask=None, sent3_mask=None,
                 mark0=None, mark1=None, mark2=None, mark3=None,
                 embedding=None, update_num=3):
        self.is_training = is_training
        self.dropout_rate = 0.0
        self.update_num = update_num
        if self.is_training:
            self.dropout_rate = 0.1
        self.all_sent = all_sent
        self.input_mask = input_mask

        self.sent1 = sent1
        self.sent2 = sent2
        self.sent3 = sent3

        self.sent1_mask = sent1_mask
        self.sent2_mask = sent2_mask
        self.sent3_mask = sent3_mask

        self.mark0 = mark0
        self.mark1 = mark1
        self.mark2 = mark2
        self.mark3 = mark3

        self.embedding = embedding

        # self.output = None
        with tf.variable_scope("CQAModel", reuse=tf.AUTO_REUSE):
            self.output = self.build_model()

    def build_model(self):
        raise NotImplementedError

    def get_output(self):
        """return a vector representing the relation between question and answer"""
        return self.output


class Baseline(CQAModel):
    def build_model(self):
        med = tf.layers.dense(
            self.mark0,
            units=768, activation=tf.tanh, name="med")
        return med


class DoubleModel(CQAModel):
    def build_model(self):
        with tf.variable_scope("inferring_module"):
            rdim = 768
            update_num = 1
            batch_size = tf.shape(self.sent1)[0]
            dim = self.sent1.get_shape().as_list()[-1]

            sr_cell = GRUCell(num_units=rdim, activation=tf.nn.relu)

            sent_cell = r_cell = sr_cell

            tri_cell = DoubleCell(num_units=rdim,
                                  sent_cell=sent_cell, r_cell=r_cell,
                                  sent1=self.sent1, sent2=self.sent2,
                                  # sent1_length=self.Q_maxlen,
                                  # sent2_length=self.C_maxlen,
                                  dim=dim,
                                  use_bias=False, activation=tf.nn.tanh,
                                  sent1_mask=self.sent1_mask, sent2_mask=self.sent2_mask,
                                  initializer=None, dtype=tf.float32)

            fake_input = tf.tile(tf.expand_dims(self.mark0, axis=1), [1, update_num, 1])
            self.init_state = tri_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

            self.double_output, last_state = dynamic_rnn(cell=tri_cell,
                                                         inputs=fake_input,
                                                         initial_state=self.init_state)
            refer_output = last_state[2]  # (B, dim)
        temp = tf.concat([refer_output, self.mark0], axis=1)

        temp = dropout(temp, self.dropout_rate)
        gate = tf.layers.dense(temp,
                               768, activation=tf.sigmoid,
                               kernel_initializer=create_initializer(0.02))

        return refer_output * (1 - gate) + gate * self.mark0


class DCMN(CQAModel):
    def build_model(self):
        from tensorflow.python.keras.layers import Dense, Dot

        dim = self.sent1.get_shape().as_list()[-1]
        temp_W = tf.layers.dense(self.sent2, dim, name="dense")  # (B, L2, dim)
        temp_W = Dot(axes=[2, 2])([self.sent1, temp_W])  # (B, L1, L2)

        if self.sent1_mask is not None:
            s1_mask_exp = tf.expand_dims(self.sent1_mask, axis=2)  # (B, L1, 1)
            s2_mask_exp = tf.expand_dims(self.sent2_mask, axis=1)  # (B, 1, L2)
            temp_W1 = temp_W - (1 - s1_mask_exp) * 1e20
            temp_W2 = temp_W - (1 - s2_mask_exp) * 1e20
        else:
            temp_W1 = temp_W
            temp_W2 = temp_W

        W1 = tf.nn.softmax(temp_W1, axis=1)
        W2 = tf.nn.softmax(temp_W2, axis=2)

        M1 = Dot(axes=[2, 1])([W2, self.sent2])
        M2 = Dot(axes=[2, 1])([W1, self.sent1])

        s1_cat = tf.concat([M2 - self.sent2, M2 * self.sent2], axis=-1)
        s2_cat = tf.concat([M1 - self.sent1, M1 * self.sent1], axis=-1)

        S1 = tf.layers.dense(s1_cat, dim, activation=tf.nn.relu, name="cat_dense")
        S2 = tf.layers.dense(s2_cat, dim, activation=tf.nn.relu, name="cat_dense", reuse=True)

        if self.is_training:
            S1 = dropout(S1, dropout_prob=0.1)
            S1 = dropout(S1, dropout_prob=0.1)

        if self.sent1_mask is not None:
            S2 = S2 * tf.expand_dims(self.sent1_mask, axis=2)
            S1 = S1 * tf.expand_dims(self.sent2_mask, axis=2)

        C1 = tf.reduce_max(S1, axis=1)
        C2 = tf.reduce_max(S2, axis=1)

        C_cat = tf.concat([C1, C2], axis=1)

        return gelu(tf.layers.dense(C_cat, dim))


class DoubleModelUpGrade(CQAModel):
    def build_model(self):
        with tf.variable_scope("inferring_module"):
            rdim = 256
            update_num = 3
            batch_size = tf.shape(self.sent1)[0]
            dim = self.sent1.get_shape().as_list()[-1]

            sr_cell = GRUCell(num_units=rdim, activation=tf.nn.relu)

            sent_cell = r_cell = sr_cell

            tri_cell = DoubleCell(num_units=rdim,
                                  sent_cell=sent_cell, r_cell=r_cell,
                                  sent1=self.sent1, sent2=self.sent2,
                                  # sent1_length=self.Q_maxlen,
                                  # sent2_length=self.C_maxlen,
                                  dim=dim,
                                  use_bias=False, activation=tf.nn.relu,
                                  sent1_mask=self.sent1_mask, sent2_mask=self.sent2_mask,
                                  initializer=None, dtype=tf.float32)

            fake_input = tf.tile(tf.expand_dims(self.mark0, axis=1), [1, update_num, 1])
            self.init_state = tri_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

            self.double_output, last_state = dynamic_rnn(cell=tri_cell,
                                                         inputs=fake_input,
                                                         initial_state=self.init_state)
            score = tf.layers.dense(self.double_output[:, :, :rdim], units=1, activation=tf.tanh)
            alpha = tf.nn.softmax(score, axis=1)
            refer_output = tf.reduce_sum(alpha * score, axis=1)

        return refer_output


class DoubleModelUpGradeLoss(CQAModel):
    def build_model(self):
        with tf.variable_scope("inferring_module"):
            rdim = 256
            update_num = 3
            batch_size = tf.shape(self.sent1)[0]
            dim = self.sent1.get_shape().as_list()[-1]

            sr_cell = GRUCell(num_units=rdim, activation=tf.nn.relu)

            sent_cell = r_cell = sr_cell

            tri_cell = DoublePCell(num_units=rdim,
                                   sent_cell=sent_cell, r_cell=r_cell,
                                   sent1=self.sent1, sent2=self.sent2,
                                   # sent1_length=self.Q_maxlen,
                                   # sent2_length=self.C_maxlen,
                                   dim=dim,
                                   use_bias=False, activation=tf.nn.relu,
                                   sent1_mask=self.sent1_mask, sent2_mask=self.sent2_mask,
                                   initializer=None, dtype=tf.float32)

            fake_input = tf.tile(tf.expand_dims(self.mark0, axis=1), [1, update_num, 1])
            self.init_state = tri_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

            self.double_output, last_state = dynamic_rnn(cell=tri_cell,
                                                         inputs=fake_input,
                                                         initial_state=self.init_state)

        return self.double_output


class Double2Model(CQAModel):
    def build_model(self):
        with tf.variable_scope("inferring_module"):
            rdim = 256
            update_num = 3
            batch_size = tf.shape(self.sent1)[0]
            dim = self.sent1.get_shape().as_list()[-1]

            sr_cell = GRUCell(num_units=rdim, activation=tf.nn.relu)

            sent_cell = r_cell = sr_cell

            tri_cell = DoubleCell2(num_units=rdim,
                                   sent_cell=sent_cell, r_cell=r_cell,
                                   sent1=self.sent1, sent2=self.sent2,
                                   # sent1_length=self.Q_maxlen,
                                   # sent2_length=self.C_maxlen,
                                   dim=dim,
                                   use_bias=False, activation=tf.nn.relu,
                                   sent1_mask=self.sent1_mask, sent2_mask=self.sent2_mask,
                                   initializer=None, dtype=tf.float32)

            fake_input = tf.tile(tf.expand_dims(self.mark0, axis=1), [1, update_num, 1])
            self.init_state = tri_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

            self.double_output, last_state = dynamic_rnn(cell=tri_cell,
                                                         inputs=fake_input,
                                                         initial_state=self.init_state)
            refer_output = last_state[2]  # (B, dim)

        return refer_output


class LSTMModel(CQAModel):
    def build_model(self):
        with tf.variable_scope("inferring_module"), tf.device("/device:GPU:0"):
            rdim = 768
            update_num = 3
            batch_size = tf.shape(self.sent1)[0]
            dim = self.sent1.get_shape().as_list()[-1]

            gru_layer = BiGRU(num_layers=1, num_units=rdim, batch_size=batch_size,
                              input_size=dim, keep_prob=0.9, is_train=self.is_training,
                              activation=tf.nn.relu)
            seq_len = tf.reduce_sum(self.input_mask, axis=1)
            gru_output = gru_layer(self.all_sent, seq_len=seq_len)
            gru_vec = tf.reduce_max(gru_output, axis=1)

            gru_vec = dropout(gru_vec, self.dropout_rate)
            gru_vec = tf.layers.dense(gru_vec, 768,
                                      activation=tf.tanh,
                                      kernel_initializer=create_initializer(0.02))

            # gate = tf.layers.dense(tf.concat([gru_vec, self.mark0], axis=1),
            #                        rdim, activation=tf.sigmoid,
            #                        kernel_initializer=create_initializer(0.02))

            with tf.variable_scope("merge"):
                # refer_output = self.mark0 * gate + (1 - gate) * gru_vec
                vec_cat = tf.concat([self.mark0, gru_vec], axis=1)
                vec_cat = dropout(vec_cat, self.dropout_rate)
                pooled_output = tf.layers.dense(vec_cat, 768,
                                                activation=tf.tanh,
                                                kernel_initializer=create_initializer(0.02))

        return pooled_output


class IBERT(CQAModel):
    def DCMN(self, sent1, sent2, sent1_mask, sent2_mask):
        from tensorflow.python.keras.layers import Dense, Dot

        dim = sent1.get_shape().as_list()[-1]
        temp_W = tf.layers.dense(sent2, dim, name="dense")  # (B, L2, dim)
        temp_W = Dot(axes=[2, 2])([sent1, temp_W])  # (B, L1, L2)

        if sent1_mask is not None:
            s1_mask_exp = tf.expand_dims(sent1_mask, axis=2)  # (B, L1, 1)
            s2_mask_exp = tf.expand_dims(sent2_mask, axis=1)  # (B, 1, L2)
            temp_W1 = temp_W - (1 - s1_mask_exp) * 1e20
            temp_W2 = temp_W - (1 - s2_mask_exp) * 1e20
        else:
            temp_W1 = temp_W
            temp_W2 = temp_W

        W1 = tf.nn.softmax(temp_W1, axis=1)
        W2 = tf.nn.softmax(temp_W2, axis=2)

        M1 = Dot(axes=[2, 1])([W2, sent2])
        M2 = Dot(axes=[2, 1])([W1, sent1])

        # s1_cat = tf.concat([M2 - sent2, M2 * sent2], axis=-1)
        # s2_cat = tf.concat([M1 - sent1, M1 * sent1], axis=-1)

        # S1 = tf.layers.dense(s1_cat, dim, activation=tf.nn.relu, name="cat_dense")
        # S2 = tf.layers.dense(s2_cat, dim, activation=tf.nn.relu, name="cat_dense", reuse=True)

        # if self.is_training:
        #     S1 = dropout(S1, dropout_prob=0.1)
        #     S2 = dropout(S2, dropout_prob=0.1)
        #
        # if sent1_mask is not None:
        #     S2 = S2 * tf.expand_dims(sent1_mask, axis=2)
        #     S1 = S1 * tf.expand_dims(sent2_mask, axis=2)

        C1 = tf.reduce_max(M1, axis=1)
        C2 = tf.reduce_max(M2, axis=1)

        C_cat = tf.concat([C1, C2], axis=1)

        return gelu(tf.layers.dense(C_cat, dim))

    def build_model(self):
        from modeling import transformer_model, create_attention_mask_from_input_mask

        if self.is_training:
            dropout_prob = 0.1
        else:
            dropout_prob = 0.0

        attention1_mask = create_attention_mask_from_input_mask(
            self.sent1, self.sent1_mask)
        attention2_mask = create_attention_mask_from_input_mask(
            self.sent2, self.sent2_mask)

        # sent1 = transformer_model(self.sent1, attention1_mask,
        #                           hidden_size=768, num_hidden_layers=1,
        #                           intermediate_size=3072,
        #                           hidden_dropout_prob=dropout_prob,
        #                           attention_probs_dropout_prob=dropout_prob)
        # sent2 = transformer_model(self.sent2, attention2_mask,
        #                           hidden_size=768, num_hidden_layers=1,
        #                           intermediate_size=3072,
        #                           hidden_dropout_prob=dropout_prob,
        #                           attention_probs_dropout_prob=dropout_prob)
        sent1 = self.sent1
        sent2 = self.sent2

        d_vec = self.DCMN(sent1, sent2, self.sent1_mask, self.sent2_mask)

        gate = tf.layers.dense(tf.concat([d_vec, self.mark0], axis=1),
                               768, activation=tf.sigmoid,
                               kernel_initializer=create_initializer(0.02))

        refer_output = self.mark0 * gate + (1 - gate) * d_vec
        tf.keras.layers.BatchNormalization

        return refer_output


class MergeDouble(CQAModel):
    def build_model(self):
        with tf.variable_scope("inferring_module"), tf.device("/device:GPU:0"):
            rdim = 768
            update_num = 3
            batch_size = tf.shape(self.sent1)[0]
            dim = self.sent1.get_shape().as_list()[-1]

            gru_layer = BiGRU(num_layers=1, num_units=rdim, batch_size=batch_size,
                              input_size=dim, keep_prob=0.9, is_train=self.is_training,
                              activation=tf.nn.relu)
            seq_len = tf.reduce_sum(self.input_mask, axis=1)
            gru_output = gru_layer(self.all_sent, seq_len=seq_len)


class GRUAttModel(CQAModel):
    def build_model(self):
        with tf.variable_scope("inferring_module"), tf.device("/device:GPU:0"):
            rdim = 768
            update_num = 3
            batch_size = tf.shape(self.sent1)[0]
            dim = self.sent1.get_shape().as_list()[-1]

            gru_layer = BiGRU(num_layers=1, num_units=rdim, batch_size=batch_size,
                              input_size=dim, keep_prob=0.9, is_train=self.is_training,
                              activation=tf.nn.tanh)
            seq_len = tf.reduce_sum(self.input_mask, axis=1)
            gru_output = gru_layer(self.all_sent, seq_len=seq_len)

            with tf.variable_scope("att"):
                all_seq_len = self.all_sent.get_shape().as_list()[1]
                cls = tf.tile(tf.expand_dims(self.mark0, axis=1), [1, all_seq_len, 1])
                cat_att = tf.concat([cls, gru_output], axis=2)

                res = tf.layers.dense(cat_att, units=512, activation=tf.nn.relu)
                res = tf.layers.dense(res, units=1, use_bias=False)
                res_mask = tf.expand_dims(tf.cast(self.input_mask, tf.float32), axis=2)
                res = res - (1 - res_mask) * 10000.0

                alpha = tf.nn.softmax(res, 1)
                gru_vec = tf.reduce_sum(alpha * gru_output, axis=1)

            # gru_vec = dropout(gru_vec, self.dropout_rate)
            gru_vec = tf.layers.dense(gru_vec, 768,
                                      activation=gelu,
                                      kernel_initializer=create_initializer(0.02))
            gru_vec = dropout(gru_vec, self.dropout_rate)
            gru_vec = layer_norm(gru_vec + self.mark0)
            gru_vec = tf.layers.dense(gru_vec, 768,
                                      activation=tf.tanh,
                                      kernel_initializer=create_initializer(0.02))
            # gate = tf.layers.dense(tf.concat([gru_vec, self.mark0], axis=1),
            #                        rdim, activation=tf.sigmoid,
            #                        kernel_initializer=create_initializer(0.02))

            # with tf.variable_scope("merge"):
            #     # refer_output = self.mark0 * gate + (1 - gate) * gru_vec
            #     vec_cat = tf.concat([self.mark0, gru_vec], axis=1)
            #     vec_cat = dropout(vec_cat, self.dropout_rate)
            #     pooled_output = tf.layers.dense(vec_cat, 768,
            #                                     activation=tf.tanh,
            #                                     kernel_initializer=create_initializer(0.02))

        return gru_vec


def multi_hop(mark, all_sent, seq_len, gru_layer: BiGRU, dropout_rate, is_first=False):
    length = all_sent.get_shape().as_list()[1]
    rdim = mark.get_shape().as_list()[-1]

    exp_mark = tf.tile(tf.expand_dims(mark, axis=1), [1, length, 1])

    gru_output = gru_layer(tf.concat([all_sent, exp_mark], axis=2), seq_len)
    gru_vec = tf.reduce_max(gru_output, axis=1)
    gru_vec = dropout(gru_vec, dropout_rate)

    if is_first:
        trans = "trans"
    else:
        trans = "trans1"
    gru_vec = tf.layers.dense(gru_vec, rdim,
                              activation=tf.tanh, name=trans,
                              kernel_initializer=create_initializer(0.02))

    gate = tf.layers.dense(tf.concat([gru_vec, mark], axis=1),
                           rdim, activation=tf.sigmoid, name="gate",
                           kernel_initializer=create_initializer(0.02))
    refer_output = mark * gate + (1 - gate) * gru_vec

    return refer_output, gru_output


class MHGRUModel(CQAModel):
    def build_model(self):
        with tf.variable_scope("inferring_module"), tf.device("/device:GPU:0"):
            rdim = 768
            update_num = 2
            batch_size = tf.shape(self.sent1)[0]
            dim = self.sent1.get_shape().as_list()[-1]

            seq_len = tf.reduce_sum(self.input_mask, axis=1)

            with tf.variable_scope("multi_hop", reuse=tf.AUTO_REUSE):
                mark0 = self.mark0
                all_sent = self.all_sent
                mark_list = [mark0]

                gru_layer = BiGRU(num_layers=1, num_units=rdim // 2, batch_size=batch_size,
                                  input_size=2 * dim, keep_prob=0.9, is_train=self.is_training,
                                  activation=tf.nn.relu, scope="native_gru0")

                mark0, all_sent = multi_hop(mark0, all_sent, seq_len, gru_layer, self.dropout_rate, is_first=True)
                mark_list.append(mark0)

                # gru_layer = BiGRU(num_layers=1, num_units=rdim, batch_size=batch_size,
                #                   input_size=3 * dim, keep_prob=0.9, is_train=self.is_training,
                #                   activation=tf.nn.relu, scope="native_gru1")
                for _ in range(update_num-1):
                    mark0, all_sent = multi_hop(mark0, all_sent, seq_len, gru_layer, self.dropout_rate, is_first=True)
                    mark_list.append(mark0)

            # mark0 = tf.add_n(mark_list) / float(update_num)

        return mark0


class MultiPool(CQAModel):
    def build_model(self):
        from layers.ParallelInfo import TextCNN, RNNExtract, InteractionExtract, SingleSentenceExtract

        with tf.variable_scope("inferring_module"), tf.device("/device:GPU:0"):
            rdim = 768
            batch_size = tf.shape(self.sent1)[0]
            sent_length = self.all_sent.get_shape().as_list()[1]
            dim = self.sent1.get_shape().as_list()[-1]

            # text_cnn = TextCNN(rdim, [1, 2, 3, 4, 5, 7], 50)
            rnn_ext = RNNExtract(num_units=rdim, batch_size=batch_size, input_size=dim,
                                 keep_prob=0.9, is_train=self.is_training)
            # img_ext = InteractionExtract(num_units=256, seq_len=sent_length)

            # text_vec = text_cnn(self.all_sent, mask=self.input_mask)
            rnn_vec = rnn_ext(self.all_sent, input_mask=self.input_mask)
            # img_vec = img_ext(self.all_sent, self.sent1_mask, self.sent2_mask, self.dropout_rate)

            temp_res = tf.concat([rnn_vec, self.mark0], axis=1)
            # temp_res = tf.reshape(temp_res, [-1, 3, dim])
            # alpha = tf.layers.dense(temp_res, 1, activation=tf.tanh)
            # alpha = tf.nn.softmax(alpha, axis=1)
            temp_res = dropout(temp_res, self.dropout_rate)

            # gate0 = tf.layers.dense(temp_res, units=rdim,  # activation=tf.nn.sigmoid,
            #                        kernel_initializer=create_initializer(0.02))
            # gate1 = tf.layers.dense(temp_res, units=rdim,  # activation=tf.nn.sigmoid,
            #                        kernel_initializer=create_initializer(0.02))
            # gate2 = tf.layers.dense(temp_res, units=rdim,  # activation=tf.nn.sigmoid,
            #                        kernel_initializer=create_initializer(0.02))
            # gate = tf.concat([gate0, gate1, gate2], axis=1)

            # res_vec = tf.reshape(temp_res, [-1, 3, rdim])
            # gate = tf.nn.softmax(tf.reshape(gate, [-1, 3, rdim]), axis=1)
            # score = transformer_model(res_vec, hidden_size=rdim, num_hidden_layers=1,
            #                           num_attention_heads=1, intermediate_size=rdim)
            # gate = tf.nn.softmax(score, axis=1)
            # return self.mark0 * gate + (1 - gate) * rnn_vec

            return tf.layers.dense(temp_res, 768, tf.tanh,
                                   kernel_initializer=create_initializer(0.02))
            # return tf.reduce_sum(alpha * temp_res, axis=1)
            # return img_vec


class HPool(CQAModel):
    def build_model(self):
        from layers.ParallelInfo import TextCNN, RNNExtract, InteractionExtract, SingleSentenceExtract

        with tf.variable_scope("inferring_module"), tf.device("/device:GPU:0"):
            rdim = 768
            batch_size = tf.shape(self.sent1)[0]
            sent_length = self.all_sent.get_shape().as_list()[1]
            update_num = 3
            dim = self.sent1.get_shape().as_list()[-1]

            gru_layer = BiGRU(num_layers=1, num_units=rdim, batch_size=batch_size,
                              input_size=dim, keep_prob=0.9, is_train=self.is_training,
                              activation=tf.nn.tanh)
            seq_len = tf.reduce_sum(self.input_mask, axis=1)
            gru_output = gru_layer(self.all_sent, seq_len=seq_len)

            with tf.variable_scope("att"):
                all_seq_len = self.all_sent.get_shape().as_list()[1]
                cls = tf.tile(tf.expand_dims(self.mark0, axis=1), [1, all_seq_len, 1])
                cat_att = tf.concat([cls, gru_output], axis=2)

                res = tf.layers.dense(cat_att, units=512, activation=tf.nn.relu)
                res = tf.layers.dense(res, units=1, use_bias=False)
                res_mask = tf.expand_dims(tf.cast(self.input_mask, tf.float32), axis=2)
                res = res - (1 - res_mask) * 10000.0

                alpha = tf.nn.softmax(res, 1)
                gru_vec = tf.reduce_sum(alpha * gru_output, axis=1)

            # gru_vec = dropout(gru_vec, self.dropout_rate)
            # gru_vec = tf.layers.dense(gru_vec, 768,
            #                           activation=gelu,
            #                           kernel_initializer=create_initializer(0.02))
            # gru_vec = dropout(gru_vec, self.dropout_rate)
            # gru_vec = layer_norm(gru_vec + self.mark0)
            # gru_vec = tf.layers.dense(gru_vec, 768,
            #                           activation=tf.tanh,
            #                           kernel_initializer=create_initializer(0.02))

            text_cnn = TextCNN(2*rdim, [1, 2, 3, 4, 5, 7], 128)
            # img_ext = InteractionExtract(num_units=256, seq_len=sent_length)

            text_vec = text_cnn(self.all_sent, mask=self.input_mask)
            # rnn_vec, rnn_att = rnn_ext(self.all_sent, input_mask=self.input_mask, mark0=self.mark0)
            # img_vec = img_ext(gru_output, self.sent1_mask, self.sent2_mask, self.dropout_rate)

            temp_vec_org = tf.concat([text_vec, gru_vec, self.mark0], axis=1)

            temp_vec = tf.layers.dense(temp_vec_org, 3072,
                                       activation=gelu,
                                       kernel_initializer=create_initializer(0.02))
            temp_vec = dropout(temp_vec, self.dropout_rate)

            temp_vec = tf.layers.dense(temp_vec, 768,
                                       kernel_initializer=create_initializer(0.02))

            # temp_vec = dropout(temp_vec, self.dropout_rate)

            # gru_vec = tf.layers.dense(gru_vec, 768,
            #                           activation=tf.tanh,
            #                           kernel_initializer=create_initializer(0.02))

            return temp_vec


class TripleModel(CQAModel):
    def build_model(self):
        with tf.variable_scope("inferring_module"):
            rdim = 768
            update_num = 2
            batch_size = tf.shape(self.sent1)[0]
            dim = self.sent1.get_shape().as_list()[-1]

            sr_cell = GRUCell(num_units=rdim, activation=tf.nn.relu)

            r_cell = sr_cell

            tri_cell = TriangularCell(num_units=rdim,
                                      r_cell=r_cell,
                                      sent1=self.sent1, sent2=self.sent2, sent3=self.sent3,
                                      sent1_length=39,
                                      sent2_length=110,
                                      sent3_length=152,
                                      dim=dim,
                                      use_bias=False, activation=tf.nn.relu,
                                      sent1_mask=self.sent1_mask, sent2_mask=self.sent2_mask, sent3_mask=self.sent3_mask,
                                      initializer=None, dtype=tf.float32)

            fake_input = tf.tile(tf.expand_dims(self.mark0, axis=1), [1, update_num, 1])
            self.init_state = tri_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

            self.double_output, last_state = dynamic_rnn(cell=tri_cell,
                                                         inputs=fake_input,
                                                         initial_state=self.init_state)
            r1_output, r2_output, r3_output = last_state[3:]  # (B, dim)
        temp13 = tf.concat([r1_output, r3_output, r1_output * r3_output], axis=1)
        temp23 = tf.concat([r2_output, r3_output, r2_output * r3_output], axis=1)

        temp13 = dropout(temp13, self.dropout_rate)
        temp23 = dropout(temp23, self.dropout_rate)
        r13 = tf.layers.dense(temp13,
                               768, activation=tf.tanh,
                               kernel_initializer=create_initializer(0.02))
        r23 = tf.layers.dense(temp23,
                              768, activation=tf.tanh,
                              kernel_initializer=create_initializer(0.02))
        temp = tf.concat([self.mark0, r13, r23], axis=1)
        refer_output = tf.layers.dense(temp, 768,
                                       activation=None,
                                       kernel_initializer=create_initializer(0.02))
        return refer_output


class DoubleJointModel(CQAModel):
    def build_model(self):
        with tf.variable_scope("inferring_module"):
            rdim = 768
            update_num = self.update_num
            batch_size = tf.shape(self.sent1)[0]
            dim = self.sent1.get_shape().as_list()[-1]

            gru_layer = BiGRU(num_layers=1, num_units=rdim, batch_size=batch_size,
                              input_size=dim, keep_prob=0.9, is_train=self.is_training,
                              activation=tf.nn.tanh)
            sent1_len = tf.cast(tf.reduce_sum(self.sent1_mask, axis=1), tf.int32)
            sent2_len = tf.cast(tf.reduce_sum(self.sent2_mask, axis=1), tf.int32)
            self.sent1 = gru_layer(self.sent1, sent1_len)
            self.sent2 = gru_layer(self.sent2, sent2_len)

            sr_cell = GRUCell(num_units=2*rdim, activation=tf.nn.relu)

            r_cell = sr_cell

            tri_cell = DoubleJointCell(num_units=2*rdim,
                                       r_cell=r_cell,
                                       sent1=self.sent1, sent2=self.sent2,
                                       dim=2*dim,
                                       update_num=update_num,
                                       use_bias=False, activation=tf.tanh,
                                       dropout_rate=self.dropout_rate,
                                       sent1_mask=self.sent1_mask, sent2_mask=self.sent2_mask,
                                       initializer=None, dtype=tf.float32)

            fake_input = tf.tile(tf.expand_dims(self.mark0, axis=1), [1, update_num, 1])
            self.init_state = tri_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

            self.double_output, last_state = dynamic_rnn(cell=tri_cell,
                                                         inputs=fake_input,
                                                         initial_state=self.init_state)
            refer_output = tf.reduce_mean(self.double_output, axis=1)  # (B, dim)
        # temp = tf.concat([refer_output, self.mark0], axis=1)
        #
        # temp = dropout(temp, self.dropout_rate)
        refer_output = tf.layers.dense(refer_output,
                               768, activation=tf.nn.tanh,
                               kernel_initializer=create_initializer(0.02))

        # return refer_output * (1 - gate) + gate * self.mark0
        return refer_output + self.mark0


class BiGRUModel(CQAModel):
    def build_model(self):
        with tf.variable_scope("inferring_module"):
            rdim = 768
            update_num = self.update_num
            batch_size = tf.shape(self.sent1)[0]
            dim = self.sent1.get_shape().as_list()[-1]

            gru_layer = BiGRU(num_layers=1, num_units=rdim, batch_size=batch_size,
                              input_size=dim, keep_prob=0.9, is_train=self.is_training,
                              activation=tf.nn.tanh)
            sent1_len = tf.cast(tf.reduce_sum(self.sent1_mask, axis=1), tf.int32)
            sent2_len = tf.cast(tf.reduce_sum(self.sent2_mask, axis=1), tf.int32)
            self.sent1 = gru_layer(self.sent1, sent1_len, return_type=1)
            self.sent2 = gru_layer(self.sent2, sent2_len, return_type=1)

            sent1_vec = tf.reduce_mean(self.sent1, axis=1)
            sent2_vec = tf.reduce_mean(self.sent2, axis=1)

        # temp = tf.concat([refer_output, self.mark0], axis=1)
        #
        # temp = dropout(temp, self.dropout_rate)
        refer_output = tf.layers.dense(tf.concat([sent1_vec, sent2_vec], axis=1),
                               768, activation=tf.nn.tanh,
                               kernel_initializer=create_initializer(0.02))

        # return refer_output * (1 - gate) + gate * self.mark0
        return refer_output + self.mark0


class DoubleJointWOGRUModel(CQAModel):

    def build_model(self):
        with tf.variable_scope("inferring_module"):
            rdim = 768
            update_num = self.update_num
            batch_size = tf.shape(self.sent1)[0]
            dim = self.sent1.get_shape().as_list()[-1]

            # gru_layer = BiGRU(num_layers=1, num_units=rdim, batch_size=batch_size,
            #                   input_size=dim, keep_prob=0.9, is_train=self.is_training,
            #                   activation=tf.nn.tanh)
            # sent1_len = tf.cast(tf.reduce_sum(self.sent1_mask, axis=1), tf.int32)
            # sent2_len = tf.cast(tf.reduce_sum(self.sent2_mask, axis=1), tf.int32)
            # self.sent1 = gru_layer(self.sent1, sent1_len)
            # self.sent2 = gru_layer(self.sent2, sent2_len)

            sr_cell = GRUCell(num_units=2*rdim, activation=tf.nn.relu)

            r_cell = sr_cell

            tri_cell = DoubleJointCell(num_units=2*rdim,
                                       r_cell=r_cell,
                                       sent1=self.sent1, sent2=self.sent2,
                                       dim=dim,
                                       update_num=update_num,
                                       use_bias=False, activation=tf.tanh,
                                       dropout_rate=self.dropout_rate,
                                       sent1_mask=self.sent1_mask, sent2_mask=self.sent2_mask,
                                       initializer=None, dtype=tf.float32)

            fake_input = tf.tile(tf.expand_dims(self.mark0, axis=1), [1, update_num, 1])
            self.init_state = tri_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

            self.double_output, last_state = dynamic_rnn(cell=tri_cell,
                                                         inputs=fake_input,
                                                         initial_state=self.init_state)
            refer_output = tf.reduce_mean(self.double_output, axis=1)  # (B, dim)
        # temp = tf.concat([refer_output, self.mark0], axis=1)
        #
        # temp = dropout(temp, self.dropout_rate)
        refer_output = tf.layers.dense(refer_output,
                               768, activation=tf.nn.tanh,
                               kernel_initializer=create_initializer(0.02))

        # return refer_output * (1 - gate) + gate * self.mark0
        return refer_output + self.mark0


class BERTCoATTModel(CQAModel):
    def MULT(self, sent1, sent2, sent1_mask, sent2_mask):
        from tensorflow.python.keras.layers import Dense, Dot

        dim = sent1.get_shape().as_list()[-1]
        length1 = sent1.get_shape().as_list()[1]
        length2 = sent2.get_shape().as_list()[1]
        temp_W = tf.layers.dense(sent2, dim, name="dense")  # (B, L2, dim)
        temp_W = Dot(axes=[2, 2])([sent1, temp_W])  # (B, L1, L2)

        if sent1_mask is not None:
            s1_mask_exp = tf.expand_dims(sent1_mask, axis=2)  # (B, L1, 1)
            s2_mask_exp = tf.expand_dims(sent2_mask, axis=1)  # (B, 1, L2)
            temp_W1 = temp_W - (1 - s1_mask_exp) * 1e20
            temp_W2 = temp_W - (1 - s2_mask_exp) * 1e20
        else:
            temp_W1 = temp_W
            temp_W2 = temp_W

        W1 = tf.nn.softmax(temp_W1, axis=1)
        W2 = tf.nn.softmax(temp_W2, axis=2)

        W1 = tf.transpose(W1, perm=[0, 2, 1])

        w1_val, w1_index = tf.nn.top_k(W1, k=20)
        w2_val, w2_index = tf.nn.top_k(W2, k=20)

        sent1_repeat = tf.tile(tf.expand_dims(sent1, axis=1), [1, length2, 1, 1])
        sent2_repeat = tf.tile(tf.expand_dims(sent2, axis=1), [1, length1, 1, 1])

        sent1_top = tf.batch_gather(sent1_repeat, w1_index)
        sent2_top = tf.batch_gather(sent2_repeat, w2_index)

        w1_val = w1_val / tf.reduce_sum(w1_val, axis=2, keepdims=True)
        w2_val = w2_val / tf.reduce_sum(w2_val, axis=2, keepdims=True)
        w1_val = tf.expand_dims(w1_val, axis=3)
        w2_val = tf.expand_dims(w2_val, axis=3)

        M1 = tf.reduce_sum(w2_val * sent2_top, axis=2)
        M2 = tf.reduce_sum(w1_val * sent1_top, axis=2)

        # M1 = Dot(axes=[2, 1])([W2, sent2])
        # M2 = Dot(axes=[1, 1])([W1, sent1])

        # s1_cat = tf.concat([M2 - sent2, M2 * sent2], axis=-1)
        # s2_cat = tf.concat([M1 - sent1, M1 * sent1], axis=-1)

        # S1 = tf.layers.dense(s1_cat, dim, activation=tf.nn.relu, name="cat_dense")
        # S2 = tf.layers.dense(s2_cat, dim, activation=tf.nn.relu, name="cat_dense", reuse=True)

        # if self.is_training:
        #     S1 = dropout(S1, dropout_prob=0.1)
        #     S2 = dropout(S2, dropout_prob=0.1)
        #
        S1 = M1 * sent1
        S2 = M2 * sent2

        if sent1_mask is not None:
            S1 = S1 * tf.expand_dims(sent1_mask, axis=2)
            S2 = S2 * tf.expand_dims(sent2_mask, axis=2)

        from layers.ParallelInfo import TextCNN
        cnn1 = TextCNN(dim, [1, 2, 3, 4, 5], dim, scope_name="cnn1")
        cnn2 = TextCNN(dim, [1, 2, 3, 4, 5], dim, scope_name="cnn2")
        S1 = cnn1(S1)
        S2 = cnn2(S2)
        feature1 = tf.layers.dense(S1, dim, activation=tf.tanh)
        feature2 = tf.layers.dense(S2, dim, activation=tf.tanh)
        feature_total = tf.concat([feature1, feature2], axis=1)

        return feature_total

    def build_model(self):
        with tf.variable_scope("inferring_module"):
            rdim = 768
            update_num = self.update_num
            batch_size = tf.shape(self.sent1)[0]
            dim = self.sent1.get_shape().as_list()[-1]

            gru_layer = BiGRU(num_layers=1, num_units=rdim, batch_size=batch_size,
                              input_size=dim, keep_prob=0.9, is_train=self.is_training,
                              activation=tf.nn.tanh)
            sent1_len = tf.cast(tf.reduce_sum(self.sent1_mask, axis=1), tf.int32)
            sent2_len = tf.cast(tf.reduce_sum(self.sent2_mask, axis=1), tf.int32)
            self.sent1 = gru_layer(self.sent1, sent1_len)
            self.sent2 = gru_layer(self.sent2, sent2_len)
            refer_output = self.MULT(self.sent1, self.sent2, self.sent1_mask, self.sent2_mask)

        refer_output = tf.layers.dense(refer_output,
                               768, activation=tf.nn.tanh,
                               kernel_initializer=create_initializer(0.02))

        # return refer_output * (1 - gate) + gate * self.mark0
        return refer_output + self.mark0


class DoubleJointWOBERTModel(CQAModel):
    def build_model(self):
        with tf.variable_scope("inferring_module"):
            rdim = 768
            update_num = self.update_num
            batch_size = tf.shape(self.sent1)[0]
            dim = self.sent1.get_shape().as_list()[-1]

            gru_layer = BiGRU(num_layers=1, num_units=rdim, batch_size=batch_size,
                              input_size=dim, keep_prob=0.9, is_train=self.is_training,
                              activation=tf.nn.tanh)
            sent1_len = tf.cast(tf.reduce_sum(self.sent1_mask, axis=1), tf.int32)
            sent2_len = tf.cast(tf.reduce_sum(self.sent2_mask, axis=1), tf.int32)
            self.sent1 = gru_layer(self.sent1, sent1_len)
            self.sent2 = gru_layer(self.sent2, sent2_len)

            sr_cell = GRUCell(num_units=2*rdim, activation=tf.nn.relu)

            r_cell = sr_cell

            tri_cell = DoubleJointCell(num_units=2*rdim,
                                       r_cell=r_cell,
                                       sent1=self.sent1, sent2=self.sent2,
                                       dim=2*dim,
                                       update_num=update_num,
                                       use_bias=False, activation=tf.tanh,
                                       dropout_rate=self.dropout_rate,
                                       sent1_mask=self.sent1_mask, sent2_mask=self.sent2_mask,
                                       initializer=None, dtype=tf.float32)

            fake_input = tf.tile(tf.expand_dims(self.mark0, axis=1), [1, update_num, 1])
            self.init_state = tri_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

            self.double_output, last_state = dynamic_rnn(cell=tri_cell,
                                                         inputs=fake_input,
                                                         initial_state=self.init_state)
            refer_output = tf.reduce_mean(self.double_output, axis=1)  # (B, dim)
        # temp = tf.concat([refer_output, self.mark0], axis=1)
        #
        # temp = dropout(temp, self.dropout_rate)
        refer_output = tf.layers.dense(refer_output,
                               768, activation=tf.nn.tanh,
                               kernel_initializer=create_initializer(0.02))

        # return refer_output * (1 - gate) + gate * self.mark0
        return refer_output


class DoubleJointWOconModel(CQAModel):
    def build_model(self):
        with tf.variable_scope("inferring_module"):
            rdim = 768
            update_num = self.update_num
            batch_size = tf.shape(self.sent1)[0]
            dim = self.sent1.get_shape().as_list()[-1]

            gru_layer = BiGRU(num_layers=1, num_units=rdim, batch_size=batch_size,
                              input_size=dim, keep_prob=0.9, is_train=self.is_training,
                              activation=tf.nn.tanh)
            sent1_len = tf.cast(tf.reduce_sum(self.sent1_mask, axis=1), tf.int32)
            sent2_len = tf.cast(tf.reduce_sum(self.sent2_mask, axis=1), tf.int32)
            self.sent1 = gru_layer(self.sent1, sent1_len)
            self.sent2 = gru_layer(self.sent2, sent2_len)

            sr_cell = GRUCell(num_units=2*rdim, activation=tf.nn.relu)

            r_cell = sr_cell

            tri_cell = DoubleJointCell(num_units=2*rdim,
                                       r_cell=r_cell,
                                       sent1=self.sent1, sent2=self.sent2,
                                       dim=2*dim,
                                       update_num=update_num,
                                       use_bias=False, activation=tf.tanh,
                                       dropout_rate=self.dropout_rate,
                                       sent1_mask=self.sent1_mask, sent2_mask=self.sent2_mask,
                                       initializer=None, dtype=tf.float32)

            fake_input = tf.tile(tf.expand_dims(self.mark0, axis=1), [1, update_num, 1])
            self.init_state = tri_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

            self.double_output, last_state = dynamic_rnn(cell=tri_cell,
                                                         inputs=fake_input,
                                                         initial_state=self.init_state)
            refer_output = tf.reduce_mean(self.double_output, axis=1)  # (B, dim)
        # temp = tf.concat([refer_output, self.mark0], axis=1)
        #
        # temp = dropout(temp, self.dropout_rate)
        refer_output = tf.layers.dense(refer_output,
                               768, activation=tf.nn.tanh,
                               kernel_initializer=create_initializer(0.02))

        # return refer_output * (1 - gate) + gate * self.mark0
        return refer_output


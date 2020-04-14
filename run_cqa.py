import os
import numpy as np
import tensorflow as tf
import pickle as pkl
import modeling
import optimization

from tensorflow.contrib import tpu

from EvalHook import EvalHook
from run_classifier import FLAGS
from utils import PRF, eval_reranker, print_metrics
from CQAModel import DoubleModel, Baseline, DoubleModelUpGrade, DCMN, IBERT, GRUAttModel, MHGRUModel, MultiPool, HPool, \
    LSTMModel, TripleModel, DoubleJointModel

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
CQAMODEL = DoubleJointModel


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_extract, input_mask, segment_ids, q_type, label_id):
        self.input_ids = input_ids
        self.input_extract = input_extract
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.q_type = q_type
        self.label_id = label_id


def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_extract = []
    all_input_mask = []
    all_segment_ids = []
    all_q_type = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_extract.append(feature.input_extract)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_q_type.append(feature.q_type)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        print(batch_size)
        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_extract":
                tf.constant(
                    all_input_extract, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "q_type":
                tf.constant(all_q_type, shape=[num_examples], dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


def input_fn_builder_v2(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input1_extract = []
    all_input2_extract = []
    all_input_mask = []
    all_segment_ids = []
    all_q_type = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input1_extract.append(feature.input1_extract)
        all_input2_extract.append(feature.input2_extract)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_q_type.append(feature.q_type)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        print(batch_size)
        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input1_extract":
                tf.constant(
                    all_input1_extract, shape=[num_examples, 110],
                    dtype=tf.int32),
            "input2_extract":
                tf.constant(
                    all_input2_extract, shape=[num_examples, 150],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "q_type":
                tf.constant(all_q_type, shape=[num_examples], dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


def input_fn_builder_v3(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input1_extract = []
    all_input2_extract = []
    all_input3_extract = []
    all_input_mask = []
    all_segment_ids = []
    all_q_type = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input1_extract.append(feature.input1_extract)
        all_input2_extract.append(feature.input2_extract)
        all_input3_extract.append(feature.input3_extract)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_q_type.append(feature.q_type)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        print(batch_size)
        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input1_extract":
                tf.constant(
                    all_input1_extract, shape=[num_examples, 39],
                    dtype=tf.int32),
            "input2_extract":
                tf.constant(
                    all_input2_extract, shape=[num_examples, 110],
                    dtype=tf.int32),
            "input3_extract":
                tf.constant(
                    all_input3_extract, shape=[num_examples, 152],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "q_type":
                tf.constant(all_q_type, shape=[num_examples], dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


def _creat_bert(is_training, features, bert_config, use_one_hot_embeddings, init_checkpoint, layer_num, plus_position):
    global initialized_variable_names
    input_ids = features["input_ids"]
    if "input_extract" in features:
        input_extract = features["input_extract"]
        input1_extract = None
        input2_extract = None
        input3_extract = None
    elif "input3_extract" not in features:
        input_extract = None
        input1_extract = features["input1_extract"]
        input2_extract = features["input2_extract"]
        input3_extract = None
    else:
        input_extract = None
        input1_extract = features["input1_extract"]
        input2_extract = features["input2_extract"]
        input3_extract = features["input3_extract"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    q_type = features["q_type"]
    label_ids = features["label_ids"]

    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        output_layer_index=layer_num,
        plus_position=plus_position)

    tvars = tf.trainable_variables()

    scaffold_fn = None
    if init_checkpoint:
        (assignment_map,
         initialized_variable_names) = modeling.get_assigment_map_from_checkpoint(tvars, init_checkpoint)

        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        # print("initialing checkpoint finished")

    # tf.logging.info("**** Trainable Variables ****")
    # residue = []
    # for var in tvars:
    #     init_string = ""
    #     if var.name in initialized_variable_names:
    #         init_string = ", *INIT_FROM_CKPT*"
    #     else:
    #         residue.append(var)
    #     tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
    #                     init_string)

    predictions = {"input_extract": input_extract,
                   "input1_extract": input1_extract,
                   "input2_extract": input2_extract,
                   "input3_extract": input3_extract,
                   "embedding": model.get_embedding_output(),
                   "input_mask": input_mask,
                   "q_type": q_type,
                   "label_ids": label_ids,
                   "output_layer": model.get_output_layer(),
                   "last_layer": model.get_sequence_output()}

    return predictions


def _create_cqa_modules(is_training, predictions, update_num):
    num_labels = 2
    input_extract = predictions["input_extract"]
    input1_extract = predictions["input1_extract"]
    input2_extract = predictions["input2_extract"]
    input3_extract = predictions["input3_extract"]
    embedding = predictions["embedding"]
    input_mask = predictions["input_mask"]
    q_type = predictions["q_type"]
    labels = predictions["label_ids"]
    encoder_output1 = predictions["last_layer"]
    # encoder_output = encoder_output1 + encoder_output2 + \
    #     encoder_output3 + encoder_output4
    encoder_output = predictions["output_layer"]



    sent1 = None
    sent2 = None
    sent3 = None

    sent1_mask = None
    sent2_mask = None
    sent3_mask = None

    mark0 = None
    mark1 = None
    mark2 = None
    mark3 = None

    if input_extract is None and input3_extract is None:
        sent1_mask = tf.cast(tf.not_equal(input1_extract, 0), tf.float32)
        sent2_mask = tf.cast(tf.not_equal(input2_extract, 0), tf.float32)

        sent1 = tf.batch_gather(encoder_output, input1_extract)
        sent2 = tf.batch_gather(encoder_output, input2_extract)
    elif input3_extract is None:
        sent1_mask = tf.cast(tf.equal(input_extract, 1), tf.float32)
        sent2_mask = tf.cast(tf.equal(input_extract, 2), tf.float32)

        sent1 = encoder_output * tf.expand_dims(sent1_mask, axis=-1)
        sent2 = encoder_output * tf.expand_dims(sent2_mask, axis=-1)
    else:
        sent1_mask = tf.cast(tf.not_equal(input1_extract, 0), tf.float32)
        sent2_mask = tf.cast(tf.not_equal(input2_extract, 0), tf.float32)
        sent3_mask = tf.cast(tf.not_equal(input3_extract, 0), tf.float32)

        sent1 = tf.batch_gather(encoder_output, input1_extract)
        sent2 = tf.batch_gather(encoder_output, input2_extract)
        sent3 = tf.batch_gather(encoder_output, input3_extract)

    mark0 = tf.squeeze(encoder_output1[:, 0:1, :], axis=1)

    model = CQAMODEL(is_training=is_training,
                      all_sent=encoder_output, input_mask=input_mask,
                      sent1=sent1, sent2=sent2, sent3=sent3,
                      sent1_mask=sent1_mask, sent2_mask=sent2_mask, sent3_mask=sent3_mask,
                      mark0=mark0, mark1=mark1, mark2=mark2, mark3=mark3,
                      embedding=embedding, update_num=update_num)
    # model = Baseline(is_training=is_training,
    #                  sent1=sent1, sent2=sent2, sent3=sent3,
    #                  sent1_mask=sent1_mask, sent2_mask=sent2_mask, sent3_mask=sent3_mask,
    #                  mark0=mark0, mark1=mark1, mark2=mark2, mark3=mark3)

    result = model.get_output()  # (B, dim)
    # mark0 = tf.layers.dense(mark0, 768, activation=tf.tanh)
    # result = mark0

    hidden_size = result.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights_v2", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias_v2", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            result = tf.nn.dropout(result, keep_prob=0.9)

        logits = tf.matmul(result, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        prob = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        total_loss = tf.reduce_mean(per_example_loss)

    return total_loss, logits, prob


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, update_num, layer_num,
                     plus_position):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        global initialized_variable_names
        # tf.logging.info("*** Features ***")
        # for name in sorted(features.keys()):
        #     tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        predictions = _creat_bert(is_training, features, bert_config, use_one_hot_embeddings, init_checkpoint, layer_num, plus_position)

        # the concatenate of predictions is the output of bert encoder
        # and it will be seen as input of other modules
        total_loss, logits, prob = _create_cqa_modules(is_training, predictions, update_num)

        scaffold_fn = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op, grade = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.PREDICT:
            output_spec = tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"logits": logits, "prob": prob},
                scaffold_fn=scaffold_fn)
        else:
            raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

        return output_spec

    return model_fn


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    session_config = tf.ConfigProto(log_device_placement=True)
    session_config.gpu_options.allow_growth = True
    run_config.replace(session_config=session_config)

    num_train_steps = None
    num_warmup_steps = None

    with open('dataset.pkl', 'rb') as fr:
        train_features, dev_cid, dev_features, test_cid, test_features = pkl.load(fr)
        dev_label = [feature.label_id for feature in dev_features]
        test_label = [feature.label_id for feature in test_features]

    if FLAGS.do_train:
        num_train_steps = int(
            len(train_features) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=2,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        # params={'batch_size': FLAGS.train_batch_size},
        train_batch_size=FLAGS.train_batch_size,
        predict_batch_size=FLAGS.eval_batch_size)

    if FLAGS.do_train:
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_features))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = input_fn_builder(
            features=train_features,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)

        estimator.train(input_fn=train_input_fn,
                        max_steps=num_train_steps,
                        hooks=[EvalHook(estimator=estimator,
                                        dev_features=dev_features,
                                        dev_label=dev_label,
                                        dev_cid=dev_cid,
                                        max_seq_length=FLAGS.max_seq_length,
                                        eval_steps=FLAGS.save_checkpoints_steps,
                                        checkpoint_dir=FLAGS.output_dir)])

    if FLAGS.do_eval:
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(test_features))
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        test_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            # Eval will be slightly WRONG on the TPU because it will truncate
            # the last batch.
            test_steps = int(len(test_features) / FLAGS.eval_batch_size)

        test_drop_remainder = True if FLAGS.use_tpu else False
        test_input_fn = input_fn_builder(
            features=test_features,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=test_drop_remainder)

        predictions = estimator.predict(test_input_fn, yield_single_examples=False)
        res = np.concatenate([a for a in predictions], axis=0)
        print(res.shape, np.array(dev_label).shape)
        metrics = PRF(np.array(dev_label), res.argmax(axis=-1))
        # print((np.array(dev_label) != res.argmax(axis=-1))[:1000])
        MAP, AvgRec, MRR = eval_reranker(test_cid, test_label, res[:, 0])
        metrics['MAP'] = MAP
        metrics['AvgRec'] = AvgRec
        metrics['MRR'] = MRR

        print_metrics(metrics, 'test')


if __name__ == '__main__':
    main()

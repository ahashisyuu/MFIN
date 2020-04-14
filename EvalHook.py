import os
import numpy as np
import tensorflow as tf

from tensorflow.python.training.training_util import _get_or_create_global_step_read as get_global_step
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer, CheckpointSaverHook
from tensorflow.python.training.session_run_hook import SessionRunArgs, SessionRunHook

from run_classifier import input_fn_builder
from utils import PRF, eval_reranker, print_metrics


class EvalHook(SessionRunHook):
    def __init__(self, estimator, dev_features, dev_label, dev_cid, max_seq_length, th=82.0, eval_steps=None,
                 checkpoint_dir=None, model_name=None, _input_fn_builder=None, tail_num=0, type_word=''):
        logging.info("Create EvalHook.")
        self.estimator = estimator
        self.dev_features = dev_features
        self.dev_label = dev_label
        self.dev_cid = dev_cid
        self.max_seq_length = max_seq_length
        self.th = th
        self._checkpoint_dir = checkpoint_dir
        if os.path.exists('./EVAL_LOG') is False:
            os.mkdir('./EVAL_LOG')
        self.model_name = model_name
        self.tail_num = tail_num
        self.org_dir = "CQA_" + type_word + self.model_name + "_{}".format(self.tail_num)

        self._log_save_path = os.path.join('./EVAL_LOG', model_name + '_' + type_word + '_log')
        self._save_path = checkpoint_dir
        if os.path.exists(self._save_path) is False:
            os.mkdir(self._save_path)
        self._timer = SecondOrStepTimer(every_steps=eval_steps)
        self._steps_per_run = 1
        self._global_step_tensor = None
        self._saver = None

        if _input_fn_builder is not None:
            self.input_fn_builder = _input_fn_builder
        else:
            self.input_fn_builder = input_fn_builder

    def _set_steps_per_run(self, steps_per_run):
        self._steps_per_run = steps_per_run

    def begin(self):
        # self._summary_writer = SummaryWriterCache.get(self._checkpoint_dir)
        self._global_step_tensor = get_global_step()  # pylint: disable=protected-access
        if self._global_step_tensor is None:
            raise RuntimeError(
                "Global step should be created to use CheckpointSaverHook.")

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        stale_global_step = run_values.results
        if self._timer.should_trigger_for_step(
                stale_global_step + self._steps_per_run):
            # get the real value after train op.
            global_step = run_context.session.run(self._global_step_tensor)
            if self._timer.should_trigger_for_step(global_step):
                self._timer.update_last_triggered_step(global_step)
                MAP, MRR = self.evaluation(global_step)
                # print("================", MAP, MRR, self.th, type(MAP), type(MRR), type(self.th))
                if MAP > self.th:
                    # print("================", MAP, MRR)
                    self._save(run_context.session, global_step, MAP, MRR)

    def end(self, session):
        last_step = session.run(self._global_step_tensor)
        if last_step != self._timer.last_triggered_step():
            MAP, MRR = self.evaluation(last_step)
            if MAP > self.th:
                self._save(session, last_step, MAP, MRR)

    def evaluation(self, global_step):
        eval_input_fn = self.input_fn_builder(
            features=self.dev_features,
            seq_length=self.max_seq_length,
            is_training=False,
            drop_remainder=False)

        predictions = self.estimator.predict(eval_input_fn, yield_single_examples=False)
        res = np.concatenate([a["prob"] for a in predictions], axis=0)

        metrics = PRF(np.array(self.dev_label), res.argmax(axis=-1))

        print('\n Global step is : ', global_step)
        MAP, AvgRec, MRR = eval_reranker(self.dev_cid, self.dev_label, res[:, 0])

        metrics['MAP'] = MAP
        metrics['AvgRec'] = AvgRec
        metrics['MRR'] = MRR

        metrics['global_step'] = global_step

        print_metrics(metrics, 'dev', save_dir=self._log_save_path)

        return MAP * 100, MRR

    def _save(self, session, step, map=None, mrr=None):
        """Saves the latest checkpoint, returns should_stop."""
        save_path = os.path.join(self._save_path, "step{}_MAP{:5.4f}_MRR{:5.4f}".format(step, map, mrr))

        list_name = os.listdir(self.org_dir)
        for name in list_name:
            if "model.ckpt-{}".format(step-1) in name:
                org_name = os.path.join(self.org_dir, name)
                tag_name = save_path + "." + name.split(".")[-1]
                print("save {} to {}".format(org_name, tag_name))
                with open(org_name, "rb") as fr, open(tag_name, 'wb') as fw:
                    fw.write(fr.read())






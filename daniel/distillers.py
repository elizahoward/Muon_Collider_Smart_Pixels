"""
Author: Claude chat
Date: September 25, 2026
Description:
distillers.py — Knowledge distillation for SmartPix models.

Two implementations:
  - OfflineDistiller   : teacher inference runs once; soft logits + hint
                         features are saved into augmented TFRecords, then
                         an OfflineStudentModel is trained with model.fit().
  - OnlineDistiller    : teacher runs every batch inside compute_loss().
                         Simpler setup, slower to train.

Both match existing codebase conventions:
  - generators yield (X_dict, y) keyed by feature name
  - models accept a feature-key dict as input (Keras routes by Input layer name)
  - TFRecords use tf.train.Example + tf.io.parse_tensor (same as ODG4)
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def build_extractor(model: keras.Model) -> keras.Model:
    """Sub-model that outputs (second_to_last_layer_activations, final_output)."""
    return keras.Model(
        inputs=model.input,
        outputs=[model.layers[-2].output, model.output],
        name=f"{model.name}_extractor",
    )


def check_hint_layer_compatibility(teacher: keras.Model, student: keras.Model):
    """
    Raise a clear error if the teacher and student second-to-last layers have
    different output shapes, since the hint loss (MSE) requires them to match.

    Call this before constructing OnlineDistiller or OfflineDistiller so you
    get a helpful message at setup time rather than a cryptic shape error
    mid-training.
    """
    teacher_shape = teacher.layers[-2].output.shape[1:]
    student_shape = student.layers[-2].output.shape[1:]
    if teacher_shape != student_shape:
        raise ValueError(
            f"Hint layer shape mismatch: teacher second-to-last layer has shape "
            f"{teacher_shape} but student has {student_shape}. "
            f"The hint loss requires identical shapes. Adjust the number of "
            f"units in the student's second-to-last layer to match the "
            f"teacher's ({teacher_shape[0]} units)."
        )


# ---------------------------------------------------------------------------
# OfflineStudentModel
# ---------------------------------------------------------------------------

class OfflineStudentModel(keras.Model):
    """
    A Keras Model wrapper around a student that reads pre-computed teacher
    targets from the input dict rather than running the teacher live.

    The input dict is expected to contain the student's normal feature keys
    plus 'teacher_logits' and 'teacher_feat', which OfflineDistiller writes
    into the augmented TFRecords. The student model itself never sees those
    extra keys — they are stripped before forwarding.

    Serializes cleanly with model.save() / tf.keras.models.load_model()
    because it is a top-level class with a standard get_config().
    """

    def __init__(
        self,
        student: keras.Model,
        # temperature: float = 3.0,
        # alpha: float = 0.5,
        # beta: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.student           = student
        # self.student_extractor = build_extractor(student)
        # self.temperature       = temperature #set in .compile()
        # self.alpha             = alpha
        # self.beta              = beta

        # Derive student input keys from its named Input layers so call() and
        # compute_loss() pass only the keys the student actually accepts,
        # regardless of what else is in the input dict (teacher keys, features
        # from other models, etc.).
        self.student_input_keys = {inp.name.split("/")[0] for inp in student.inputs}

        # loss function set by compile()
        self.student_loss_fn = None

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.hard_loss_tracker   = keras.metrics.Mean(name="hard_loss")
        self.distil_loss_tracker = keras.metrics.Mean(name="distil_loss")
        self.hint_loss_tracker   = keras.metrics.Mean(name="hint_loss")

    def compile(self, optimizer, student_loss_fn, alpha=0.5, beta=0.1, 
            temperature=3.0, **kwargs):
            super().compile(optimizer=optimizer, **kwargs)
            self.student_loss_fn = student_loss_fn
            self.alpha           = alpha
            self.beta            = beta
            self.temperature     = temperature
            self.student_extractor = build_extractor(self.student)

    def call(self, x):
        return self.student(x)

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None,
                     allow_empty=False):
        teacher_logits = x["teacher_logits"]
        teacher_feat   = x["teacher_feat"]

        student_x = {k: v for k, v in x.items() if k in self.student_input_keys}
        student_feat, _ = self.student_extractor(student_x, training=True)

        # Hard label loss: student predictions vs ground truth
        hard_loss = self.student_loss_fn(y, y_pred)

        # Distillation loss: BCE between temperature-softened sigmoid outputs.
        # teacher_logits is already a sigmoid value in (0, 1); dividing by t > 1
        # pushes it toward 0.5, producing softer targets. t=1 gives no softening.
        t = self.temperature
        # teacher_soft = tf.sigmoid(tf.math.log(teacher_logits / (1.0 - teacher_logits + 1e-7)) / t)
        # student_soft = tf.sigmoid(tf.math.log(
        #             tf.stop_gradient(y_pred) / (1.0 - tf.stop_gradient(y_pred) + 1e-7)
        #         ) / t)
        # distillation_loss = tf.reduce_mean(
        #     tf.keras.losses.binary_crossentropy(teacher_soft, student_soft)
        # )

        # MSE between logits (what the paper recommends)
        y_pred_clipped = tf.clip_by_value(tf.stop_gradient(y_pred), 1e-7, 1.0 - 1e-7)
        teacher_clipped = tf.clip_by_value(teacher_logits, 1e-7, 1.0 - 1e-7)

        teacher_logit = tf.math.log(teacher_clipped / (1.0 - teacher_clipped))
        student_logit = tf.math.log(y_pred_clipped / (1.0 - y_pred_clipped))
        distillation_loss = tf.reduce_mean(tf.square(teacher_logit - student_logit)) / 100

        hint_loss = tf.reduce_mean(tf.square(teacher_feat - student_feat))

        # tf.print("distil contribution:", self.alpha * distillation_loss, 
        #  "hint contribution:", self.beta * hint_loss)

        # tf.print("hard:", hard_loss, "distil:", distillation_loss, "hint:", hint_loss)
        # return hard_loss + 0.0 * hint_loss
        # return hard_loss + 0.0 * distillation_loss
        # return hard_loss + 0.0 * distillation_loss + 0.0 * hint_loss
        total_loss = (1.0 - self.alpha - self.beta) * hard_loss + self.alpha * distillation_loss + self.beta * hint_loss
        self.loss_tracker.update_state(total_loss)
        self.hard_loss_tracker.update_state(hard_loss)
        self.distil_loss_tracker.update_state(distillation_loss)
        self.hint_loss_tracker.update_state(hint_loss)
        return total_loss
        # return tf.experimental.numpy.nansum(losses)
    @property
    def metrics(self):
        return [self.loss_tracker, self.hard_loss_tracker,
                self.distil_loss_tracker, self.hint_loss_tracker] + super().metrics

    def get_config(self):
        base = super().get_config()
        base.update({
            "temperature": self.temperature,
            "alpha":       self.alpha,
            "beta":        self.beta,
        })
        return base


# ---------------------------------------------------------------------------
# OnlineDistiller
# ---------------------------------------------------------------------------

class OnlineDistiller(keras.Model):
    """
    Teacher runs every batch inside compute_loss().

    Follows the Keras knowledge distillation tutorial pattern, extended with
    a hint loss on the second-to-last layer.

    The generator yields (X_dict, y) exactly as in the rest of the codebase.
    Teacher and student each pick up their own input keys from the shared dict;
    Keras routes by Input layer name and ignores keys a model has no Input for.

    Usage:
        # --- Load teacher from a saved quantized model file ---
        filepath = "path/to/model_trial_095.h5"
        teacher_keras = Model_Classes.loadQuantizedModel(filepath)

        # --- Build student (quantized, matching teacher's input features) ---
        student_m3 = Model3(tfRecordFolder=TF_RECORD_DIR)
        student_m3.makeQuantizedModel()
        student_keras = student_m3.models["quantized_8w0i"]  # or whichever config

        # --- Load generators via the student model instance ---
        student_m3.loadTfRecords()
        training_generator   = student_m3.training_generator
        validation_generator = student_m3.validation_generator

        distiller = OnlineDistiller(student_keras, teacher_keras,
                                    temperature=3.0, alpha=0.5, beta=0.1)
        distiller.compile(
            optimizer=Adam(1e-3),
            student_loss_fn=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()],
        )
        distiller.fit(training_generator, validation_data=validation_generator,
                      epochs=50)
    """

    def __init__(
        self,
        student: keras.Model,
        teacher: keras.Model,
        temperature: float = 3.0,
        alpha: float = 0.5,
        beta: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.student     = student
        self.teacher     = teacher
        self.temperature = temperature
        self.alpha       = alpha
        self.beta        = beta

        self.teacher_extractor = build_extractor(teacher)
        self.student_extractor = build_extractor(student)
        check_hint_layer_compatibility(teacher, student)

        self.teacher_input_keys = {inp.name.split("/")[0] for inp in teacher.inputs}
        self.student_input_keys = {inp.name.split("/")[0] for inp in student.inputs}

        self.student_loss_fn = None

    def compile(self, optimizer, student_loss_fn, metrics=None, **kwargs):
        super().compile(optimizer=optimizer, metrics=metrics or [], **kwargs)
        self.student_loss_fn = student_loss_fn

    def call(self, x, training=False):
        student_x = {k: v for k, v in x.items() if k in self.student_input_keys}
        return self.student(student_x, training=training)

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None,
                     allow_empty=False):
        student_x    = {k: v for k, v in x.items() if k in self.student_input_keys}
        teacher_x    = {k: v for k, v in x.items() if k in self.teacher_input_keys}
        teacher_feat, teacher_logits = self.teacher_extractor(teacher_x, training=False)
        student_feat, _              = self.student_extractor(student_x, training=True)

        # Hard label loss: student predictions vs ground truth
        hard_loss = self.student_loss_fn(y, y_pred)

        # Distillation loss: BCE between temperature-softened sigmoid outputs.
        # teacher_logits is already a sigmoid value in (0, 1); dividing by t > 1
        # pushes it toward 0.5, producing softer targets. t=1 gives no softening.
        t = self.temperature
        # tryAlternate=True
        # if tryAlternate:
        #     # # Soften by blending with 0.5
        #     # teacher_soft = teacher_logits * (1/t) + 0.5 * (1 - 1/t)
        #     # student_soft = tf.stop_gradient(y_pred) * (1/t) + 0.5 * (1 - 1/t)
        #     # # distillation_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(teacher_soft, student_soft))
        #     # # Explicit numerically stable BCE avoiding Keras's internal implementation
        #     # student_soft_clipped = tf.clip_by_value(student_soft, 1e-7, 1.0 - 1e-7)
        #     # distillation_loss = -tf.reduce_mean(
        #     #     teacher_soft * tf.math.log(student_soft_clipped) 
        #     #     + (1.0 - teacher_soft) * tf.math.log(1.0 - student_soft_clipped)
        #     # )
        #     teacher_soft = teacher_logits * (1.0/t) + 0.5 * (1.0 - 1.0/t)
        #     student_soft = tf.stop_gradient(y_pred) * (1.0/t) + 0.5 * (1.0 - 1.0/t)
        #     distillation_loss = tf.reduce_mean(tf.square(teacher_soft - student_soft))
        # else:
        teacher_soft = tf.sigmoid(tf.math.log(teacher_logits / (1.0 - teacher_logits + 1e-7)) / t)
        student_soft = tf.sigmoid(tf.math.log(y_pred        / (1.0 - y_pred        + 1e-7)) / t)
        distillation_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(teacher_soft, student_soft)
        )

        hint_loss = tf.reduce_mean(tf.square(teacher_feat - student_feat))

        return (
            (1.0 - self.alpha) * hard_loss
            + self.alpha       * distillation_loss
            + self.beta        * hint_loss
        )

    def get_config(self):
        base = super().get_config()
        base.update({
            "temperature": self.temperature,
            "alpha":       self.alpha,
            "beta":        self.beta,
        })
        return base



# ---------------------------------------------------------------------------
# MDMM infrastructure
# ---------------------------------------------------------------------------
# Adapted from ArghyaRanjanDas/smart-pixels-ml softtimerouter branch
# (two_bit_optimization_helpers/mdmm.py). Changes from the original:
#   - Removed conditional_nll imports and bandit_update hooks
#   - Added needs_inputs flag so constraints can access the full input dict x
#     (needed for teacher_logits, teacher_feat, and student input features)
#   - Stripped SmartPixel-specific constraint classes (MinStd, MinMad, etc.)
# ---------------------------------------------------------------------------

class OutputConstraint(keras.layers.Layer):
    """Base class: a constraint on the model output with its own multiplier."""

    def __init__(self, scale=1.0, damping=1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale   = scale
        self.damping = damping
        self.lmbda   = self.add_weight(
            name=self.name + "_lmbda",
            shape=(),
            initializer="zeros",
            trainable=True,
        )

    def fn(self, outputs, **kwargs):
        raise NotImplementedError

    def infeasibility(self, fn_value):
        raise NotImplementedError

    def call(self, outputs, **kwargs):
        fn_value = self.fn(outputs, **kwargs) #loss computed standard, like hint_loss or distillation_loss
        inf      = self.infeasibility(fn_value) #how infeasible is the loss, e.g. how close to loss<max_loss satisfied
        l_term   = tf.math.maximum(self.lmbda, 0.0) * inf #self.lmbda is analogue to the alpha/beta coefficients, controlled by optimization in training
        damp_term = self.damping * tf.square(inf) / 2.0
        return self.scale * (l_term + damp_term), fn_value


class MDMM(keras.Model):
    """Wraps a model; adds OutputConstraint penalties to the training loss.

    train_step: loss = task_loss + sum(constraint penalties). The gradient
    sign is FLIPPED for lambda variables (ascent) so each multiplier grows
    while its constraint is violated and relaxes once satisfied.
    val_loss stays the plain compiled loss (test_step untouched) so it
    remains directly comparable to non-MDMM runs.

    constraint_pass options:
      'primary'      : penalties computed on the training=True y_pred of the
                       main pass. Right for distillation constraints where
                       gradients need to flow through quantizer STE.
      'deterministic': a second training=False forward pass. Right for spread
                       constraints where dropout noise inflates measurements.

    Constraints signal their input requirements via class attributes:
      needs_truth  = True : constraint.call receives y_true=y
      needs_inputs = True : constraint.call receives x=x (full input dict)
    """

    def __init__(self, model, constraints, constraint_samples=None,
                 constraint_pass='primary', name='MDMM', **kwargs):
        super().__init__(name=name, **kwargs)
        assert constraint_pass in ('deterministic', 'primary'), constraint_pass
        self.model              = model
        self.constraints_list   = list(constraints)
        self._lmbda_ids         = {id(c.lmbda) for c in self.constraints_list}
        self.constraint_samples = constraint_samples
        self.constraint_pass    = constraint_pass
        self.constraint_fn_trackers = {
            c.name: keras.metrics.Mean(name=c.name + "_fn")
            for c in self.constraints_list
        }

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred   = self.model(x, training=True)
            loss_obj = self.compute_loss(x=x, y=y, y_pred=y_pred)

            # Slice for constraint evaluation if constraint_samples is set
            n        = self.constraint_samples
            x_c      = x if n is None else {k: v[:n] for k, v in x.items()}
            y_c      = y if n is None else y[:n]

            if self.constraint_pass == 'primary':
                y_det = y_pred if n is None else y_pred[:n]
            else:
                y_det = self.model(x_c, training=False)

            penalties = {}
            for c in self.constraints_list:
                kwargs = {}
                if getattr(c, "needs_truth",  False):
                    kwargs["y_true"] = y_c
                if getattr(c, "needs_inputs", False):
                    kwargs["x"] = x_c
                penalty, fn_val = c(y_det, **kwargs)
                self.constraint_fn_trackers[c.name].update_state(fn_val)
                penalties["pen_" + c.name] = penalty

            loss = loss_obj + tf.add_n(list(penalties.values()))

        grads = tape.gradient(loss, self.trainable_variables)
        grads_and_vars = []
        for grad, var in zip(grads, self.trainable_variables):
            if grad is None:
                continue
            if id(var) in self._lmbda_ids:
                grads_and_vars.append((-grad, var))   # ascent for lambdas
            else:
                grads_and_vars.append((grad, var))
        self.optimizer.apply_gradients(grads_and_vars)

        self.compiled_metrics.update_state(y, y_pred)

        out = {"loss": loss, "loss_obj": loss_obj}
        out.update(penalties)
        # tf.print(self.compiled_metrics.metrics,self.metrics)
        out.update({m.name: m.result() for m in self.compiled_metrics.metrics})
        out.update({c.name + "_fnLoss": self.constraint_fn_trackers[c.name].result() 
            for c in self.constraints_list})
        # out.update({m.name: m.result() for m in self.metrics})
        return out

    # Delegate save/load/summary to the inner model so existing callbacks
    # and checkpointing work unchanged
    def save_weights(self, filepath, *args, **kwargs):
        self.model.save_weights(filepath, *args, **kwargs)

    def load_weights(self, filepath, *args, **kwargs):
        self.model.load_weights(filepath, *args, **kwargs)

    def summary(self, *args, **kwargs):
        return self.model.summary(*args, **kwargs)
    
    @property
    def metrics(self):
        return list(self.constraint_fn_trackers.values()) + super().metrics

    def test_step(self, data):
        x, y = data
        y_pred = self.model(x, training=False)
        loss_obj = self.compute_loss(x=x, y=y, y_pred=y_pred)

        n   = self.constraint_samples
        x_c = x if n is None else {k: v[:n] for k, v in x.items()}
        y_c = y if n is None else y[:n]

        for c in self.constraints_list:
            kwargs = {}
            if getattr(c, "needs_truth",  False):
                kwargs["y_true"] = y_c
            if getattr(c, "needs_inputs", False):
                kwargs["x"] = x_c
            _, fn_val = c(y_pred if n is None else y_pred[:n], **kwargs)
            self.constraint_fn_trackers[c.name].update_state(fn_val)

        self.compiled_metrics.update_state(y, y_pred)
        out = {"loss": loss_obj}
        out.update({m.name: m.result() for m in self.compiled_metrics.metrics})
        out.update({c.name + "_fnLoss": self.constraint_fn_trackers[c.name].result()
                    for c in self.constraints_list})
        return out


# ---------------------------------------------------------------------------
# Distillation constraints
# ---------------------------------------------------------------------------

class DistillationConstraint(OutputConstraint):
    """Logit-space MSE between student and teacher outputs <= max_value.

    Implements the logit-matching approach from Comparing KL Divergence and
    MSE Loss in Knowledge Distillation (arxiv 2105.08919), scaled by /100 to
    bring it to the same order of magnitude as BCE hard loss (~0.5-1.0).

    Override fn() to experiment with different distillation loss functions,
    e.g. BCE on sigmoid outputs or linear-blend temperature scaling.
    """
    needs_inputs = True

    def __init__(self, max_value=0.3, scale=1.0, damping=1.0, **kwargs):
        super().__init__(scale=scale, damping=damping, **kwargs)
        self.max_value = max_value

    def fn(self, outputs, x=None, **kwargs):
        teacher_clipped = tf.clip_by_value(x["teacher_logits"], 1e-7, 1.0 - 1e-7)
        student_clipped = tf.clip_by_value(tf.stop_gradient(outputs), 1e-7, 1.0 - 1e-7)
        teacher_logit   = tf.math.log(teacher_clipped / (1.0 - teacher_clipped))
        student_logit   = tf.math.log(student_clipped / (1.0 - student_clipped))
        return tf.reduce_mean(tf.square(teacher_logit - student_logit)) / 100.0

    def infeasibility(self, fn_value):
        return tf.math.maximum(fn_value - self.max_value, 0.0)


class HintConstraint(OutputConstraint):
    """MSE between teacher and student second-to-last layer activations <= max_value.

    Requires hint layer dimensions to match (enforced by
    check_hint_layer_compatibility at OfflineDistiller construction time).

    Override fn() to experiment with different feature-matching losses,
    e.g. cosine similarity instead of MSE.
    """
    needs_inputs = True

    def __init__(self, student_extractor, student_input_keys, max_value=0.6,
                 scale=1.0, damping=1.0, **kwargs):
        super().__init__(scale=scale, damping=damping, **kwargs)
        self.student_extractor  = student_extractor
        self.student_input_keys = student_input_keys
        self.max_value          = max_value

    def fn(self, outputs, x=None, **kwargs):
        teacher_feat = x["teacher_feat"]
        student_x    = {k: v for k, v in x.items() if k in self.student_input_keys}
        student_feat, _ = self.student_extractor(student_x, training=True)
        return tf.reduce_mean(tf.square(teacher_feat - student_feat))

    def infeasibility(self, fn_value):
        return tf.math.maximum(fn_value - self.max_value, 0.0)


# ---------------------------------------------------------------------------
# end of MDMM infrastructure
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# OfflineDistiller
# ---------------------------------------------------------------------------

class OfflineDistiller:
    """
    Runs teacher inference once, saves augmented TFRecords, then hands off
    to OfflineStudentModel for training.

    Step 1 - extract_and_save_teacher_outputs():
        Iterates the generator, runs the teacher extractor, and writes new
        TFRecords containing all original feature keys plus:
            'teacher_logits'  - soft targets from the teacher's output layer
            'teacher_feat'    - activations from the teacher's second-to-last layer

    Step 2 - make a dummy Model1 (or whichever SmartPixModel subclass matches
        the student's features), extend its x_feature_description with
        'teacher_logits' and 'teacher_feat', point it at the augmented TFRecord
        directory, and call loadTfRecords(). The generator's _parse_tfrecord_fn
        builds its feature description dynamically from x_feature_description,
        so those keys are picked up automatically alongside the student's
        normal inputs.

    Usage:
        # --- Step 0: load teacher from a saved quantized model file ---
        tfRecordFolder="/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026V4_June/TF_Records/filtering_records16384_data_shuffled_single_bigData_normalized/"
        filepath = "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/Results_June2026_99SigEff/model3_fin_results/model3_10bit_normalised_selected/pareto_primary/FailedTrialsRetryIfCoureageous/model_trial_095.h5"
        teacher = Model_Classes.loadQuantizedModel(filepath)
        def numOutNodes(model):
            outputLayerNodes = model.layers[-2].input_spec.axes[-1]
            return outputLayerNodes;

        # --- Step 0: build student (quantized, matching teacher's input features) ---
        student = constructModel()

        # --- Step 1: run teacher once, write augmented TFRecords ---
        # Load generators first — the teacher's input features drive what goes
        # into the TFRecords, so use a model instance that matches the teacher.
        d = distillers.OfflineDistiller(teacher, student,
                                    temperature=3.0, alpha=0.5, beta=0.1)

        model1Dummy = Model1(tfRecordFolder = tfRecordFolder)     
        model3Dummy = Model3(tfRecordFolder = tfRecordFolder)    
        model1Dummy.x_feature_description = model1Dummy.x_feature_description + ["z_global"] + model3Dummy.x_feature_description + ["nPix"]
        model1Dummy.loadTfRecords()
        odgTrain = model1Dummy.training_generator
        odgTest = model1Dummy.validation_generator

        augTfRecordDir = "./augRecords"
        augTrainDir = f"{augTfRecordDir}/tfrecords_train/"
        augValDir = f"{augTfRecordDir}/tfrecords_validation/"

        # Step 1.5: actually write the new records
        regenerateRecords=False
        if regenerateRecords:
            d.extract_and_save_teacher_outputs(
                training_generator=odgTrain,
                output_dir=augTrainDir,
                validation_generator=odgTest,
                val_output_dir=augValDir,
            )

        # --- Step 2: reload generators from augmented TFRecords ---
        # Extend the feature list with teacher keys, point at the new directory,
        # and call loadTfRecords() — same pattern as getNpixYtest/getPredVarDF
        # in tfLoaderUtils.py.
        sys.path.append("../MuC_Smartpix_Data_Production/tfRecords")
        import OptimizedDataGenerator4_data_shuffled_bigData_NewFormat as ODG2
        aug_train_gen = ODG2.OptimizedDataGeneratorDataShuffledBigData(
            load_records=True,
            tf_records_dir=augTrainDir,
            x_feature_description=model1Dummy.x_feature_description + ["teacher_logits", "teacher_feat"],
            batch_size=16384,
        )
        aug_val_gen = ODG2.OptimizedDataGeneratorDataShuffledBigData(
            load_records=True,
            tf_records_dir=augValDir,
            x_feature_description=model1Dummy.x_feature_description + ["teacher_logits", "teacher_feat"],
            batch_size=16384,
        )

        # --- Step 3: compile and train the student wrapper ---
        model = d.build_student_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            student_loss_fn=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()],
            run_eagerly=True,  # required for QKeras models
        )
        model.fit(aug_train_gen,
                validation_data=aug_val_gen,
                epochs=2)
    """

    def __init__(
        self,
        teacher: keras.Model,
        student: keras.Model,
        # temperature: float = 3.0,
        # alpha: float = 0.5,
        # beta: float = 0.1,
    ):
        self.teacher     = teacher
        self.student     = student
        # self.temperature = temperature
        # self.alpha       = alpha
        # self.beta        = beta

        self.teacher_extractor = build_extractor(teacher)
        check_hint_layer_compatibility(teacher, student)

    # ------------------------------------------------------------------
    # Step 1
    # ------------------------------------------------------------------

    def extract_and_save_teacher_outputs(
        self,
        training_generator,
        output_dir: str,
        validation_generator,
        val_output_dir: str,
    ):
        print("=== OfflineDistiller: extracting teacher outputs (train) ===")
        self._write_augmented_records(training_generator, output_dir)

        print("=== OfflineDistiller: extracting teacher outputs (val) ===")
        self._write_augmented_records(validation_generator, val_output_dir)

    def _write_augmented_records(self, generator, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        n_batches = len(generator)

        for batch_idx in range(n_batches):
            X_batch, y_batch = generator[batch_idx]
            teacher_feat, teacher_logits = self.teacher_extractor(
                X_batch, training=False
            )

            # Serialize the whole batch tensor at once — one Example per file,
            # matching the format ODG4 writes and reads (tf.io.parse_tensor on
            # the full batch). This avoids a per-sample Python loop over
            # batch_size (typically 16384) iterations.
            feature = {"y": self._bytes_feature(tf.io.serialize_tensor(y_batch))}
            for key, tensor in X_batch.items():
                feature[key] = self._bytes_feature(tf.io.serialize_tensor(tensor))
            feature["teacher_logits"] = self._bytes_feature(
                tf.io.serialize_tensor(teacher_logits)
            )
            feature["teacher_feat"] = self._bytes_feature(
                tf.io.serialize_tensor(teacher_feat)
            )

            writer = tf.io.TFRecordWriter(
                os.path.join(output_dir, f"batch_{batch_idx:05d}.tfrecord")
            )
            writer.write(
                tf.train.Example(
                    features=tf.train.Features(feature=feature)
                ).SerializeToString()
            )
            writer.close()

            if batch_idx % 10 == 0:
                print(f"  Written {batch_idx + 1}/{n_batches} batches")

        print(f"  Done. Augmented TFRecords saved to: {output_dir}")

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[value.numpy()])
        )

    # ------------------------------------------------------------------
    # Step 2
    # ------------------------------------------------------------------

    def build_student_model(self) -> OfflineStudentModel:
        return OfflineStudentModel(
            student=self.student,
            # temperature=self.temperature,
            # alpha=self.alpha,
            # beta=self.beta,
        )

    def build_mdmm_model(
        self,
        distil_max_value: float = 0.3,
        hint_max_value: float = None,
        scaleHint: float = 1.0,
        scaleDistil: float = 1.0,
        dampingHint: float = 1.0,
        dampingDistil: float = 1.0,
        constraint_pass: str = 'primary',
    ) -> MDMM:
        """Wrap the student in an MDMM model with distillation constraints.

        The student minimizes hard_loss (BCE vs ground truth) subject to:
          - distillation_loss <= distil_max_value  (always added)
          - hint_loss <= hint_max_value            (added only if hint_max_value is not None)

        Lagrange multipliers learn the constraint weights automatically,
        replacing the manual alpha/beta hyperparameters of build_student_model().

        Starting values based on observed loss magnitudes (logit MSE / 100):
          distil_max_value = 0.3  (distil_loss starts ~1.0, decreases to ~0.6)
          hint_max_value   = 0.6  (hint_loss starts ~3.5, decreases to ~2.0)

        To experiment with different distillation or hint loss functions,
        subclass DistillationConstraint or HintConstraint and override fn().

        Usage:
            mdmm_model = d.build_mdmm_model(distil_max_value=0.3,
                                             hint_max_value=0.6)
            mdmm_model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-3),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy()],
            )
            mdmm_model.fit(aug_train_gen, validation_data=aug_val_gen,
                           epochs=50)
            # Save the inner student (plain QKeras model, loadable by
            # Model_Classes.loadQuantizedModel):
            mdmm_model.model.save("model.h5")
        """
        constraints = [
            DistillationConstraint(
                max_value=distil_max_value,
                scale=scaleDistil,
                damping=dampingDistil,
            )
        ]
        if hint_max_value is not None:
            constraints.append(
                HintConstraint(
                    student_extractor=build_extractor(self.student),
                    student_input_keys={
                        inp.name.split("/")[0] for inp in self.student.inputs
                    },
                    max_value=hint_max_value,
                    scale=scaleHint,
                    damping=dampingHint,
                )
            )
        return MDMM(
            model=self.student,
            constraints=constraints,
            constraint_pass=constraint_pass,
        )

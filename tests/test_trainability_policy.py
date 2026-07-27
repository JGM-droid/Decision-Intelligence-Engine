from __future__ import annotations

import pytest


def _toy_backbone(tf):
    inputs = tf.keras.Input(shape=(8, 8, 3))
    x = tf.keras.layers.Conv2D(8, 3, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(8, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.ReLU()(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def test_frozen_policy_keeps_all_backbone_layers_non_trainable() -> None:
    tf = pytest.importorskip("tensorflow")
    from src.decision_intelligence_engine.baseline_training import _apply_backbone_trainability

    backbone = _toy_backbone(tf)
    trainable_layers, total_layers = _apply_backbone_trainability(
        backbone=backbone,
        freeze_backbone=True,
        unfreeze_last_n_layers=0,
        unfreeze_last_fraction=None,
        train_batch_norm=False,
    )

    assert total_layers == len(backbone.layers)
    assert trainable_layers == 0
    assert all(layer.trainable is False for layer in backbone.layers)


def test_partial_unfreeze_keeps_batch_norm_frozen() -> None:
    tf = pytest.importorskip("tensorflow")
    from src.decision_intelligence_engine.baseline_training import _apply_backbone_trainability

    backbone = _toy_backbone(tf)
    trainable_layers, _ = _apply_backbone_trainability(
        backbone=backbone,
        freeze_backbone=False,
        unfreeze_last_n_layers=3,
        unfreeze_last_fraction=None,
        train_batch_norm=False,
    )

    bn_layers = [layer for layer in backbone.layers if isinstance(layer, tf.keras.layers.BatchNormalization)]
    assert bn_layers
    assert all(layer.trainable is False for layer in bn_layers)
    assert trainable_layers > 0

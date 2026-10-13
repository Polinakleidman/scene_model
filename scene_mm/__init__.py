"""Multimodal SCENE (RNA + protein) with likelihood variants and fold-in of new cells."""
from .model import MultiSCENE, ModalitySpec
from .train import Config, fit, export

from .eeg_classifier import EEGClassifier20CNN, extract_eeg_classifier20_arch_metadata
from .eeg_encoder import EEGEncoderCNN, extract_eeg_encoder_cnn_arch_metadata

__all__ = [
    "EEGClassifier20CNN",
    "EEGEncoderCNN",
    "extract_eeg_classifier20_arch_metadata",
    "extract_eeg_encoder_cnn_arch_metadata",
]

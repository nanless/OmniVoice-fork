---
datasets:
- k2-fsa/TTS_eval_datasets
language:
- en
- zh
license: apache-2.0
pipeline_tag: text-to-speech
library_name: transformers
---

# TTS Evaluation Models

This repository contains models for the objective evaluation of text-to-speech (TTS) models, as presented in the papers [ZipVoice: Fast and High-Quality Zero-Shot Text-to-Speech with Flow Matching](https://huggingface.co/papers/2506.13053), [ZipVoice-Dialog: Non-Autoregressive Spoken Dialogue Generation with Flow Matching](https://huggingface.co/papers/2507.09318), [OmniVoice: Towards Omnilingual Zero-Shot Text-to-Speech with Diffusion Language Models](https://huggingface.co/papers/2604.00688).

- **Code:** [k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice) and [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice)


## Evaluation Metrics

This repository specifically supports the following evaluation metrics:

- **WER**: Includes [Hubert-based ASR model](https://huggingface.co/facebook/hubert-large-ls960-ft) for LibriSpeech-PC testset, [Paraformer-based ASR model](https://huggingface.co/funasr/paraformer-zh) for Chinese datasets, [Whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) model for general English and other languages test sets, [WhisperD](https://huggingface.co/jordand/whisper-d-v1a) model for English dialogue speech.

- **cpWER**: [WhisperD](https://huggingface.co/jordand/whisper-d-v1a) model is used to compute concatenated minimum permutation word error rate ([cpWER](https://arxiv.org/abs/2507.09318)) for English dialogue speech.

- **SIM-o**: A [wavlm-based speaker verification model](https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification) is used to compute the speaker similarity between prompt and generated speech.

- **cpSIM**: A [speaker diarization model](https://huggingface.co/pyannote/speaker-diarization-3.1) is used along with the above wavlm-based model to compute concatenated maximum permutation speaker similarity ([cpSIM](https://arxiv.org/abs/2507.09318)).

- **UTMOS**: The mos prediction model [UTMOS](https://github.com/sarulab-speech/UTMOS22) is used.

For more details, please refer to repositories [ZipVoice](https://github.com/k2-fsa/ZipVoice) and [OmniVoice](https://github.com/k2-fsa/OmniVoice).

## Citation

```bibtex
@article{zhu2025zipvoice,
      title={ZipVoice: Fast and High-Quality Zero-Shot Text-to-Speech with Flow Matching},
      author={Zhu, Han and Kang, Wei and Yao, Zengwei and Guo, Liyong and Kuang, Fangjun and Li, Zhaoqing and Zhuang, Weiji and Lin, Long and Povey, Daniel},
      journal={arXiv preprint arXiv:2506.13053},
      year={2025}
}

@article{zhu2025zipvoicedialog,
      title={ZipVoice-Dialog: Non-Autoregressive Spoken Dialogue Generation with Flow Matching},
      author={Zhu, Han and Kang, Wei and Guo, Liyong and Yao, Zengwei and Kuang, Fangjun and Zhuang, Weiji and Li, Zhaoqing and Han, Zhifeng and Zhang, Dong and Zhang, Xin and Song, Xingchen and Lin, Long and Povey, Daniel},
      journal={arXiv preprint arXiv:2507.09318},
      year={2025}
}

@article{zhu2026omnivoice,
      title={OmniVoice: Towards Omnilingual Zero-Shot Text-to-Speech with Diffusion Language Models},
      author={Zhu, Han and Ye, Lingxuan and Kang, Wei and Yao, Zengwei and Guo, Liyong and Kuang, Fangjun and Han, Zhifeng and Zhuang, Weiji and Lin, Long and Povey, Daniel},
      journal={arXiv preprint arXiv:2604.00688},
      year={2026}
}

```
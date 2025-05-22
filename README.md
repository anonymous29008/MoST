# MoST: Efficient Speech-Text Foundation Model with Modality-Aware Mixture of Experts
## Overview
This repository contains the implementation of MoST (Mixture of Speech and Text), a multimodal foundation model built upon DeepSeekV2's architecture. MoST extends the standard Mixture of Experts (MoE) architecture to better support multimodal inputs, particularly text and audio.

## Modality-Aware Mixture of Experts (MAMoE)

### What is MAMoE?
Modality-Aware Mixture of Experts (MAMoE) is an enhanced routing mechanism that directs tokens to modality-specific experts based on the token's modality type. Unlike traditional MoE where all tokens compete for all experts, MAMoE ensures that:

- Text tokens are only routed to text-specific experts
- Audio tokens are only routed to audio-specific experts
- Shared experts (if configured) remain accessible to all modalities

This architecture ensures that experts can specialize in processing specific modalities, improving overall model performance for multimodal tasks.

### How it Works
1. Each token is tagged with a modality mark:
   - 0: Text tokens (token IDs 0-100001)
   - 1: Audio tokens (token IDs 100002-100503)
   - 2: Image tokens (token IDs 100503-102400)

2. The routing mechanism (`MAMoEGate`) filters the gating scores so that:
   - Text tokens can only select experts from the text expert pool
   - Audio tokens can only select experts from the audio expert pool

3. The top-k selection and rest of the MoE forward pass proceeds as usual, but with modality-constrained expert selection

### How to Use MAMoE
To enable Modality-Aware Mixture of Experts, configure your model with the following parameters:

```python
from transformers import MoSTConfig, MoSTForCausalLM

# Configure with MAMoE
config = MoSTConfig(
    # Standard MoE parameters
    n_shared_experts=0,  # Number of shared experts (accessible to all modalities)
    n_routed_experts=8,  # Total number of routed experts
    num_experts_per_tok=2,  # Number of experts to select per token
    
    # MAMoE specific parameters
    use_modality_aware_routing=True,  # Enable modality-aware routing
    text_expert_indices=[0, 1, 2, 3],  # First 4 experts are for text
    audio_expert_indices=[4, 5, 6, 7],  # Last 4 experts are for audio
)

# Initialize model with MAMoE configuration
model = MoSTForCausalLM(config)
```

### Advantages of MAMoE
1. **Modality Specialization**: Experts can focus on processing patterns specific to a single modality
2. **Balanced Training**: Ensures both modalities receive equal expert capacity
3. **Flexible Configuration**: The division of experts between modalities can be adjusted based on task requirements

### Token Modality Determination
Token modalities are determined based on token ID ranges:
- 0-100001: Text tokens
- 100002-100503: Audio tokens
- 100503-102400: Image tokens

This identification happens automatically during the forward pass based on the input token IDs.

## Building MoST from Scratch

Follow these steps to initialize and train a MoST model starting from pre-trained DeepSeek-V2 Lite and HuBERT weights:

1.  **Convert HuBERT Weights:**
    The audio component of MoST relies on HuBERT features. If you have a pre-trained HuBERT model in the Fairseq format, you need to convert its weights first using the provided script.
    ```bash
    python convert_hubert.py --fairseq_path /path/to/your/hubert_base.pt --output_path /path/to/save/converted_hubert.pt
    ```
    Replace `/path/to/your/hubert_base.pt` with the path to your Fairseq HuBERT checkpoint and `/path/to/save/converted_hubert.pt` with your desired output location.

2.  **Initialize MoST with Pre-trained Weights:**
    Next, initialize the MoST model structure and load weights from both the pre-trained DeepSeek-V2 Lite model and the converted HuBERT model.
    *   Ensure your `config_mostc.json` file (or the config file you intend to use) points to the correct path for the converted HuBERT weights. This is typically handled during the initialization of the `AudioWaveProcessor` within the `MoSTCForCausalLM` model, based on its configuration.
    *   Run the `load_deepseek_weights.py` script. This script will:
        *   Load the DeepSeek-V2 Lite model specified by `deepseek_model_path` within the script.
        *   Load the HuBERT weights based on the path specified in the `config_mostc.json` used to initialize the `MoSTCForCausalLM` model.
        *   Combine these weights into a new MoST model instance.
        *   Save the initialized MoST model to the `save_path` specified in the script (e.g., `/path/to/save/MoSTC-initialized`).
    ```bash
    # NOTE: Modify deepseek_model_path and save_path inside load_deepseek_weights.py 
    #       or adapt the script to use command-line arguments before running.
    python load_deepseek_weights.py 
    ```
    *Note: Currently, paths are hardcoded in `load_deepseek_weights.py`. It's recommended to modify it to accept command-line arguments for `deepseek_model_path`, `config_path`, and `save_path` for better usability.*

3.  **Train the Initialized MoST Model:**
    Now that you have an initialized MoST model checkpoint, you can train it on your multimodal dataset using the `train/run_clm_no_trainer.py` script. This script leverages `accelerate` and DeepSpeed for efficient distributed training.
    ```bash
    # Example using accelerate for multi-GPU training with DeepSpeed ZeRO Stage 2
    # Create an accelerate config file first (e.g., using 'accelerate config')
    # Adjust parameters like paths, batch size, learning rate, etc. as needed.
    
    accelerate launch train/run_clm_no_trainer.py \\
        --model_name_or_path /path/to/your/initialized/MoSTC-initialized \\
        --output_dir /path/to/save/trained_most_model \\
        --train_asr_dirs /path/to/asr/train/data \\
        --train_tts_dirs /path/to/tts/train/data \\
        --val_asr_dirs /path/to/asr/validation/data \\
        --val_tts_dirs /path/to/tts/validation/data \\
        --per_device_train_batch_size 4 \\
        --per_device_eval_batch_size 4 \\
        --gradient_accumulation_steps 8 \\
        --learning_rate 1e-5 \\
        --num_train_epochs 3 \\
        --max_seq_length 4096 \\
        --bf16 \\
        --with_tracking \\
        --report_to wandb \\
        --checkpointing_steps 500 \\
        --evaluate_every 500 
    ```
    *   Replace `/path/to/your/initialized/MoSTC-initialized` with the actual path where you saved the model in step 2.
    *   Configure data paths (`--train_asr_dirs`, `--train_tts_dirs`, etc.) according to your dataset locations.
    *   Adjust training hyperparameters (`--learning_rate`, `--per_device_train_batch_size`, etc.) as needed for your setup.
    *   Ensure you have created a suitable `accelerate` configuration file. The script is set up to use DeepSpeed Stage 2 by default.

## Experiment Results

### ASR and TTS Performance

| Model          | LS-Clean ASR (WER % ↓) | LS-Clean TTS (CER % ↓) | LS-Other ASR (WER % ↓) | LS-Other TTS (CER % ↓) |
|----------------|------------------------|------------------------|------------------------|------------------------|
| **MoST**       | **4.6**              | **6.0**              | **10.5**             | **6.5**              |
| SpeechGPT      | 11.0                   | 13.2                   | 16.7                   | 14.5                   |
| AudioLM        | 9.5                    | 8.8                    | 12.0                   | 9.5                    |
| SpiritLM       | 6.0                    | 6.7                    | 11.0                   | 7.9                    |
| Moshi          | 5.5                    | 7.0                    | 12.0                   | **6.5**              |

Lower values are better. Best results are highlighted in **bold**.

### Audio Language Modeling Performance

| Model                | sWUGGY          | sBLIMP          | sTopic-StoryCloze | sStoryCloze     | Average         |
|----------------------|-----------------|-----------------|-------------------|-----------------|-----------------|
| AudioLM              | 71.50           | **64.70**       | -                 | -               | -               |
| SpeechGPT            | 51.82           | 49.75           | 60.13             | 53.13           | 53.71           |
| spiritLM             | 40.14           | 48.28           | 83.32             | 58.95           | 57.67           |
| Moshi                | 51.14           | 53.31           | 46.34             | 45.16           | 48.99           |
| **MoST (Ours)**      | **75.28**       | 60.35           | **83.64**         | **65.43**       | **71.18**       |

Accuracies are reported. Best results are highlighted in **bold**.

### Spoken Question Answering Performance

| Task                    | Model      | Score |
|-------------------------|------------|-------|
| Llama Q (S→S)           | **MoST**   | **62.6**  |
|                         | SpeechGPT  | 34.2  |
|                         | AudioLM    | 25.8  |
|                         | SpiritLM   | 45.1  |
|                         | Moshi      | 40.3  |
| Trivial QA (S→T)        | **MoST**   | **43.5**  |
|                         | SpeechGPT  | 22.1  |
|                         | AudioLM    | 15.7  |
|                         | SpiritLM   | 30.9  |
|                         | Moshi      | 28.4  |
| Trivial QA (S→S)        | **MoST**   | **32.1**  |
|                         | SpeechGPT  | 18.5  |
|                         | AudioLM    | 10.2  |
|                         | SpiritLM   | 24.6  |
|                         | Moshi      | 20.7  |

Scores are task-specific (e.g., accuracy). Best results for MoST are highlighted in **bold**. 

## Generated Audio Examples

Here are some audio examples generated by MoST:

| Example | Transcript                                                                                               | Audio Sample |
|---------|----------------------------------------------------------------------------------------------------------|--------------|
| EXP1    | I love to play Golf.                                                                                     | 

https://github.com/user-attachments/assets/a14e0a42-09db-4013-9f66-27761739270f

 |
| EXP2    | Today, we find ourselves at a critical point in the history of scientific breakthroughs.                 | <audio controls><source src="./asset/exp_2.wav" type="audio/wav"></audio> |
| EXP3    | But the rapid acceleration of AI's development has raised concerns about its other effects on the rest of our society. | <audio controls><source src="./asset/exp_3.wav" type="audio/wav"></audio> |
| EXP4    | China is a very large country and is the most populous country in the world.                             | <audio controls><source src="./asset/exp_4.wav" type="audio/wav"></audio> |



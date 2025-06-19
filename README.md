# COS760-ASR-Topic-Modeling-Group1

## Optimizing Topic Modeling from Speech: Evaluating ASR and Topic Model Combinations for Setswana Podcasts

**A Study on How ASR Transcription Quality Affects Topic Coherence in Low-Resource African Languages**

### Team Members (Group 1)
- **Leo** 
- **Tina** 
- **Ruan Carlinsky**

### Project Overview

This research investigates which combination of ASR model and topic modeling technique yields the most coherent topics when processing Setswana podcast transcripts. The study systematically evaluates how different ASR systems paired with topic modeling methods affect the coherence of topics extracted from African language speech data.

**Research Question:** Which combination of ASR model and topic modeling technique yields the most coherent topics for Setswana podcasts?

### Background & Motivation

Most ASR models perform poorly on African languages like Setswana, significantly affecting downstream tasks like topic modeling that depend on clean text. This project addresses a critical gap in African language NLP by being the first to explore how ASR transcription quality impacts topic coherence in low-resource African languages.

### Methodology

#### Three-Stage Approach:
1. **Data Preparation**: Audio standardization and preprocessing
2. **ASR Transcription**: Multiple ASR system evaluation
3. **Topic Modeling**: Comparative analysis of topic modeling techniques

#### ASR Systems Evaluated:
- **OpenAI Whisper**: Strong performance on low-resource languages and noisy audio
- **Lelapa AI**: Optimized specifically for African languages
- **wav2vec 2.0**: Self-supervised model robust to small datasets

#### Topic Modeling Techniques:
- **Latent Dirichlet Allocation (LDA)**: Traditional probabilistic approach
- **BERTopic**: Modern transformer-based approach with better noise resilience

### Datasets

**Primary Dataset:**
- Setswana COVID-19 podcasts from DSFSI Lab (University of Pretoria)

**Supplementary Datasets:**
- NCHLT Setswana Speech Corpus (56 hours, 3.7 GB, CC BY 3.0)
- OpenSLR SLR32 (729 MB, CC BY-SA 4.0)

### Project Structure

```
COS760-ASR-Topic-Modeling-Group1/
│
├── data/
│   ├── raw/                    # Original podcast audio files
│   ├── processed/              # Segmented and normalized audio
│   └── transcripts/            # ASR outputs organized by system
│       ├── whisper/
│       ├── lelapa/
│       └── wav2vec/
│
├── src/
│   ├── data_processing/        # Audio preprocessing scripts
│   ├── asr/                    # ASR system integrations
│   ├── topic_modeling/         # Topic modeling implementations
│   ├── evaluation/             # Analysis and evaluation scripts
│   └── visualization/          # Plotting and visualization utilities
│
├── results/
│   ├── topic_modeling/         # Topic models and coherence scores
│   ├── error_analysis/         # ASR error analysis results
│   └── visualizations/         # Generated plots and charts
│
├── notebooks/                  # Jupyter notebooks for exploration
│   ├── 1_data_exploration.ipynb
│   ├── 2_asr_comparison.ipynb
│   └── 3_topic_analysis.ipynb
│
├── README.md
├── requirements.txt
├── config.json
└── .gitignore
```

### Key Components

#### Data Processing (`src/data_processing/`)
- **prepare_dataset.py**: Segments audio and creates metadata
- Audio standardization (WAV, 16kHz format)
- 30-second segment extraction

#### ASR Integration (`src/asr/`)
- **whisper_transcriber.py**: OpenAI Whisper implementation
- **lelapa_transcriber.py**: Lelapa AI API integration  
- **wav2vec_transcriber.py**: wav2vec model integration

#### Topic Modeling (`src/topic_modeling/`)
- **bertopic_analyzer.py**: BERTopic implementation
- **lda_analyzer.py**: Latent Dirichlet Allocation implementation

#### Evaluation (`src/evaluation/`)
- **error_analysis.py**: ASR error analysis and impact assessment
- **coherence_metrics.py**: Topic coherence calculation (UMass, UCI, NPMI)

#### Visualization (`src/visualization/`)
- **topic_visualizer.py**: Topic model visualizations

### Success Metrics

**Primary Metrics:**
- **Topic Coherence**: UMass, UCI, and NPMI scores
- **Robustness**: Comparison of LDA vs BERTopic resilience to noisy transcripts

**Expected Outputs:**
- Ranked comparison of ASR + Topic Modeling combinations
- Correlation analysis between transcription quality and topic coherence
- Practical recommendations for African language NLP pipelines

### Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/COS760-ASR-Topic-Modeling-Group1.git
   cd COS760-ASR-Topic-Modeling-Group1
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Data Preparation:**
   ```bash
   python src/data_processing/audio_segmentation.py
   ```

2. **ASR Transcription:**
   ```bash
   python src/asr/whisper_transcription.py
   python notebooks/lelapa1.ipynb
   python src/asr/wav2vec_asr.py
   ```

3. **Topic Modeling:**
   ```bash
   python notebooks/Final_Topic_Modeling.ipynb
   ```

4. **Evaluation & Visualization:**
   ```bash
   python notebooks/Final_Topic_Modeling.ipynb
   ```

### Expected Challenges

- Small dataset size for African language processing
- High ASR error rates due to language-specific challenges
- Potential lack of gold-standard transcripts for validation
- Limited Setswana language resources for evaluation

### Contribution to Field

This study represents the first systematic investigation of how ASR quality affects topic modeling in African languages, providing:
- Practical guidance for African language NLP pipelines
- Evidence-based recommendations for ASR-topic modeling combinations
- Insights for low-resource language processing
- Advancement of African NLP research

### Dependencies

Key libraries include:
- `openai-whisper`
- `transformers`
- `bertopic`
- `gensim` (for LDA)
- `librosa` (audio processing)
- `scikit-learn`
- `numpy`
- `pandas`
- `matplotlib`

<!-- 
keywords: process mining, meeting analysis, shadow workflows, multimodal AI, conformance checking, 
BPMN, event log extraction, Whisper speech recognition, pose estimation, RTMPose, GPT-4o-mini, 
Sentence-BERT, SBERT, token-replay fitness, Declare constraints, POWL, civic governance, 
city council meetings, parliamentary compliance, Robert's Rules of Order, organizational routine theory,
Design Science Research, PM4Py, Streamlit, Python, NLP, computer vision, LLM abstraction,
deviance detection, process discovery, Inductive Miner, Technology Acceptance Model, TAM,
ostensive performative gap, Feldman Pentland, digital twin, meeting intelligence,
video to event log, unstructured data process mining, multimodal event extraction
-->

# Meeting Process Twin

**An end-to-end multimodal AI pipeline that transforms meeting video recordings into structured process models for automated conformance checking, shadow workflow detection, and parliamentary compliance analysis using process mining.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/frontend-Streamlit-FF4B4B.svg)]()
[![PM4Py](https://img.shields.io/badge/process%20mining-PM4Py-green.svg)]()
[![DESRIST 2026](https://img.shields.io/badge/paper-DESRIST%202026-orange.svg)]()

> **Core finding:** Formal agendas predict only 31% of actual meeting behavior. The other 69% are shadow workflows, overwhelmingly substantive policy discussions, not procedural noise.

---

## Problem Statement

Process mining assumes structured event logs from information systems (ERP, CRM, workflow engines). But meetings, the backbone of organizational decision-making, produce no event logs. They unfold through speech, gestures, and interpersonal dynamics that no system captures. This creates a blind spot: we can mine how a purchase order flows through SAP, but not how a city council actually deliberates policy.

The **Meeting Process Twin** closes this gap by constructing event logs from raw meeting video, then applying the full process mining toolkit: discovery, conformance, and enhancement.

---

## What This Project Does

Given a **meeting video** and its **published agenda**, the pipeline:

1. **Extracts multimodal events** from the recording (audio via Whisper, visual via RTMPose, NLP via regex + keyword rules)
2. **Fuses modalities** using temporal proximity rules (e.g., a hand-raise + "all in favor" within 5s = Confirmed Vote)
3. **Abstracts events** into formal activity labels using LLM sliding-window classification (GPT-4o-mini) or keyword rules
4. **Maps activities to the agenda** using Sentence-BERT cosine similarity (threshold *t* = 0.35)
5. **Performs conformance checking** via PM4Py token-replay fitness against a BPMN normative model
6. **Detects shadow workflows** (activities not matching any agenda item) and classifies them by deviance type
7. **Evaluates parliamentary compliance** by formalizing Robert's Rules of Order as Declare constraints
8. **Discovers structural patterns** in shadow activities using POWL (Partially Ordered Workflow Language)

The result: a structured governance report that distills hours of meeting video into minutes of actionable intelligence.

---

## Architecture

```
Meeting Video + Agenda
        |
        v
+-------------------+     +----------------------+     +---------------------+
| Phase 1:          |     | Phase 2:             |     | Phase 3:            |
| Multimodal        | --> | Semantic             | --> | Conformance         |
| Event Extraction  |     | Abstraction          |     | Checking            |
+-------------------+     +----------------------+     +---------------------+
| - Whisper (ASR)   |     | - GPT-4o-mini        |     | - BPMN discovery    |
| - RTMPose (pose)  |     |   sliding-window     |     |   (Inductive Miner) |
| - NLP keywords    |     |   classification     |     | - Token-replay      |
| - Temporal fusion |     | - SBERT mapping      |     |   fitness           |
|   (vote/motion)   |     |   (agenda matching)  |     | - Declare (Robert's |
+-------------------+     +----------------------+     |   Rules)            |
                                                       +---------------------+
                                                               |
                                                               v
                                                  +---------------------+
                                                  | Phase 4:            |
                                                  | Shadow Workflow     |
                                                  | Analysis            |
                                                  +---------------------+
                                                  | - Deviance taxonomy |
                                                  |   (5 categories)    |
                                                  | - POWL patterns     |
                                                  |   (4 structural)    |
                                                  | - Governance report |
                                                  +---------------------+
                                                               |
                                                               v
                                                  +---------------------+
                                                  | Streamlit Dashboard |
                                                  | - Colored BPMN      |
                                                  | - Meeting chapters  |
                                                  | - Report card       |
                                                  | - Shadow timeline   |
                                                  +---------------------+
```

---

## Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Speech-to-text | OpenAI Whisper (local or API) | Transcribe meeting audio to timestamped segments |
| Pose estimation | RTMPose via rtmlib (ONNX) | Detect hand raises, standing, voting gestures from video |
| NLP extraction | 30+ regex keyword rules | Identify parliamentary speech acts (motions, votes, roll call) |
| Event abstraction | GPT-4o-mini sliding window | Classify raw events into activity labels with formal/shadow distinction |
| Activity mapping | Sentence-BERT (all-MiniLM-L6-v2) | Map abstracted activities to agenda items via cosine similarity |
| Process discovery | PM4Py Inductive Miner | Discover BPMN models from event logs |
| Conformance checking | PM4Py token-replay + alignment | Compute fitness scores against BPMN reference models |
| Declarative compliance | Custom Declare engine | Check Robert's Rules constraints (existence, precedence, response, succession) |
| Structural analysis | POWL (Partially Ordered Workflow Language) | Discover shadow workflow patterns (isolated, concurrent, sequential, recurring) |
| Deviance classification | Rule-based taxonomy | Categorize shadows as benign, innovative, efficiency, disruptive, or violation |
| Frontend | Streamlit | Interactive dashboard with video playback, colored BPMN, report card |

---

## Evaluation Results

Evaluated on **54 city council meetings** from 4 U.S. cities (Alameda, Boston, Denver, Seattle), totaling **103.5 hours** of video and **26,526 extracted events**.

| Metric | Value |
|--------|-------|
| Mean deduplicated fitness | 0.314 |
| Shadow activity prevalence | 41.9% (mean per-meeting: 42.3%) |
| Dominant deviance category | Innovation (95.7%) |
| Source distribution | Audio 47.9%, Visual 40.6%, Fused 11.4% |
| Declare compliance | Bimodal: 11 A-grades, 42 F-grades |
| POWL shadow clusters | 1,013 (isolated 44.4%, concurrent 28.7%) |
| Expert focus group (TAM) | 4.09/5.00 across 4 domain experts |
| Golden dataset validation | 8 meetings, 9,348 human annotations |

### Per-City Breakdown

| City | Meetings | Mean Fitness | Shadow % | Agenda Cov. | Declare A-Grade |
|------|----------|-------------|----------|-------------|-----------------|
| Alameda | 3 | 0.248 | 21.1% | 40.1% | 0/3 |
| Boston | 11 | 0.714 | 55.6% | 84.6% | 2/11 |
| Denver | 15 | 0.150 | 39.5% | 27.9% | 1/15 |
| Seattle | 25 | 0.245 | 40.7% | 44.9% | 8/25 |

### Ablation Study

| Configuration | Fitness | Shadow % | Events |
|--------------|---------|----------|--------|
| Audio only | 0.322 | 35.2% | 12,720 |
| Visual only | 0.000 | 100% | 10,779 |
| Combined (full) | 0.314 | 41.9% | 26,526 |

Audio carries all agenda-matching signal. Visual events contribute exclusively shadow activities (gestures, posture changes). Multimodality adds completeness, not redundancy.

---

## Project Structure

### Core Pipeline

| File | Purpose |
|------|---------|
| `app.py` | Streamlit application entry point |
| `video_processor.py` | Phase 1: Audio extraction, Whisper transcription, RTMPose visual detection, NLP keywords, temporal fusion |
| `compliance_engine.py` | Phase 2-3: LLM sliding-window abstraction, SBERT mapping, PM4Py fitness calculation |
| `bpmn_gen.py` | BPMN generation (agenda-based normative model, Inductive Miner discovery, colored compliance overlay) |
| `keyword_rules.py` | 30+ council-specific NLP patterns for event extraction |

### Pipeline Package

| File | Purpose |
|------|---------|
| `pipeline/orchestrator.py` | Four-phase pipeline orchestration with formal/shadow partitioning |
| `pipeline/state.py` | Streamlit session state management |
| `pipeline/time_utils.py` | Timestamp parsing, formatting, and conversion utilities |

### UI Package

| File | Purpose |
|------|---------|
| `ui/bpmn_views.py` | Reference and discovered BPMN rendering with compliance overlay |
| `ui/metrics.py` | Dashboard metrics, report card generation, and Declare grading |
| `ui/sidebar.py` | Configuration sidebar (API keys, thresholds, model selection) |
| `ui/styles.py` | Custom CSS for chapter cards, shadow timeline, BPMN styling |

### Research & Analysis

| File | Purpose |
|------|---------|
| `test_pipeline.py` | Headless pipeline runner with caching for iterative testing |
| `run_no_api.py` | LLM-free pipeline variant (keyword rules + SBERT only, no API needed) |
| `batch_analyze.py` | Batch processing for the full 54-meeting corpus |
| `param_sweep.py` | Grid search over abstraction parameters (81+ combinations) |
| `ablation_study.py` | Modality ablation analysis (audio-only, visual-only, combined) |
| `sbert_sensitivity.py` | SBERT threshold sensitivity analysis (0.15-0.60, 10 levels) |
| `golden_comparison.py` | Ground-truth validation against 9,348 human annotations |
| `generate_figures.py` | Generates all thesis figures from conformance.json results |
| `stratified_analysis.py` | Fitness stratified by agenda complexity tiers |
| `rerun_deviance.py` | Re-runs deviance classification without API calls |

### Research Modules

| Directory | Purpose |
|-----------|---------|
| `research/declare/` | Declare constraint formalization of Robert's Rules, conformance checking, violation analysis |
| `research/powl/` | POWL-based shadow pattern discovery (isolated, concurrent, sequential, recurring) |

---

## Three Abstraction Strategies

| Strategy | Method | Dedup Fitness | API Required | Best For |
|----------|--------|---------------|--------------|----------|
| **A** | Keyword rules + SBERT | 68.2% | No | Offline / free deployments |
| **B** | LLM sliding-window (GPT-4o-mini) | 72.7% | Yes | Maximum single-meeting accuracy |
| **C** | LLM + SBERT (batch) | 31.4% (corpus mean) | Yes | Large-scale corpus analysis |

---

## Installation

### Prerequisites
- Python 3.10 or 3.11
- FFmpeg (for audio extraction from video)
- OpenAI API key (for LLM abstraction; optional if using Strategy A)

### Setup

```bash
git clone https://github.com/azizketata/meeting-process-twin.git
cd meeting-process-twin
pip install -r requirements.txt
```

No GPU required. The pipeline runs on CPU using ONNX Runtime for pose estimation and Whisper small for transcription. GPU accelerates Whisper and RTMPose if available (CUDA).

---

## Usage

### Interactive Dashboard (Streamlit)

```bash
streamlit run app.py
```

Upload a meeting video and agenda, configure parameters in the sidebar, and click "Process Video." The dashboard provides:
- Meeting chapters with timestamped video playback links
- Colored BPMN overlay (green = executed, gray = skipped, orange = shadow)
- Compliance report card with fitness score and Declare grade
- Shadow workflow timeline with deviance classification

### Headless Pipeline (CLI)

```bash
# Full pipeline (Whisper + LLM abstraction)
python test_pipeline.py --api-key sk-... \
    --video meeting.mp4 --agenda agenda.txt --output-dir ./results

# Skip video processing (reuse cached transcription)
python test_pipeline.py --api-key sk-... \
    --video meeting.mp4 --agenda agenda.txt --output-dir ./results \
    --skip-video-processing --window-seconds 60 --overlap-ratio 0.5

# LLM-free mode (no API key needed)
python run_no_api.py --sweep
```

### Batch Processing

```bash
# Process full meeting corpus
python batch_analyze.py --api-key sk-...

# Generate thesis figures from results
python generate_figures.py

# Run SBERT sensitivity analysis
python sbert_sensitivity.py

# Validate against golden dataset
python golden_comparison.py
```

---

## Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `window_seconds` | 60 | LLM classification window size (try 45-120) |
| `overlap_ratio` | 0.5 | Window overlap ratio (higher = more context, slower) |
| `sbert_threshold` | 0.35 | Cosine similarity threshold for formal/shadow partition |
| `sbert_model` | all-MiniLM-L6-v2 | SBERT model (use all-mpnet-base-v2 for higher accuracy) |
| `min_events_per_window` | 5 | Minimum events to classify a window |
| `min_label_support` | 1 | Minimum occurrences to retain an activity label |
| `openai_abstraction_model` | gpt-4o-mini | LLM model for semantic event classification |

---

## Computational Requirements

Tested on: Windows 11, Intel Core i7, 32 GB RAM, no GPU required.

| Phase | Time (1h meeting) | Share |
|-------|-------------------|-------|
| Whisper transcription (CPU) | ~12 min | 60% |
| RTMPose visual extraction | ~4 min | 20% |
| LLM abstraction (API) | ~2 min | 10% |
| Audio extraction + NLP + fusion | ~2 min | 10% |
| **Total** | **~20 min** | |

Full 54-meeting corpus (103.5 hours): ~29.5 hours on CPU.

---

## Research Context

This project is the artifact of a **Master's thesis** at the **Technical University of Munich (TUM)**, supervised at the **SAP University Competence Center (UCC)**. It follows the **Design Science Research (DSR)** methodology and has been accepted at **DESRIST 2026** as a research-in-progress paper.

### Theoretical Foundations

- **Organizational Routine Theory** (Feldman & Pentland, 2003): Ostensive (formal) vs. performative (enacted) routines
- **Process Mining** (van der Aalst, 2016): Discovery, conformance checking, enhancement from event logs
- **Technology Acceptance Model** (Davis, 1989): PEOU, PU, ATU constructs for expert evaluation
- **Shadow Workflows** (Gavric et al., 2024): Informal processes invisible to traditional monitoring
- **Declare** (Pesic et al., 2007): Declarative temporal constraints for parliamentary compliance
- **POWL** (Kourani & van Zelst, 2023): Partially Ordered Workflow Language for structural pattern discovery
- **Robert's Rules of Order** (Robert, 2020): Parliamentary authority for deliberative assemblies

### Thesis Chapters

1. **Introduction** - Motivation, research gap, three research questions
2. **Theoretical Foundations** - BPM, process mining, multimodal fusion, shadow workflows, routine dynamics, Declare, POWL, civic governance
3. **Research Methodology** - DSR framework, 54-meeting corpus from 4 cities, golden dataset (10 meetings, 9,348 annotations), evaluation design
4. **Artifact Design** - Four-phase pipeline architecture with mathematical formalization (3 definitions, 1 algorithm, 7 equations)
5. **Evaluation** - RQ1 (multimodal extraction), RQ2 (conformance), RQ3 (shadow characterization), sensitivity analysis, ablation study, golden validation, TAM focus group
6. **Discussion** - Six key findings, implications for research and practice, limitations
7. **Conclusion** - Contributions and future work

---

## Citation

```bibtex
@mastersthesis{Ketata2026MeetingProcessTwin,
  author  = {Ketata, Aziz},
  title   = {Meeting Process Twin: Multimodal Process Mining for Shadow Workflow Detection in City Council Meetings},
  school  = {Technical University of Munich},
  year    = {2026},
  type    = {Master's Thesis}
}
```

---

## License

Research artifact developed for academic purposes at TUM. Contact the author for licensing inquiries.

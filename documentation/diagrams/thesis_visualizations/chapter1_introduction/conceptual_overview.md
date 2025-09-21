# Chapter 1: Conceptual Overview Diagram

**Purpose**: Illustrating the goal: Synchronized capture of RGB, Thermal, and GSR data to enable future contactless GSR prediction.

## Figure 1.1: Conceptual Overview

```mermaid
flowchart TD
    %% Participant and Goal
    P[[UNICODE] Research Participant]
    GOAL["[TARGET] Goal: Contactless GSR Prediction<br/>Enable stress monitoring without electrodes"]

    %% Data Collection Phase
    subgraph DATA["[DATA] Multi-Modal Synchronized Data Collection"]
        direction TB

        %% Traditional GSR (Ground Truth)
        subgraph CONTACT["[SENSOR] Contact-Based (Ground Truth)"]
            GSR_SENSOR["[SIGNAL] Shimmer3 GSR+ Sensor<br/>128Hz, 16-bit resolution<br/>Skin conductance measurement"]
        end

        %% Contactless Sensors
        subgraph CONTACTLESS["[NETWORK] Contactless Sensors"]
            THERMAL["[THERMAL] Topdon TC001 Thermal Camera<br/>256[UNICODE]192 pixels, 25Hz<br/>Nasal temperature, palm regions"]
            RGB["[UNICODE] RGB Camera<br/>30fps high-resolution<br/>Facial expressions, blood flow"]
        end

        %% Synchronization
        SYNC["[INTEGRATION] Hardware Timestamp Synchronization<br/>+/-3.2ms accuracy using NTP"]

        GSR_SENSOR --> SYNC
        THERMAL --> SYNC
        RGB --> SYNC
    end

    %% Machine Learning Pipeline
    subgraph ML["[UNICODE] Future Machine Learning Pipeline"]
        direction TB
        DATASET["[PERFORMANCE] Synchronized Dataset<br/>GSR signals aligned with<br/>thermal + RGB features"]

        FEATURES["[ANALYSIS] Feature Extraction<br/>[UNICODE] Thermal ROI temperature changes<br/>[UNICODE] RGB-based physiological signals<br/>[UNICODE] Temporal pattern analysis"]

        MODEL["[AI] Prediction Model<br/>Contactless GSR estimation<br/>from thermal + RGB data"]

        DATASET --> FEATURES
        FEATURES --> MODEL
    end

    %% Stress Scenarios
    subgraph SCENARIOS["[TEST] Controlled Stress Induction"]
        STROOP["[FORMAT] Stroop Color-Word Test<br/>Cognitive interference task"]
        TSST["[UNICODE] Trier Social Stress Test<br/>Public speaking protocol"]
        BASELINE["[UNICODE] Baseline Rest Periods<br/>Relaxed state recording"]
    end

    %% Future Applications
    subgraph APPLICATIONS["[PERFORMANCE] Future Applications"]
        THERAPY["[UNICODE] Therapy Sessions<br/>Real-time stress feedback"]
        UX["[PC] UX Research<br/>Unconscious stress detection"]
        HEALTH["[UNICODE] Health Monitoring<br/>Continuous stress assessment"]
    end

    %% Connections
    P --> DATA
    P --> SCENARIOS

    DATA --> ML
    SCENARIOS --> ML

    ML --> APPLICATIONS

    %% Advantage Callouts
    ADVANTAGE1["[OK] No electrode artifacts<br/>[OK] Comfortable long-term use<br/>[OK] Natural user behavior"]
    ADVANTAGE2["[DATA] Research-grade accuracy<br/>[DATA] Multi-modal validation<br/>[DATA] Synchronized ground truth"]

    CONTACTLESS -.-> ADVANTAGE1
    CONTACT -.-> ADVANTAGE2

    %% Current vs Future Scope
    CURRENT["[LIST] Current Thesis Scope<br/>Data collection platform<br/>Synchronization validation"]
    FUTURE["[UNICODE] Future Work<br/>ML model development<br/>Real-time prediction"]

    DATA --> CURRENT
    ML --> FUTURE

    %% Styling
    classDef goalStyle fill:#e1f5fe,stroke:#0277bd,stroke-width:3px
    classDef dataStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef mlStyle fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef scenarioStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef appStyle fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef advantageStyle fill:#f1f8e9,stroke:#689f38,stroke-width:1px,stroke-dasharray: 5 5
    classDef scopeStyle fill:#e0f2f1,stroke:#00796b,stroke-width:2px,stroke-dasharray: 10 5

    class GOAL goalStyle
    class DATA,CONTACT,CONTACTLESS,SYNC dataStyle
    class ML,DATASET,FEATURES,MODEL mlStyle
    class SCENARIOS,STROOP,TSST,BASELINE scenarioStyle
    class APPLICATIONS,THERAPY,UX,HEALTH appStyle
    class ADVANTAGE1,ADVANTAGE2 advantageStyle
    class CURRENT,FUTURE scopeStyle
```

## Alternative PlantUML Version

```plantuml
@startuml ConceptualOverview
!theme plain
title Conceptual Overview: Multi-Modal Synchronized Data Collection\nfor Contactless GSR Prediction Research

skinparam backgroundColor #FAFAFA
skinparam defaultFontSize 10

actor "Research\nParticipant" as P
cloud "Goal: Contactless GSR Prediction" as GOAL

package "Multi-Modal Data Collection Platform" as PLATFORM {
    package "Contact-Based (Ground Truth)" as CONTACT {
        component [Shimmer3 GSR+ Sensor\n128Hz, 16-bit\nSkin conductance] as GSR
    }

    package "Contactless Sensors" as CONTACTLESS {
        component [Topdon TC001\nThermal Camera\n256[UNICODE]192, 25Hz] as THERMAL
        component [RGB Camera\n30fps HD\nFacial analysis] as RGB
    }

    component [Hardware Timestamp\nSynchronization\n+/-3.2ms accuracy] as SYNC

    GSR --> SYNC
    THERMAL --> SYNC
    RGB --> SYNC
}

package "Future ML Pipeline" as ML {
    database "Synchronized\nDataset" as DATA
    process "Feature\nExtraction" as FEATURES
    process "Prediction\nModel" as MODEL

    DATA --> FEATURES
    FEATURES --> MODEL
}

package "Stress Induction Protocols" as STRESS {
    usecase "Stroop Color-Word Test" as STROOP
    usecase "Trier Social Stress Test" as TSST
    usecase "Baseline Rest Periods" as BASE
}

package "Future Applications" as APPS {
    usecase "Therapy Sessions" as THERAPY
    usecase "UX Research" as UX
    usecase "Health Monitoring" as HEALTH
}

P --> PLATFORM
P --> STRESS
PLATFORM --> ML
STRESS --> ML
ML --> APPS

note right of CONTACTLESS : [OK] No electrode artifacts\n[OK] Comfortable long-term use\n[OK] Natural user behavior

note right of CONTACT : [DATA] Research-grade accuracy\n[DATA] Multi-modal validation\n[DATA] Synchronized ground truth

note bottom of PLATFORM : **Current Thesis Scope**:\nData collection platform\nSynchronization validation

note bottom of ML : **Future Work**:\nML model development\nReal-time prediction

@enduml
```

## Caption and Usage

**Figure 1.1**: Conceptual overview of the multi-modal physiological sensing platform showing the integration of contact-based GSR measurement (ground truth) with contactless thermal and RGB sensors. The platform enables synchronized data collection during controlled stress induction protocols to create datasets for future contactless GSR prediction research. Current thesis scope focuses on the data collection infrastructure, while machine learning model development represents future work.

**Thesis Placement**: Chapter 1, Section 1.1 (Introduction and Motivation)
**File Reference**: `docs/diagrams/thesis_visualizations/chapter1_introduction/conceptual_overview.md`

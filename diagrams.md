# System Diagrams

## 1. System design — the 7 locked components

```mermaid
flowchart TB
    subgraph Edge["Edge Pod (DistilBERT)"]
        EM[ResourceMonitor\npsutil: cpu/mem/disk/thermal]
        ED[EdgeDevice\nemit_report / heartbeat\nmark_unreachable]
        MC[MiscalibrationClassifier\nlogistic regression\nfallback heuristic]
        EM --> ED
        MC --> ED
    end

    subgraph Medallion["Dynamic Medallion Pipeline"]
        BR[Bronze\nDynamicMetricRegistry\nregister/snapshot]
        SI[Silver\nSilverEnricher\nconditional enrichment\npredicted_queue_wait, predicted_rtt]
        GO[Gold\nGoldNormalizer\n13-slot vector, zero-masked]
        BR --> SI --> GO
    end

    subgraph MCP["MCP Skill Interface"]
        LT[list_tools discovery]
        SK[perception skills\nget_model_confidence etc.]
        LT --> SK
    end

    subgraph Agent["PPO Agent"]
        PC[perceive]
        RE[reason\nPPONetwork forward pass]
        AC[act\nentropy fallback tau=0.9]
        RF[reflect\nlog buffer, NOT online learning]
        PC --> RE --> AC --> RF
    end

    subgraph Cascade["Cascade Controller"]
        DI[dispatch]
        FB["_unreachable_fallback\nhard override"]
    end

    Cloud[Cloud Pod\nBERT-large via TorchServe]

    ED -->|EdgeContextReport| BR
    GO --> MCP
    MCP --> Agent
    Agent -->|action| DI
    DI -->|ROUTE_TO_EDGE| Edge
    DI -->|ESCALATE_TO_CLOUD| Cloud
    DI -->|edge unreachable| FB
    FB --> Cloud

    DI -.->|update_from_outcome\nBACKWARD LOOP| ED

    subgraph Orchestration["Policy Orchestration Layer (offline/governance)"]
        SEL[select_decision\nreuse / fine_tune / train_new]
        LOG[(orchestrator_decisions.jsonl)]
        LIB[(policy_library/)]
        SEL --> LOG
        SEL --> LIB
    end

    Agent -.->|frozen policy assigned at\nonboarding, not per-request| Orchestration
```

**Reading this**: the top three subgraphs (Edge, Medallion, MCP) are the
**forward** perception path; Agent + Cascade are decision + dispatch; the
dotted line back into `EdgeDevice` is the **backward** loop (Component 6) —
note it updates edge state, not the policy weights. The Policy Orchestration
Layer (Component 7) sits outside the per-request path entirely — it assigns
which frozen policy a client's traffic gets routed through at onboarding
time, never inside a single request's decision.

---

## 2. Per-request workflow — perceive/reason/act/reflect

```mermaid
sequenceDiagram
    participant Req as Request
    participant Edge as Edge Model
    participant ED as EdgeDevice
    participant Med as Bronze/Silver/Gold
    participant Agent as AgenticOrchestrator
    participant Cloud as Cloud Pod

    Req->>Edge: predict(text)
    alt edge inference throws/times out
        Edge--xED: exception/timeout
        ED->>ED: mark_unreachable()
        ED->>Cloud: hard override, skip agent
        Cloud-->>Req: response (fallback_triggered=True)
    else edge inference succeeds
        Edge->>ED: emit_report(confidence, is_correct)
        ED->>ED: compute calibration_delta,\error_rate, classifier.predict()
        ED->>Med: EdgeContextReport
        Med->>Med: Silver: conditional enrichment\n(gated on operational_state)
        Med->>Med: Gold: normalize + zero-mask
        Med->>Agent: 13-slot state vector
        Agent->>Agent: reason() - PPO forward pass
        alt entropy > 0.9 (out-of-distribution)
            Agent->>Agent: static tau=0.9 fallback
        else confident decision
            Agent->>Agent: use PPO action
        end
        alt action == ROUTE_TO_EDGE
            Agent-->>Req: edge result
        else action == ESCALATE_TO_CLOUD
            Agent->>Cloud: dispatch
            Cloud-->>Req: cloud result
        end
        Req->>ED: update_from_outcome(latency, sla_met, accuracy)
        Note over ED: BACKWARD LOOP: next request's<br/>Bronze snapshot reflects this outcome
        Agent->>Agent: reflect() - log buffer only,<br/>PPO weights stay frozen
    end
```

---

## 3. Policy Orchestration Layer — governance-time decision flow

```mermaid
flowchart LR
    NC[New client scenario] --> EV[Evaluate every policy\nin library against scenario\n3x averaged episodes]
    EV --> SD{select_decision}
    SD -->|within tolerance| RU[REUSE\nassign existing checkpoint]
    SD -->|closest miss| FT["FINE_TUNE\n5,000 steps from\nclosest checkpoint"]
    SD -->|nothing close / empty library| TN["TRAIN_NEW\n50,000 steps from scratch"]
    RU --> LOG[(decision log:\nclient, scenario, candidates,\ndecision, dominant_signal)]
    FT --> LOG
    TN --> LOG
    RU --> ASSIGN[Client routed to\nassigned policy for\nall future requests]
    FT --> ASSIGN
    TN --> ASSIGN
```

**Tolerance thresholds** (`orchestrator.py`): `max_sla_violation_rate=0.15,
min_accuracy=0.80, max_escalation_rate=0.60`. This is a governance-time,
offline decision — it never runs inside the synchronous per-request path in
diagram 2.

---

## 4. Ablation Run 7 result, current run

```mermaid
flowchart TB
    subgraph Shared["Single Shared Policy (control)"]
        S1[client_streamforge: SLA 0%, P95 134ms]
        S2[client_nhs: SLA 0%, P95 120ms]
        S3[client_babcock: SLA 100%, P95 727ms]
        S4[client_newco: SLA 0%, P95 118ms]
    end
    subgraph Orch["Policy Orchestration Layer"]
        O1[client_streamforge: REUSE, P95 127ms]
        O2[client_nhs: REUSE, P95 126ms]
        O3["client_babcock: FINE_TUNE, P95 593ms\n(-134ms, SLA still 100% -\nRTT-bound, not a policy gap)"]
        O4[client_newco: REUSE, P95 122ms]
    end
    S1 -.-> O1
    S2 -.-> O2
    S3 -.->|orchestrator detects\nnear-miss, fine-tunes| O3
    S4 -.-> O4
```

# Healthcare Agent User Guide

## Orchestrate Console

Use this link to manage and test your deployed agent:

<https://dl.watson-orchestrate.ibm.com/build/manage>

## Prompt Library

Copy and paste any of the prompts below in the Orchestrate chat panel.

### Core one-shot prompts

```text
Use patient_id PT-1001 and symptom_report "I feel short of breath and dizzy this morning." Call resolve_recovery_case first. Return only tool-grounded output with risk_status, escalation_required, key_findings, immediate_actions.
```

```text
Use patient_id PT-1001 and symptom_report "Mild fatigue after walking, breathing normal, no fever." Call resolve_recovery_case first and summarize escalation risk.
```

```text
Use patient_id PT-2002 and symptom_report "My leg is more swollen and I missed my antibiotic pickup." Call resolve_recovery_case first and return final escalation summary.
```

### Strict output-control prompts

```text
Use patient_id PT-1001 and symptom_report "I feel short of breath and dizzy this morning." Call resolve_recovery_case first. Return JSON only with keys: risk_status, escalation_required, key_findings, immediate_actions.
```

```text
Do not ask follow-up questions. Use patient_id PT-1001 and symptom_report "I feel short of breath and dizzy this morning." Call resolve_recovery_case first and provide only tool-grounded output.
```

```text
If any tool fails, return TOOL_ERROR with exact failure text; otherwise return only risk_status, escalation_required, key_findings, immediate_actions.
```

### Tool-specific prompts

```text
Call get_patient_snapshot for patient_id PT-1001 and summarize only vitals, meds, and appointment facts from tool output.
```

```text
Call translate_discharge_plan for patient_id PT-1001 and return checklist, medication schedule, and safety tips.
```

```text
Call monitor_recovery_status for patient_id PT-1001 with symptom_report "I feel short of breath and dizzy this morning." Return triage and recommended action.
```

```text
Call coordinate_care_logistics for patient_id PT-1001 and return medication status, appointment status, barriers, and resolved actions.
```

### Comparison and prioritization prompts

```text
Run resolve_recovery_case for PT-1001 with symptom_report "Short of breath and dizzy this morning" and PT-2002 with symptom_report "Leg swelling increased and missed antibiotic pickup." Compare risk and escalation decisions.
```

```text
Which patient is higher priority right now: PT-1001 with "shortness of breath and dizziness" or PT-2002 with "swelling and missed antibiotic"? Use tools first.
```

### Care-team handoff prompts

```text
Create a nurse handoff for PT-1001 using resolve_recovery_case output only. Include risk level, escalation decision, and next 3 actions.
```

```text
Create a caregiver-friendly summary for PT-1001 from tool output only, plain language, max 6 bullets.
```

### Follow-up prompts

```text
Use PT-1001 and symptom_report "dizziness improved but still mild shortness of breath." Re-evaluate risk and say whether escalation is still required.
```

```text
Re-run PT-1001 with symptom_report "no shortness of breath today, no dizziness, eating well." Return updated risk and de-escalation guidance.
```

---
description: Generate a minimal Diffusers starter snippet for a requested task.
argument-hint: [task description]
disable-model-invocation: true
---

Generate a concise, runnable Diffusers starter snippet for this request:

$ARGUMENTS

Requirements:
1. Choose the correct pipeline for the task.
2. Include imports, model loading, device placement, and one generation call.
3. Add one optional optimization toggle (speed or memory).
4. End with "Next files to open" and list 2-3 relevant paths from this repository.

# Ollama Model Compatibility - Findings

**Issue:** #3 - Resolve Ollama model compatibility (llama2, gemma:latest) or document permanently
**Outcome:** Not resolved on this hardware. Root cause fully isolated across two
independent upgrade attempts. Staying on the current working version (0.10.1) is
the correct decision, not a workaround.

## Background

Pilot v1 found llama2:7b and gemma:latest (9B) failed 100% of API calls via
/api/generate with "llama runner process no longer running", while working
correctly via the interactive CLI (ollama run). This pointed at a bug specific
to that Ollama version's API-serving code path. The natural fix to try: upgrade
Ollama, since bug fixes across ~20 releases (0.10.1 to 0.31.1) seemed likely to
include a fix.

## Attempt 1: Homebrew upgrade - blocked before it could even install

Homebrew attempted to build 14 dependencies from source (this Mac's macOS 13 is
"Tier 3" for Homebrew, meaning no pre-built bottles for several packages). The
build reached the mlx dependency (Apple's ML framework, required by newer
Ollama builds for Metal acceleration) and failed outright with an error stating
Xcode 15.0 is required and that mlx does not run on macOS versions older than
Sonoma.

This is a hard OS-version gate, not a configuration issue. No amount of retrying,
installing Xcode Command Line Tools, or reconfiguring Homebrew changes this --
the formula itself declines to run.

## Attempt 2: Official installer script - installed, but broke everything

Switched to Ollama's own install method (curl -fsSL https://ollama.com/install.sh
| sh), which ships a pre-built binary rather than compiling locally. This
successfully installed the ollama CLI at version 0.31.1, bypassing the Homebrew
build entirely.

However, testing revealed this version is broken on this machine for every
model, not just the two that failed in Pilot v1:

| Model | Pilot v1 (Ollama 0.10.1) | New attempt (Ollama 0.31.1) |
|---|---|---|
| llama2:7b | API: crash / CLI: works | API: crash / CLI: crash |
| gemma:latest (9B) | API: crash / CLI: works | API: crash / CLI: crash |
| mistral:latest | Works (used throughout Pilot v1) | API: crash / CLI: crash |
| gemma:2b | Works (used throughout Pilot v1) | API: crash / CLI: crash |

Every model, including the two that had worked reliably throughout the entire
Pilot v1 study, now fails identically with: Error: 500 Internal Server Error:
llama-server process has terminated: signal: abort trap

This is a different, more severe failure than Pilot v1's original bug -- not an
API-layer-specific issue anymore, but a total crash of the underlying
llama-server process regardless of access method (CLI or API), for every model
tested. This looks like a Metal/GPU-level incompatibility between Ollama 0.31.1's
updated inference backend and this specific combination of Apple Silicon
generation and macOS 13 -- consistent with mlx's own explicit statement that it
does not support pre-Sonoma macOS.

## Resolution: rolled back to 0.10.1, confirmed working

Downloaded the exact 0.10.1 binary directly from Ollama's GitHub releases
(bypassing both Homebrew and the installer script, which both default to the
latest version), replaced the broken binary, and confirmed mistral:latest and
gemma:2b both respond correctly again.

Both models used throughout Pilot v1 are confirmed fully functional again on
0.10.1. No data or capability was lost; this investigation cost time but not
correctness.

## Conclusion

This is not a problem this project can fix through configuration, retrying, or
code changes. Two independent upgrade paths, using two different installation
mechanisms, both confirm the same underlying fact: Ollama versions newer than
~0.10.x on this machine (macOS 13, Apple Silicon) are broadly incompatible, not
narrowly incompatible with just llama2/gemma:latest as Pilot v1's original
finding suggested. The original finding undersold the scope of the problem --
staying on 0.10.1 is not a workaround for two broken models, it is the only
version confirmed to run any model reliably on this hardware.

Decision: remain on Ollama 0.10.1 for the remainder of this project.
mistral:latest and gemma:2b are the two models this pipeline can reliably
evaluate on this machine. This should be stated plainly in the Version 2 report
as a hardware/OS constraint of the research environment, not hidden as a
methodology gap.

## If model coverage needs to expand in the future

Since this is a machine-level constraint, not a code-level one, the only real
paths to restoring llama2/gemma:latest (or adding new models) are:

1. Run the pipeline on different hardware/OS (e.g. a Linux cloud VM, or a Mac on
   macOS 14+) where current Ollama versions install and run normally.
2. Use a different local inference backend that doesn't share Ollama's mlx
   dependency chain (e.g. llama.cpp directly, or LM Studio).
3. Use hosted API-based models (e.g. via Together AI, Groq, or similar) instead
   of locally-served models, removing the local hardware constraint entirely.

None of these are necessary for the current 2-model pilot scope, but should be
considered before committing to specific models for the 300-500 prompt Version 2
study (Issue #4), since this hardware ceiling will persist.

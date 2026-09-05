# Benchmark results

| Benchmark | Evaluation | Solved | Score |
| --- | --- | ---: | ---: |
| SWE-bench Verified | 500 tasks, one attempt per task | 401 / 500 | **80.2%** |
| Terminal-Bench 2.1 | 89 tasks, five attempts per task | 309 / 445 | **69.44%** |

Both archives identify the agent source version as `d8b0fb7`, the model as
`deepseek-v4-flash`, and the runner as Harbor 0.20.0. The Terminal-Bench run was
recorded on August 14–16, 2026; the SWE-bench run on August 20–22, 2026. The imported
source baseline is commit `d8b0fb7974ec934252df1e06488f65a494699002`, with project
naming updated. These historical scores do not evaluate the new TUI.

The Terminal-Bench score averages all 445 attempts; it is **not pass@5**. The
SWE-bench denominator includes all 500 tasks, including two tasks without a
verifier reward. The archived Terminal-Bench dataset reference is
`sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a`.

### Public leaderboard context

The following are the nearest lower scores on the official pages inspected on
**September 4, 2026**. For SWE-bench, the comparison uses the complete Verified
view with **Bash Only disabled**, so custom harnesses are included.

| Benchmark | Public comparison entry | Listed score | Difference from this run |
| --- | --- | ---: | ---: |
| SWE-bench Verified | Sonar Foundation Agent + Claude Opus 4.5 | 79.2% | +1.0 percentage point |
| SWE-bench Verified | live-SWE-agent + Claude Opus 4.5, medium effort | 79.2% | +1.0 percentage point |
| Terminal-Bench 2.1 | Claude Code + Claude Opus 4.7, max effort | 68.9% | +0.54 percentage point |

Sources: [SWE-bench official leaderboard](https://www.swebench.com/) and
[Terminal-Bench 2.1 official leaderboard](https://www.tbench.ai/?version=2.1).

These are numerical comparisons between separately run agent/model systems.
They do not isolate the harness effect or establish statistically significant
improvements. In particular, the Terminal-Bench comparison entry reports a
95% confidence interval of ±2.8 percentage points. The SWE-bench comparison
entries are listed submissions, not marked as checked by the SWE-bench team.
The scores here have not been accepted as official mini-harness leaderboard
submissions; they should not be described as an official rank or a SOTA claim.

The [README comparison figure](README.md#benchmarks) includes all **18 public
Terminal-Bench 2.1 entries** visible on that date plus this self-reported run.
Positions are recomputed by descending score, retaining ties. The inserted
mini-harness result is **13th of 19**, between Terminus 2 + Gemini 3 Pro (high),
73.9% ±2.5%, and Claude Code + Opus 4.7 (max), 68.9% ±2.8%. This is an
illustrative score order, not an official leaderboard placement. Public rows
show their published 95% confidence intervals; no interval was computed for
mini-harness.

### Recorded exceptions and reproducibility

The archived evaluations were **not error-free**:

- **Terminal-Bench 2.1:** 108 agent timeouts and five nonzero agent exits. All 445
  trials have verifier rewards. Among the 337 available run-telemetry records,
  all report the agent loop reaching its normal completion outcome; that does
  not mean all tasks were solved.
- **SWE-bench Verified:** one agent timeout, two nonzero agent exits, and two
  Docker environment-build failures. One nonzero exit records an API 400 error
  requiring `reasoning_content` to be passed back in thinking mode. This is a
  known API-message compatibility issue in the evaluated baseline.

The full archives are saved under the mini-harness name on Harbor Hub and are
currently **private**. Uploading archives to Harbor Hub is separate from official
leaderboard submission. This source-only repository includes the agent and its
unattended profile; the external Harbor adapter, job definitions, and raw
benchmark archives are not bundled, so the historical evaluations are not yet
reproducible from this checkout alone.


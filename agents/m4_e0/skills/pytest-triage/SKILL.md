---
name: pytest-triage
description: How to pick and run the smallest relevant pytest target and read a failure quickly inside the offline sandbox. Use when verifying a fix or when a targeted test fails.
---

# pytest triage (SKILL.md only; no scripts until run_skill_script is verified, plan Q1)

This skill is NOT declared in agent.yaml for v1 (E8 ablation). ADK requires the frontmatter name to equal the
directory name (kebab-case).

1. Find candidates: run_command with git grep -l over tests for the symbol or module name, piped through head -n 10.
2. Run one file or node id only: run_command with timeout 90 python -m pytest -x -q on that target, piped through tail -n 60. Never run the whole suite.
3. Read the failure bottom-up: the innermost repository frame is the suspect. An ImportError or SyntaxError means the edit broke the import; a wrong value means logic; a wrong exception type or message means a contract mismatch with the issue text.
4. Pre-existing failures exist. If unsure whether your edit caused one, compare with run_command: git stash, rerun the target, git stash pop.
5. Hidden grading tests are new tests for the issue; passing existing tests is necessary but not sufficient, so also keep the /tmp repro script green.

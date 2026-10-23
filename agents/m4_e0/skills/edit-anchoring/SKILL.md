---
name: edit-anchoring
description: How to write old_string/new_string for edit_file so it matches on the first try inside the offline sandbox. Use before calling edit_file, especially on a file you have not just read in full.
---

# edit anchoring (SKILL.md only; no scripts until run_skill_script is verified, plan Q1)

This skill is NOT declared in agent.yaml for v1 (E8 ablation). ADK requires the frontmatter name to equal the
directory name (kebab-case).

edit_file matches old_string in three tiers: exact (after \r\n normalization), flexible (line-by-line,
whitespace-insensitive, re-indents new_string to the matched block), then regex (tokenized around
`( ) : [ ] { } > = <` with flexible whitespace). A failed match or an ambiguous one (more than one
occurrence without allow_multiple) costs a tool call and teaches nothing — avoid both up front:

1. Copy old_string verbatim from the most recent read_file output for that exact file, not from memory
   or from an earlier read before other edits changed it. Memory drifts on whitespace, quote style, and
   blank lines; the matcher's exact tier only forgives \r\n, nothing else.
2. Include enough surrounding context (a full line above and below the change, or the full def/class
   signature line) to make old_string unique in the file, but no more than needed — a short unique
   anchor beats a long verbatim block that is more likely to contain a transcription slip.
3. If old_string could plausibly repeat (a common decorator, a short conditional, a repeated literal),
   either widen the anchor to include a neighboring unique line or pass allow_multiple=true only when
   you deliberately want every occurrence changed — never as a reflex to dodge a uniqueness error.
4. On a FileEditError, re-read the file (it may have shifted from a previous edit or does not exist at
   that path) rather than retrying the same old_string with minor guesses; guessing burns tool-call
   budget with no new information.
5. Prefer several small, independently-anchored edit_file calls over one large multi-hunk change —
   each is easier to anchor correctly and a failure in one does not block the others.

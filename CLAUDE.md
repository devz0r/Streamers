## Subagents

Use subagents on your own judgement, without being asked, when they help:

- **Broad searches**: "where is X used", "how do these modules fit together" -- anything that
  means sweeping many files. Send the `scout` subagent and work from its summary, so the main
  context stays on the task.
- **Independent work that can run at once**: separate backtests, probes, or docs lookups that do
  not depend on each other. At most 3 at a time.
- **Not for small things**: one file, one grep, or anything already known is faster done directly.
  Every subagent starts from scratch.

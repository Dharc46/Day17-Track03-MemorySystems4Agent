# Analysis

## Benchmark summary

The offline benchmark shows the expected memory trade-off:

- In the standard benchmark, `Advanced` has better cross-session recall because it writes stable facts into `User.md` and can read them from a fresh thread.
- In short or medium conversations, `Advanced` can process more prompt tokens than `Baseline` because it carries `User.md` plus recent thread context.
- In the long-context stress benchmark, `Advanced` processes fewer prompt tokens than `Baseline` because compact memory keeps only recent messages in full and moves older turns into a summary.

## Memory growth

`User.md` grows when the agent sees stable facts such as name, profession, location, response style, favorite food or drink, and technical interests. This improves recall, but it introduces risks:

- stale facts can remain if corrections are not handled;
- noisy facts can make the memory file harder to inspect;
- a large memory file increases prompt cost over time.

The implementation handles common corrections by upserting facts by key, so newer facts like `location` or `profession` replace older conflicting values instead of appending duplicates.

## Compact memory

Compact memory mainly optimizes `Prompt tokens processed`, not `Agent tokens only`. The agent still generates a response each turn, but it no longer needs to resend every old message in full. This is why compaction has the clearest benefit in the stress benchmark rather than in short conversations.

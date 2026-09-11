# Sequential workflows

Kanna executes the existing `Plan` and `PlanStep` types through `AgentLoop` and the tool registry.
There is no separate workflow engine or bypass of tool permissions. Both interactive and scheduled
requests use this path.

## Passing observed values

An argument value can be a reference object: `{"$ref": "0.data.content"}`. Step indices start at
zero and must refer to an earlier step. The first field is `data`, `files_created`, `files_modified`,
or `metadata`; following dot-separated components select dictionary keys or list indices. References
replace whole values, including objects and arrays, and can occur inside nested arguments. There is
no interpolation, expression evaluation, attribute access, or implicit conversion to text.

Example plan steps for a source-text report (content copied verbatim, not summarized):

```json
[
  {"tool_name": "fs_read_file", "args": {"path": "source.txt"},
   "expected": {"data_equals": {"truncated": false}}},
  {"tool_name": "document_generate_pdf", "args": {
    "path": "report.pdf", "title": "Source report",
    "sections": [{"heading": "Source", "paragraphs": [{"$ref": "0.data.content"}]}]
  }, "expected": {"min_files_created": 1, "files_exist": ["report.pdf"]}},
  {"tool_name": "fs_file_info", "args": {"path": {"$ref": "1.data.path"}},
   "expected": {"data_equals": {"exists": true, "is_file": true}}}
]
```

Likewise `vision_extract_structure` results can supply `title` and `sections` directly to
`document_generate_docx`/`document_generate_pdf`, preserving the existing section/table model.
Vision still requires a configured provider. Text reading and document generation run offline.

The LLM planner receives input and output schemas and the reference syntax. Literal arguments and
backward-reference syntax are validated before execution. Values available only at runtime are
resolved from verified results and then validated by the registry **before** permission evaluation.
An absent result field fails the workflow without invoking the dependent tool. Values are copied;
a downstream tool cannot modify stored upstream evidence through a shared object.

## Verification and recovery

Supported `expected` fields are `success` (true only), `min_files_created`, `min_files_modified`,
`data_nonempty_key`, `data_equals` (exact top-level field values), `files_exist` (literal sandboxed
paths), and `exit_code`. Unknown or malformed postconditions block the plan before tools run.
All claimed created/modified paths are additionally checked for existence through the sandbox.
Existence is not a guarantee of content accuracy, document layout, or semantic correctness.

`process_run` requires exit code zero by default; an explicit `exit_code` expectation can request
another code. Timeouts always fail verification. The process tool's own result contract remains
unchanged: successful invocation may contain a nonzero exit code, stdout, and stderr.

A failed check now produces `verification_failed` and FAILED, even if the tool returned success.
No later steps run. A successful call with failed verification is not replayed automatically because
it may already have written a file or transaction. Tool failures still use the existing bounded
argument correction; approval/policy/sandbox denials, invalid input/output, and unavailable
capabilities terminate without model-driven retries. No rollback or exactly-once guarantee is added;
an arbitrary failing third-party tool may have partial side effects.

## Evidence and results

`plan_steps.args` preserves the original reference template. `plan_steps.result` contains the final
ToolResult plus `resolved_args` and `attempts`. Each attempt records actual arguments, raw tool result,
and verification errors. `StepOutcome.history` exposes the same evidence to callers. Registry-level
execution logging continues independently. Final messages include output paths and deterministic
tool-provided messages, including finance answers.

Scheduled callbacks returning an agent state succeed only for `complete`; failed/blocked states
remain visible in job results and are recorded as failed jobs. Generic callbacks without a `state`
retain their prior exception-based success contract. Job timestamps use the supplied tick time;
legacy offset-aware timestamps normalize to naive UTC on reading.

## Remaining work

Plans remain sequential and fixed after planning, apart from argument correction. There is no
content-authoring call that can reason over newly read sources, adaptive replanning of a failed build,
assignment task extraction/solving, connector submission, or cross-device routing yet. Copying
extracted structure is not answering an assignment. The next milestone should add a provider-backed,
schema-validated content-authoring capability over observed sources, using these references and the
existing document renderers, with provenance and explicit unresolved questions.

`tests/test_workflows.py` uses scripted model/vision responses with real file operations, SQLite,
process execution, and document rendering. These tests verify orchestration, not live model quality.

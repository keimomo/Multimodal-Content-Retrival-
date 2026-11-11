# Open bugs and proposed fixes

Updated: 10 November 2026. Scope: the CLI and its supporting `src/` modules, checked against the current working tree.

This report contains **30 open findings**. Resolved entries have been removed; original issue numbers are retained, so gaps are intentional. Newly identified issues continue from the original numbering. Each entry proposes a fix, justification, and verification step. The work remains focused on retrieval and evidence-backed answers; Option B and model training are outside scope.

Most findings follow directly from the source. Current isolated checks confirm the corrected namespace, frame-call arguments, and original-video seek positions. The video-indexing regression suite now confirms CSV-based clip paths, separate source-prefixed document IDs, explicit video inputs, and missing-file rejection; all seven checks pass, including real FFmpeg segmentation and ffprobe validation. Two additional audio-boundary checks cover newly queued Path objects and strings reloaded from SQLite; Whisper now receives a filename string in both cases with unchanged transcription options. All nine regression tests pass. Encoder/transcription inference and vector uploads are simulated in these checks. Earlier isolated checks reproduced SQLite write contention and uncommitted default initialization. Checks used temporary databases/files and simulated capture/embedding, without model downloads or API calls. Provider compatibility, GPU behavior, live Pinecone contents, required data migration/reindexing, and a complete indexing/query run were not tested live. The list is a review inventory, not a guarantee that no other bugs exist.

Priority meanings:

- **P0:** unintended deletion of unrelated files.
- **P1:** broken core flows, lost/corrupted index data, or materially incorrect evidence.
- **P2:** configuration, reliability, isolation, or validation problems.
- **P3:** helper behavior, usability, or diagnostics.

Items 15, 27, 29, and 33 are explicitly qualified as lifecycle, product-behavior, validation, or helper-level gaps. They should not be interpreted as four independently reproduced application crashes.

## Retrieval, evidence, and data integrity

### 5. Temporary-directory cleanup can delete unrelated files

**Priority:** P0. **Evidence:** confirmed in source; destructive behavior was not executed.

**Location:** [temporary directory selection](src/processors/raw_process.py#L48), [recursive deletion](src/processors/raw_process.py#L41).

**Problem:** processing reuses `<source parent>/temp`, allows FFmpeg overwrites, then deletes the entire directory. Pre-existing files in that directory are not owned by the job but can be overwritten or removed.

**Suggested fix:** allocate a unique job-owned directory with `tempfile.TemporaryDirectory`, optionally under a dedicated application cache. Pass that directory explicitly to segmentation and clean up only that allocated directory through a context manager. Never adopt a pre-existing generic `temp` directory.

**Justification:** ownership is established when the directory is created, so cleanup cannot recursively remove a user's unrelated directory. Unique directories also prevent concurrent jobs from interfering.

**Verification:** place sentinel files in an existing sibling `temp` directory, process a fixture, and confirm the sentinels remain untouched on success and failure.

### 6. Failed vector uploads are treated as successful processing

**Priority:** P1. **Evidence:** confirmed in source.

**Location:** [video error handling](src/database/vector_db.py#L42), [text error handling](src/database/vector_db.py#L62), [source completion](src/processors/raw_process.py#L42).

**Problem:** upload methods print exceptions and return normally. Their callers advance the source to `text_embeded` or `completed` even if some or all vectors are missing. Completed sources are excluded from the startup processing queue.

**Suggested fix:** propagate upload failures, record the affected stage/chunk and error, and advance status only after all required writes succeed. Retry transient failures with bounded backoff; keep permanent failures visible and explicitly retryable. Combine this with issues 7 and 9.

**Justification:** a success status must mean the required data is available. Printing an error does not preserve enough state to recover missing vectors.

**Verification:** make an upload fail partway through a source; confirm the source remains incomplete and retry repairs the missing data.

### 7. Pinecone writes happen before corresponding SQLite commits

**Priority:** P1. **Evidence:** confirmed ordering; interruption scenario not exercised against Pinecone.

**Location:** [chunk insertion, upload, and stage commit](src/processors/raw_process.py#L135).

**Problem:** chunk records are inserted in an open SQLite transaction, then published to Pinecone before that transaction commits. Interruption can leave cloud records referencing absent local chunks. Reused local IDs can subsequently associate those records with unrelated text.

**Suggested fix:** commit stable chunk identities and a durable upload job in a short SQLite transaction before sending vectors. Upload using deterministic IDs, then mark the job/chunk indexed in another short transaction. Replay unfinished jobs on restart. Exclude incomplete or obsolete indexing generations from answer evidence and reconcile orphaned legacy records.

**Justification:** SQLite and Pinecone cannot share one local transaction. A durable job record plus idempotent uploads makes either interruption point recoverable without holding database locks across network calls.

**Verification:** simulate interruption before upload, after upload, and before completion status; restarting must converge to one valid chunk/vector pair.

### 8. Vector IDs are not unique across local databases

**Priority:** P1. **Evidence:** confirmed identity design; applies when an index is reused/shared.

**Location:** [fixed index](src/database/vector_db.py#L24), [text ID](src/database/vector_db.py#L51), [source-prefixed video ID](src/processors/raw_process.py#L29), [local chunk lookup](src/CLI.py#L433).

**Problem:** text IDs are local SQLite row numbers, and video IDs now combine a local source row number with a clip ordinal. This distinguishes recordings within one database, but all installations still target `mm-vectors-cli`. A replacement database or second installation can reuse either identity scheme, overwrite cloud records, or resolve retrieved text IDs to unrelated local chunks. Adding the source prefix fixes the original same-database video collision, not this cross-database identity problem.

**Suggested fix:** persist a library UUID and stable source/chunk UUIDs in SQLite; include library identity in both text/video cloud IDs and retrieval filters. Store the local chunk reference as explicit metadata rather than assuming the Pinecone ID is an integer row ID. Validate the library and source association before loading text.

**Justification:** a local row number identifies a row only within its own database. The cloud index needs identities that remain unambiguous across database lifetimes.

**Verification:** use two temporary libraries with overlapping integer row IDs against a simulated shared index; neither may overwrite or resolve the other's records.

## Worker reliability and lifecycle

### 9. One failed job terminates the entire worker

**Priority:** P1. **Evidence:** confirmed in source.

**Location:** [worker loop](src/CLI.py#L159).

**Problem:** model loads, transcription, segmentation, embedding, and database writes have no per-job exception boundary. One exception ends the thread, leaves queue bookkeeping incomplete, and prevents later jobs from processing.

**Suggested fix:** put each acquired job inside `try/except/finally`; roll back an open local transaction, record the failed stage/error, and always call `task_done()` for that acquired job. Continue with independent jobs. Handle initialization failures separately with bounded retry or an explicit stopped-worker status. Do not advance a failed job to the next stage.

**Justification:** an individual bad source should not disable ingestion for the whole session, and failures must remain visible instead of silently stranding work.

**Verification:** queue an invalid source followed by a valid fixture; confirm the first records an error and the second still completes.

### 10. Unsupported files are registered and queued

**Priority:** P2. **Evidence:** confirmed in source.

**Location:** [registration checks](src/CLI.py#L370), [transcription extension check](src/processors/audio_processor.py#L17).

**Problem:** registration accepts any existing file, but transcription accepts only an exact `.mp4` suffix. A PDF or `.MP4` file is registered successfully and fails later; currently that failure also triggers issue 9.

**Suggested fix:** validate supported file types before insertion using `suffix.lower()`, and use the same rule inside the processor. Reject unsupported inputs with an actionable message. Keep worker exception handling for corrupt or unreadable files that pass the extension check.

**Justification:** predictable input errors should be rejected before creating jobs; matching validation at both boundaries also supports uppercase extensions without implying that an extension guarantees valid media.

**Verification:** check `.mp4`, `.MP4`, unsupported extensions, and corrupt MP4 content.

### 11. Indexing holds SQLite write locks during slow work

**Priority:** P1. **Evidence:** lock contention reproduced with isolated connections.

**Location:** [video chunk insertion before embedding](src/processors/raw_process.py#L36), [text chunk insertion inside the upload loop](src/processors/raw_process.py#L135).

**Problem:** chunk inserts start a write transaction that stays open until the end of the stage, spanning further model inference and network uploads. A simultaneous workspace/source write from the CLI can exceed SQLite's busy timeout and fail with `database is locked`.

**Suggested fix:** perform model inference and network operations outside SQLite write transactions. Commit chunk/job updates in short transactions as described in issue 7. Configure an appropriate busy timeout and retry policy for brief contention. WAL may improve reader concurrency, but it does not allow multiple simultaneous writers and is not the primary fix.

**Justification:** shortening lock duration addresses the cause; merely increasing the timeout can make the terminal appear frozen for longer.

**Verification:** slow the simulated embedding/upload operations while repeatedly adding workspaces and sources; writes should not wait for the full indexing stage.

### 12. Model loading and unloading race with inference

**Priority:** P1. **Evidence:** unsynchronized shared access confirmed; exact interleaving not reproduced on hardware.

**Location:** [worker model switching](src/CLI.py#L164), [mode changes](src/CLI.py#L215), [query inference](src/processors/raw_process.py#L222).

**Problem:** chat, mode changes, and ingestion access one embedding instance without synchronization. One path can replace or clear `model` while another is counting tokens or encoding. Concurrent loads can also allocate multiple model instances.

**Suggested fix:** introduce one model owner or a shared lock covering load, the complete inference/tokenization operation, and unload. Ensure every access follows that ownership rule; locking only `load_model()` is insufficient. For the smallest initial change, serialize chat and ingestion model use and display when a query is waiting.

**Justification:** the model must remain available for the entire operation. Serialization is simpler and uses less memory than maintaining separate models on constrained devices.

**Verification:** coordinate a paused inference with a concurrent mode change/job and verify unload waits until inference releases ownership.

### 13. Explicit shutdown unloads a model still in use

**Priority:** P1. **Evidence:** unsafe ordering confirmed in source.

**Location:** [quit sequence](src/CLI.py#L199).

**Problem:** `quit()` unloads the embedding model before setting the exit event or waiting for the worker. An active worker can therefore fail during shutdown.

**Suggested fix:** stop accepting new work, signal shutdown, let the worker finish or stop at a durable stage/chunk boundary, join it, then unload models and close main-thread resources. Use the ownership rule from issue 12 and make shutdown idempotent. Do not forcibly kill the worker halfway through a store update.

**Justification:** resources must outlive their users, and interrupted work needs a recoverable persisted state.

**Verification:** request exit during transcription, embedding, and upload; confirm no use-after-unload and a valid completion or resume state.

### 14. Ctrl+C and EOF bypass worker cleanup

**Priority:** P2. **Evidence:** confirmed control flow; interactive behavior not exercised.

**Location:** [non-daemon worker](src/CLI.py#L50), [input loop](src/CLI.py#L189).

**Problem:** `KeyboardInterrupt` and `EOFError` escape the input loop without setting the worker's exit event. The non-daemon worker can keep the process alive after the terminal loop ends.

**Suggested fix:** wrap startup/run ownership in `try/finally`, handle Ctrl+C and EOF as shutdown requests, and call the same idempotent cleanup used by `exit`. Account for partial startup where the thread never started. Do not rely on making the worker daemonized to bypass cleanup.

**Justification:** every exit path should preserve the same persistence and resource-lifetime guarantees.

**Verification:** exercise normal exit, EOF, Ctrl+C, and a startup exception with and without an active job.

### 15. Worker database connection has no explicit cleanup

**Priority:** P3. **Evidence:** lifecycle gap confirmed; a persistent resource leak was not demonstrated.

**Location:** [worker connection allocation](src/CLI.py#L160).

**Problem:** the connection is not closed in a worker `finally` block. Cleanup depends on object destruction; outstanding transactions have no explicit resolution on worker failure.

**Suggested fix:** keep creation and closure in the worker thread, wrap the worker body in `try/finally`, and explicitly roll back unfinished transactions and close the connection on termination. Commit only completed, intentional units of work.

**Justification:** deterministic cleanup releases locks and file handles predictably across normal shutdown and exceptions. Closing the connection from the main thread would violate its thread ownership.

**Verification:** terminate the worker normally and through an injected exception; confirm its connection is closed and no write lock remains.

## Models and credentials

### 16. Provider selection exposes unsupported answer paths

**Priority:** P1. **Evidence:** request/client mismatch confirmed in source; provider APIs not called live.

**Location:** [client factory](src/processors/llm.py#L98), [answer request](src/processors/llm.py#L155).

**Problem:** the interface permits Anthropic and Google clients, but answer generation always executes an OpenAI-style `responses.parse()` call with OpenAI content objects. There are no provider-specific generation/content adapters.

**Suggested fix:** restrict the active CLI answer provider to OpenAI until other adapters are implemented. Reject unsupported persisted choices with a clear recovery path. If multiple providers become a requirement, implement separate adapters that translate content and normalize results into `RAGAnswer`, with verified provider model identifiers.

**Justification:** hiding unsupported choices is the smallest truthful fix. Constructing a provider client alone does not make the request or structured-output interface compatible.

**Verification:** unsupported providers must fail before entering chat or sending requests; the supported path must return the common answer schema.

### 17. Changing provider does not replace the active client

**Priority:** P2. **Evidence:** confirmed in source.

**Location:** [model selection handler](src/CLI.py#L480), [client initialization on mode entry](src/CLI.py#L226).

**Problem:** model selection changes configuration, but an already initialized client still belongs to the previous provider. The next query can send the new provider's model name through the old client.

**Suggested fix:** centralize model/provider changes. Validate the proposed selection, construct its supported client, and replace the configuration/client together only after successful initialization. Retain the previous working selection if any step fails. Apply the provider restriction from issue 16.

**Justification:** the provider, model ID, credentials, and client form one operational configuration and should never be partially updated.

**Verification:** switch selection while in chat and confirm the next request uses the matching client/model; failed changes must preserve the previous pair.

### 18. Chat model selections are not restored or consistently saved

**Priority:** P2. **Evidence:** confirmed in source.

**Location:** [hardcoded in-memory defaults](src/config/model_config.py#L8), [runtime selection](src/CLI.py#L486), [default-update helper](src/database/sql_db.py#L226).

**Problem:** first-run setup records a selected default in SQLite, but subsequent launches do not restore it into `ModelConfig`. Runtime selections do not update the persisted default either.

**Suggested fix:** load the saved chat model/provider at startup and persist successful runtime changes. Scope default updates to `model_usage='chat'` and clear all previous defaults for that usage in one transaction; avoid choosing one arbitrary default across chat and transcription rows. Validate a restored selection against the supported provider set.

**Justification:** restarting should preserve an explicit user choice. Defaults for different model usages should remain independent.

**Verification:** select a noninitial chat model, restart, and verify both the displayed selection and outgoing request match it without altering the transcription default.

### 19. Transcription model selection does not control the worker

**Priority:** P2. **Evidence:** confirmed in source.

**Location:** [selection](src/config/model_config.py#L61), [worker load call](src/CLI.py#L167), [loader default](src/processors/audio_processor.py#L20).

**Problem:** the menu changes `translation_model_name`, but the worker calls `load_model()` without that name. The loader also does nothing when a different model is already loaded.

**Suggested fix:** pass the selected name through the worker's job/model configuration. At a safe job boundary, compare it with the loaded model name and unload/reload when needed. Persist the choice if it is intended to survive restart, and clarify that this setting controls transcription.

**Justification:** the displayed setting must correspond to the model actually processing speech, without switching a model halfway through a job.

**Verification:** select another Whisper size and inspect the next load request, including the case where `small` was already loaded.

### 20. Credential-storage errors do not prevent successful setup status

**Priority:** P2. **Evidence:** confirmed in source.

**Location:** [swallowed keyring failure](src/config/model_config.py#L86), [setup completion](src/CLI.py#L89).

**Problem:** `save_api_key()` catches a keyring exception and returns normally. Callers can mark the provider/setup complete despite no saved credential.

**Suggested fix:** propagate a specific credential-storage exception or return an explicit result that every caller must check. Mark setup complete only after successful storage and retrieval. Never include the credential value in logs or exception messages.

**Justification:** setup status should represent the durable credential state, not whether the prompt finished. Reading the saved key checks storage, not remote API validity.

**Verification:** simulate a failing keyring backend; no configuration or status should advance, and the user should receive a retryable error.

### 21. API-key status uses tuple truthiness

**Priority:** P2. **Evidence:** confirmed in source.

**Location:** [caller](src/CLI.py#L491), [database return value](src/database/sql_db.py#L236).

**Problem:** `check_api_key()` returns a row tuple. `('Incomplete',)` is truthy, so the caller treats the row's existence as a successful credential status check.

**Suggested fix:** return a typed status value or compare `row is not None and row[0] == 'Complete'`. For actual readiness, check the authoritative keyring result and propagate storage failures from issue 20; do not rely solely on a stale SQLite flag.

**Justification:** row existence and credential readiness are different facts. An explicit result eliminates an easy-to-miss truthiness error.

**Verification:** cover absent rows, `Incomplete`, `Complete`, and a stale `Complete` row with no stored key.

### 22. LLM bootup reports a connection without initializing the client

**Priority:** P2. **Evidence:** confirmed in source.

**Location:** [bootup status](src/CLI.py#L136), [LLM constructor](src/processors/llm.py#L29).

**Problem:** construction only sets `client=None`; it neither reads the key nor creates the provider client. Bootup nevertheless records the connection as complete, and its missing-key recovery branch cannot catch the later credential lookup failure.

**Suggested fix:** either initialize the selected supported client during bootup or rename the status to reflect lazy initialization and perform readiness/recovery when entering chat. If the key is missing, invalidate the setup flag and offer the existing credential setup flow. Do not describe local client construction as a verified remote connection.

**Justification:** status and error handling must correspond to the operation actually performed. Local construction alone does not establish network availability or API authorization.

**Verification:** remove a simulated stored key while leaving SQLite setup complete; startup/chat entry must detect and recover from the mismatch.

### 23. Cancelling selection menus is not handled

**Priority:** P2. **Evidence:** confirmed unchecked `None` handling in source.

**Location:** [first-run model selection](src/config/model_config.py#L26), [chat selection parsing](src/config/model_config.py#L38), [transcription selection parsing](src/config/model_config.py#L61).

**Problem:** cancelled prompts can supply `None`, which is assigned as the selected model or passed to `.split()`. Provider selection cancellation is also passed directly into credential setup.

**Suggested fix:** check for cancellation after every selection prompt. Keep the previous configuration unchanged for runtime menus. For mandatory first-run setup, exit cleanly with setup still incomplete or explicitly return to the setup menu. Prefer structured choice values over splitting display labels.

**Justification:** cancellation is a normal interaction, not an invalid model choice or an internal error.

**Verification:** cancel each prompt and confirm no client/configuration/status changes and no traceback or `None` model.

## Commands and evidence validation

### 24. Input normalization changes paths and question content

**Priority:** P2. **Evidence:** confirmed in source.

**Location:** [input parsing](src/CLI.py#L193).

**Problem:** `.lower()` runs over the entire input. It changes case-sensitive paths and technical identifiers, acronyms, or quoted text in questions.

**Suggested fix:** preserve the raw input and normalize only command/subcommand tokens. Normalize workspace names separately if lowercase workspace identity is intentional. Preserve the query payload and file path exactly as entered after parsing.

**Justification:** command matching should not modify the data supplied to that command.

**Verification:** add a mixed-case filename on a case-sensitive filesystem and submit a question containing `US`, `us`, or a case-sensitive code identifier.

### 25. Literal-space splitting breaks quoted paths and whitespace

**Priority:** P2. **Evidence:** confirmed in source.

**Location:** [input tokenization](src/CLI.py#L193).

**Problem:** `.split(' ')` creates empty arguments for repeated spaces and does not interpret quotes. Paths containing spaces cannot be supplied as one argument; whitespace-only questions can also pass the initial argument-count check.

**Suggested fix:** separate the `q:` prefix from its remaining free-text payload before command tokenization, reject an empty trimmed query, and use `shlex.split()` for the current Unix-style command interface. Catch malformed quoting and validate argument counts. Document the quoting convention.

**Justification:** free-text questions and structured commands need different parsing rules. This preserves question content while supporting quoted paths and flexible command whitespace.

**Verification:** try repeated spaces, blank input, a blank `q:`, quoted paths, and unmatched quotes.

### 26. Query commands can run without initialized dependencies

**Priority:** P2. **Evidence:** confirmed in source.

**Location:** [query handler](src/CLI.py#L415), [chat initialization](src/CLI.py#L226).

**Problem:** `q:` is dispatched regardless of state, but embedding/client initialization happens on chat entry or incidentally through background work. Queries can fail with an unloaded model or `client=None`.

**Suggested fix:** use one `ensure_query_ready()` operation that verifies the supported model/client and acquires model ownership before inference. Either invoke it automatically for `q:` or explicitly require chat mode with a useful message. Check real readiness rather than only the state string.

**Justification:** availability should not depend on whether a background worker happened to load a model or on an undocumented command sequence.

**Verification:** query immediately after startup, after changing to idle/data, and after a failed chat initialization.

### 27. Active workspace does not scope retrieval

**Priority:** P2 if workspace isolation is intended. **Evidence:** behavior confirmed; product expectation requires a deliberate choice.

**Location:** [workspace-independent query](src/CLI.py#L423), [unfiltered vector search](src/database/vector_db.py#L65).

**Problem:** workspace selection affects source registration/listing, but queries search the whole index. This can return another workspace's recordings. If global search is intentional, the missing distinction is a UI/documentation gap rather than an accidental backend failure.

**Suggested fix:** default queries to the selected workspace and offer an explicit whole-library mode. Prefer resolving the workspace's allowed source identities from SQLite and using a supported source filter, or maintain workspace metadata in Pinecone with a synchronized move operation. Scope searches by library as well, addressing issue 8. Migrate/reindex existing records as needed.

**Justification:** search scope becomes predictable. Using SQLite's current membership avoids stale workspace metadata after `ws rm` moves sources to `uncategorized`.

**Verification:** use two workspaces with distinct evidence; scoped search must exclude the other workspace, while explicit global search can include both. Verify membership after workspace deletion.

### 28. Citation source identifiers collide

**Priority:** P2. **Evidence:** confirmed in source.

**Location:** [answer evidence metadata](src/processors/llm.py#L112).

**Problem:** source IDs are basenames. Two different `lecture.mp4` files get the same ID, and multiple retrieved chunks from one file are not individually identified. It is difficult to resolve a returned citation to one supplied evidence item and range.

**Suggested fix:** assign a unique evidence ID per supplied hit and maintain a Python-side registry mapping it to stable library/source/chunk IDs, timestamps, modality, and display filename. Have the answer model cite the evidence ID. Resolve the filename and persistent identity from that registry during rendering.

**Justification:** human-readable filenames remain useful labels, but unambiguous evidence identities are necessary for validation and playback.

**Verification:** supply two same-named files and multiple chunks from one source; every returned evidence ID must resolve to exactly one supplied hit.

### 29. Structured answers lack citation and evidence validation

**Priority:** P2. **Evidence:** validation gap confirmed; incorrect generated answers were not measured.

**Location:** [citation schema](src/processors/llm.py#L9), [parsed result return](src/processors/llm.py#L165).

**Problem:** the schema checks field types but does not establish that a source was supplied, a timestamp lies within its range, the start precedes the end, or the declared evidence type matches that source. A schema-valid claim can still be unsupported. For clarity, `evidence_type` already rejects strings outside `text` and `visual`; the missing check is their consistency with actual evidence.

**Suggested fix:** validate against the registry from issue 28. Check known evidence IDs; paired, finite, ordered timestamps; source bounds; modality; and sampled-frame timestamp constraints. Require factual answer blocks to carry citations and check sensible status/block combinations. Reject invalid output or perform a bounded repair request using the original evidence. Evaluate semantic claim support separately using the fixed query set and reviewed examples.

**Justification:** deterministic checks prevent invalid attribution, but they cannot prove a claim follows from the evidence. Structural validation and semantic evaluation address different failure modes.

**Verification:** deliberately return an unknown ID, mismatched modality, missing citation, reversed range, and out-of-range time; each must be detected. Review unsupported claims separately.

### 30. Workspace subcommands index absent arguments

**Priority:** P3. **Evidence:** confirmed in source.

**Location:** [`ws data`](src/CLI.py#L334), [`ws set`](src/CLI.py#L348).

**Problem:** both access `args[1]` without checking the argument count. The outer dispatcher catches the resulting exception but displays an internal `list index out of range` error instead of usage guidance.

**Suggested fix:** validate the exact supported argument count for every workspace subcommand before indexing. Return the relevant usage string for missing or extra arguments.

**Justification:** ordinary user mistakes should produce actionable feedback and should never reach internal indexing errors.

**Verification:** call every workspace subcommand with zero, correct, and excess arguments.

### 31. SQL workspace errors still suggest unrecognized commands

**Priority:** P3. **Evidence:** confirmed in source.

**Location:** [workspace lookup error](src/database/sql_db.py#L133), [source registration workspace error](src/database/sql_db.py#L194).

**Problem:** the terminal help and direct CLI print suggestions now describe the supported commands. However, two SQL helper exceptions still contain `Type /ws li to see all workspaces`, and their text can reach the terminal through the dispatcher. That command is not recognized. These exception messages were preserved during the output-only change; exception types and control flow were not modified.

**Suggested fix:** update the user-facing text of those two exceptions to suggest `ws ls`, preserving the exception types and raising conditions. If command definitions are centralized later, use them for both help and error suggestions without adding unsupported aliases.

**Justification:** error guidance should lead to a recognized recovery command. Changing these message strings does not require changing workspace behavior or parser rules.

**Verification:** trigger each missing-workspace error and confirm it suggests `ws ls`, then dispatch that command and verify it lists workspaces. Help coverage for the registered CLI commands has already been checked separately.

## Database helpers, setup, and diagnostics

### 32. Worker connections ignore a custom database path

**Priority:** P2 for custom-path use. **Evidence:** confirmed in source; the current CLI uses the default path.

**Location:** [constructor path](src/database/sql_db.py#L5), [hardcoded worker connection](src/database/sql_db.py#L11).

**Problem:** `SQLDB(path=...)` and `get_connection()` can open different databases. A worker may find no tables or process unrelated default-database records.

**Suggested fix:** retain the configured database target on `SQLDB` and use one connection factory for both main and worker connections, including consistent pragmas. Define explicit shared-memory behavior if `:memory:` is supported; separate `:memory:` connections do not share a database automatically.

**Justification:** all connections belonging to a database object must refer to the same store. Centralizing creation also prevents configuration differences between threads.

**Verification:** create a database at a nondefault temporary path and confirm records written through one connection are visible through the other.

### 33. Default initialization depends on a later unrelated commit

**Priority:** P3. **Evidence:** reproduced; primarily a helper transaction-contract issue.

**Location:** [default inserts](src/database/sql_db.py#L30).

**Problem:** `initialize_default()` does not commit its inserts. A separate connection sees no newly initialized settings until another method commits, and immediate closure rolls them back. The normal CLI later commits during setup/bootup, so this is not an independently demonstrated normal-startup failure.

**Suggested fix:** execute default initialization in an explicit transaction owned by the constructor/setup routine and commit before exposing the database to other components. Preserve idempotence and roll back the complete initialization unit if it fails.

**Justification:** initialization should have a predictable durable outcome without depending on later status updates, and should not retain an unnecessary open write transaction.

**Verification:** initialize, immediately close, reopen, and confirm defaults exist; another connection should see them immediately after initialization returns.

### 34. Frame sampling does not validate interval or range

**Priority:** P2 for helper safety. **Evidence:** lack of validation confirmed in source. The current CLI passes a valid fixed interval of 5 seconds, but the helper still accepts invalid intervals/ranges and has no frame cap.

**Location:** [helper arguments](src/processors/raw_process.py#L68), [loop progression](src/processors/raw_process.py#L107).

**Problem:** a zero interval repeatedly samples the same readable frame without advancing and can run indefinitely. A negative interval moves away from the termination boundary, although decoder failure may stop it first. Invalid/nonfinite time values are also unchecked.

**Suggested fix:** reject nonfinite values, require `interval > 0` and `0 <= start < end`, enforce a maximum frame count, and define handling for ranges beyond the media duration. Reject bad input before opening the capture when possible.

**Justification:** loop termination should follow from validated inputs rather than rely on a decoder eventually failing. A frame cap also bounds memory and API request size.

**Verification:** cover zero, negative, infinite, and NaN intervals; reversed/negative ranges; and a valid range whose samples reach the configured cap.

## Suggested implementation sequence

1. Prevent unrelated-file deletion first: issue **5**.
2. Add frame sampling/range guards and a frame cap: **34**.
3. Establish stable identities and recoverable persistence: **6–8**, **11**, **28**, **32–33**. Plan legacy-vector migration/reindexing explicitly.
4. Make worker and model ownership reliable: **9–15**, **26**. Handle failures and shutdown before relying on concurrent chat/ingestion.
5. Align provider, model, and credential configuration: **16–23**.
6. Resolve search scope and enforce citation checks: **27**, **29**. Record the workspace/global-search product choice explicitly.
7. Correct command parsing and the remaining SQL error guidance: **24–25**, **30–31**.

The verification steps above are proposed acceptance checks, not claims that those checks have already passed. After these fixes, run a controlled MP4 ingestion/query smoke test, then evaluate retrieval and answers separately against the existing [transcript query set](notebooks/rag_test_queries_README.md). Add reviewed visual questions because transcript-only cases cannot establish visual evidence accuracy.

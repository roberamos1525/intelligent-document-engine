# Security

## Reporting a vulnerability

Please report suspected vulnerabilities privately to the maintainer (Robert Ramos) rather than in
a public issue. Include steps to reproduce and the affected version. You will receive an
acknowledgement and a fix or mitigation plan.

## Threat model

The engine is designed to run as an internal service on a trusted network or a single machine.
Callers are authenticated with API keys; uploaded documents are treated as untrusted input.

## Controls implemented

**Authentication and authorisation**
- API keys with `reader`, `writer` and `admin` roles; the server refuses to start when
  authentication is enabled without keys.
- Keys are compared as SHA-256 digests with `hmac.compare_digest`; logs and rate-limit buckets
  use a 12-character fingerprint, never the key.
- Per-key token-bucket rate limiting with `Retry-After`.

**Untrusted uploads**
- File type decided by magic bytes; executables (PE, ELF, Mach-O, scripts), images, archives,
  legacy OLE files and non-DOCX ZIP containers are rejected. A binary extension that does not
  match the content is rejected.
- Size limits on the request body (checked from `Content-Length` and while streaming) and on the
  file itself (the upload is read in 1 MiB pieces and aborted past the limit).
- DOCX: total uncompressed size and compression ratio are checked before parsing (zip-bomb
  defence). PDF: page limit, password-protected files rejected, parser exceptions contained.
- HTML is parsed with the standard library parser; scripts and other active content are dropped,
  nothing is executed or fetched.
- Client file names are sanitised (no directories or control characters); blobs are stored under
  their SHA-256, so names never reach the filesystem. Blob keys are validated to prevent path
  traversal.
- Original files are served back only as attachments with `X-Content-Type-Options: nosniff`.

**Input validation**
- Pydantic schemas forbid unknown fields; queries, filters (operator whitelist, depth limit,
  list sizes), tags, categories, titles, training sets and inline texts are bounded.
- SQL uses parameters exclusively; sortable columns are whitelisted.

**Secrets and configuration**
- Secrets come from the environment (`.env` is git-ignored; `.env.example` holds placeholders).
- The LLM API key is a `SecretStr`; `idengine config` and diagnostics mask keys.
- External LLM adapters are disabled by default; when enabled they use bounded timeouts and only
  send the question and the retrieved passages.

**Runtime**
- Docker image runs as a non-root user; compose uses a read-only root filesystem and
  `no-new-privileges` (not verified locally, Docker was unavailable).
- Errors returned to clients never include stack traces; unexpected exceptions are logged
  server-side with the request id.

## Known gaps

- No TLS termination: run behind a reverse proxy (nginx, Caddy, a cloud load balancer).
- API keys are static configuration: no rotation endpoint, expiry or per-document access control.
  Any reader can read every document.
- No audit log of who uploaded, read or deleted what (request logs contain key fingerprints only).
- Rate limiting is in-process and per instance.
- Data at rest (SQLite, blobs, indexes) is not encrypted by the application; use disk encryption.
- Document content is passed to an external LLM if (and only if) an HTTP adapter is configured.

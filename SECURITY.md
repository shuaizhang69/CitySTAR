# Security Policy

Do not report private API keys, tokens, datasets, or server credentials in
public issues.

This project expects secrets to be supplied through environment variables:

```text
DEEPSEEK_API_KEY
OPENAI_API_KEY
DASHSCOPE_API_KEY
SGLANG_BASE_URL
```

Before publishing or sharing a bundle, run:

```bash
python scripts/verify_bundle.py
```

The verifier checks for Python bytecode caches and `sk-*` token-like strings in
source-like files. It is a guardrail, not a substitute for reviewing private
paths and credentials manually.

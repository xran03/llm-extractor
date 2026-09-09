# Credentials

## Credentials

### The easy way: paste it once

If you do not want to edit configuration files, run:

```bash
llm-extract login
```

It asks for the gateway URL and then for your API key (**input is hidden**),
checks the key against the gateway, and only saves it if it actually works — so
a mistyped or expired key is reported immediately rather than halfway through a
run. From then on, just extract:

```bash
llm-extract -i ./docs -o ./out
```

The key is written to a file that only your account can read
(`0600`, inside a `0700` directory):

| Platform | Location |
|---|---|
| Windows | `%APPDATA%\llm-extractor\credentials.json` |
| macOS / Linux | `$XDG_CONFIG_HOME` (or `~/.config`) `/llm-extractor/credentials.json` |

It is kept outside your project folder on purpose, so a key can never be swept
into a commit. To remove it again:

```bash
llm-extract logout            # forget the selected backend
llm-extract logout --all      # forget every backend
```

### The file way: `.env`

For servers, CI and shared installations, copy `.env.example` to `.env` and fill
in the backend you use:

```ini
LLM_HUB_BASE_URL=https://your-gateway
LLM_HUB_API_KEY=...
# or OAuth2 client credentials, which mint a short-lived token:
LLM_HUB_CLIENT_ID=...
LLM_HUB_CLIENT_SECRET=...
LLM_HUB_TOKEN_URL=...
```

Resolution order, first hit wins:

`--api-key` → environment variable → nearest `.env` → saved login → paste prompt

A saved login therefore never overrides an explicit flag, a real environment
variable or a `.env` file, which keeps servers and CI behaving exactly as
before. To paste a one-off key without saving it, pass `-`:

```bash
llm-extract -i ./docs -o ./out --api-key -      # prompts, input hidden
```

Verify everything before spending tokens:

```bash
llm-extract check
```


---

[Back to the README](../README.md)

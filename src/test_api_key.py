"""Send one minimal request to the OpenAI API to confirm the configured key works.

Usage:
    python src/test_api_key.py

The API key is loaded via config.py and is never printed or logged.
"""

import sys

from openai import OpenAI

from config import OPENAI_API_KEY

MODEL = "gpt-4o-mini"


def main() -> None:
    client = OpenAI(api_key=OPENAI_API_KEY)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Reply with exactly: API key works"}],
            max_tokens=10,
        )
    except Exception as exc:  # noqa: BLE001 - surface any failure clearly to the user
        print("ERROR: The test request to the OpenAI API failed.", file=sys.stderr)
        print(f"Reason: {exc}", file=sys.stderr)
        print(
            "Check that your key in .env is valid and that you have network access.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Model response:", response.choices[0].message.content)


if __name__ == "__main__":
    main()

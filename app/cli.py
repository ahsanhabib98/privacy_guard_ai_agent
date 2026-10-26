from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.main import create_agent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="privacyguard",
        description="PrivacyGuard command-line interface",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    subparsers.add_parser(
        "status",
        help="Check PrivacyGuard status.",
    )

    process_parser = subparsers.add_parser(
        "process",
        help="Process sensitive text.",
    )

    input_group = process_parser.add_mutually_exclusive_group(
        required=True
    )

    input_group.add_argument(
        "--text",
        type=str,
        help="Text to process.",
    )

    input_group.add_argument(
        "--file",
        type=str,
        help="Input text file.",
    )

    process_parser.add_argument(
        "--scope",
        type=str,
        default="default",
        help="Transformation scope.",
    )

    process_parser.add_argument(
        "--output",
        type=str,
        default="output.txt",
        help="Output file for transformed text.",
    )

    subparsers.add_parser(
        "memory-stats",
        help="Show memory statistics.",
    )

    return parser


async def handle_status() -> None:
    create_agent()

    print(
        json.dumps(
            {
                "status": "ok",
                "message": "PrivacyGuard initialized successfully.",
            },
            indent=2,
        )
    )


async def handle_process(
    text: str,
    scope: str,
    output_file: str,
) -> None:
    agent = create_agent()

    response = await agent.process(
        text=text,
        scope=scope,
    )

    # Show complete response in terminal
    print(response.model_dump_json(indent=2))

    # Save transformed text to output file
    if response.transformed_text is not None:
        Path(output_file).write_text(
            response.transformed_text,
            encoding="utf-8",
        )

        print(f"\nTransformed text saved to: {output_file}")

    else:
        Path(output_file).write_text(
            "PrivacyGuard did not release transformed text because validation failed.",
            encoding="utf-8",
        )

        print(
            f"\nNo transformed text was released. "
            f"Validation result saved to: {output_file}"
        )


async def handle_memory_stats() -> None:
    agent = create_agent()

    stats = agent.memory.stats()

    if hasattr(stats, "model_dump_json"):
        print(stats.model_dump_json(indent=2))
    else:
        print(json.dumps(stats, indent=2))


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "status":
        await handle_status()
        return

    if args.command == "process":

        if args.text is not None:
            text = args.text

        else:
            text = Path(args.file).read_text(
                encoding="utf-8"
            )

        await handle_process(
            text=text,
            scope=args.scope,
            output_file=args.output,
        )

        return

    if args.command == "memory-stats":
        await handle_memory_stats()
        return


if __name__ == "__main__":
    asyncio.run(main())
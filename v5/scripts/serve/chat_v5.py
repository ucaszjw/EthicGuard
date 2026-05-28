#!/usr/bin/env python3
"""EthicGuard v5 — 多轮对话终端 (流式输出)"""
import sys, json
sys.path.insert(0, "/root/train/v5")
from scripts.serve.safe_generator_v5 import SafeGenerator

gen = SafeGenerator()
history = []

print("\n" + "="*60)
print("  EthicGuard v5 — 多轮对话终端 (双分类器融合)")
print("  输入 exit 退出")
print("="*60 + "\n")

while True:
    try:
        user = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye.")
        break

    if not user:
        continue
    if user.lower() in ("exit", "quit"):
        print("Bye.")
        break

    sys.stdout.write("Assistant: ")
    sys.stdout.flush()

    result = None
    for item in gen.generate_stream(user, history=history,
                                     max_new_tokens=1024, temperature=0.3):
        if isinstance(item, dict):
            result = item
        else:
            # Strip any think tags
            clean = item.replace('<think>', '').replace('</think>', '')
            if clean:
                sys.stdout.write(clean)
                sys.stdout.flush()

    if result:
        history = result.get("history", history)
        if result.get("truncated"):
            print("\n\n  ⛔ 实时阻断 — 原因: " + result.get("reason", "unknown"))
    print()

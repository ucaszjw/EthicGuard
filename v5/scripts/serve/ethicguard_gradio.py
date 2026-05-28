"""
EthicGuard v5 Gradio — 实时流式安全监控
阻断轮次不计入历史，避免污染后续生成。
"""
from __future__ import annotations
import argparse, gradio as gr
from scripts.serve.safe_generator_v5 import SafeGenerator

SAFE_REFUSAL = "抱歉，我无法生成此内容。请提出其他合规的问题。"

def build_app(gen: SafeGenerator):
    def respond(message, chat_history, clean_history):
        message = (message or "").strip()
        if not message:
            yield "", chat_history, clean_history, {}
            return

        # Use CLEAN history for model (blocked turns excluded)
        hist_dict = list(clean_history) if clean_history else []

        # Update chat display
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": ""})
        yield "", chat_history, clean_history, {}

        full_reply = ""
        final = None
        for item in gen.generate_stream(message, history=hist_dict,
                                         max_new_tokens=256, temperature=0.3):
            if isinstance(item, dict):
                final = item
            else:
                full_reply += item
                chat_history[-1]["content"] = full_reply
                yield "", chat_history, clean_history, {}

        if final and final.get("truncated"):
            reason = final.get("reason", "unknown")
            display = full_reply + "\n\n⛔ 实时阻断\n原因: " + reason if full_reply else "⛔ 实时阻断\n原因: " + reason
            chat_history[-1]["content"] = display
            # Don't update clean_history — blocked turns excluded from future context
        else:
            clean_history.append({"role": "user", "content": message})
            clean_history.append({"role": "assistant", "content": full_reply})

        status = {
            "blocked": final.get("truncated", False) if final else False,
            "reason": final.get("reason", "") if final else "",
            "steps": len(final.get("safety_log", [])) if final else 0,
            "log": final.get("safety_log", [])[:20] if final else [],
        }
        yield "", chat_history, clean_history, status

    def clear_fn():
        return [], [], {}

    with gr.Blocks(title="EthicGuard v5 - 实时流式安全监控") as demo:
        gr.Markdown(
            "# EthicGuard v5 — 双分类器融合实时阻断\n"
            "模型逐 token 流式生成，每步提取**深层隐状态**，"
            "v3+v4 双分类器联合判定。阻断轮次不污染后续对话。"
        )
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=520, label="对话")
                with gr.Row():
                    text = gr.Textbox(placeholder="输入内容…", lines=2,
                                      label="用户输入", scale=4)
                    send = gr.Button("发送", variant="primary", scale=1)
                clear_btn = gr.Button("清空")
            with gr.Column(scale=2):
                status = gr.JSON(label="阻断状态", value={})
                gr.Markdown(
                    "**状态说明**\n"
                    "- `blocked: true` → 生成被实时阻断\n"
                    "- `reason` → 触发类别和置信度\n"
                    "- `log` → 每步判定记录\n"
                    "---\n"
                    "**v5 特点**\n"
                    "- v3（安全门）+ v4（分类器）双模型\n"
                    "- 共享隐状态，零额外开销\n"
                    "- 仅两者同时判定风险才阻断\n"
                    "- **阻断轮次不计入历史**"
                )

        clean_state = gr.State([])
        send.click(respond, [text, chatbot, clean_state],
                   [text, chatbot, clean_state, status])
        text.submit(respond, [text, chatbot, clean_state],
                    [text, chatbot, clean_state, status])
        clear_btn.click(clear_fn, None, [chatbot, clean_state, status])
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    gen = SafeGenerator()
    demo = build_app(gen)
    demo.queue().launch(server_name=args.host, server_port=args.port, share=False)

if __name__ == "__main__":
    main()

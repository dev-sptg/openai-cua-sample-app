from utils import strip_model_only_items


def test_strip_model_only_items():
    items = [
        {"type": "input_text", "text": "hello"},
        {"type": "reasoning", "content": "private"},
        {"type": "tool_result", "name": "computer", "result": {"ok": True}},
        {"type": "output_text", "text": "final"},
    ]
    got = strip_model_only_items(items)
    assert {"type": "input_text", "text": "hello"} in got
    assert {
        "type": "tool_result",
        "name": "computer",
        "result": {"ok": True},
    } in got
    assert not any(i.get("type") == "reasoning" for i in got)
    assert not any(i.get("type") == "output_text" for i in got)

